from typing import Dict, Callable, Union, Tuple
from PIL import Image
import json
from typing import List
import numpy as np
import tensorflow as tf

from cityscapes_od.config import CONFIG
from cityscapes_od.data.preprocess import load_cityscapes_data, CATEGORIES, CATEGORIES_no_background, \
    CATEGORIES_id_no_background, Cityscapes
from cityscapes_od.metrics import calculate_iou, od_loss, calc_od_losses
from cityscapes_od.utils.gcs_utils import _download
from cityscapes_od.utils.general_utils import extract_bounding_boxes_from_instance_segmentation_polygons, \
    bb_array_to_object, get_predict_bbox_list, instances_num, avg_bb_aspect_ratio, avg_bb_area_metadata, \
    count_small_bbs, number_of_bb, filter_out_unknown_classes_id

from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import (
    LeapDataType, MetricDirection
)
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_gt_encoder,
                                                                 tensorleap_input_encoder, tensorleap_metadata,
                                                                 tensorleap_custom_visualizer, tensorleap_custom_metric)

# ----------------------------------------------------data processing--------------------------------------------------
@tensorleap_preprocess()
def load_cityscapes_data_leap() -> List[PreprocessResponse]:
    all_images, all_gt_images, all_gt_labels, all_gt_labels_for_bbx, all_file_names, all_metadata, all_cities = \
        load_cityscapes_data()

    train_len = 700
    val_len = 100
    test_len = 200

    lengths = [train_len, val_len, test_len]
    responses = [
        PreprocessResponse(length=lengths[i], data={
            "image_path": all_images[i],
            "subset_name": ["train", "val", "test"][i],
            "gt_path": all_gt_labels[i],
            "gt_bbx_path": all_gt_labels_for_bbx[i],
            "gt_image_path": all_gt_images[i],
            "real_size": lengths[i],
            "file_names": all_file_names[i],
            "cities": all_cities[i],
            "metadata": all_metadata[i],
            "dataset": ["cityscapes_od"] * lengths[i]
        }) for i in range(3)
    ]
    return responses


# ------------------------------------------input and gt------------------------------------------
@tensorleap_input_encoder('image',channel_dim=-1)
def encode_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['image_path'][idx]
    fpath = _download(str(cloud_path))
    img = np.array(Image.open(fpath).convert('RGB').resize(CONFIG['IMAGE_SIZE'])) / 255.
    return img.astype(np.float32)

@tensorleap_gt_encoder('bbox')
def ground_truth_bbox(idx: int, data: PreprocessResponse) -> np.ndarray:
    """
    Description: This function takes an integer index idx and a PreprocessResponse object data as input and returns an
                 array of bounding boxes representing ground truth annotations.

    Input: idx (int): sample index.
    data (PreprocessResponse): An object of type PreprocessResponse containing data attributes.
    Output: bounding_boxes (np.ndarray): An array of bounding boxes extracted from the instance segmentation polygons in
            the JSON data. Each bounding box is represented as an array containing [x_center, y_center, width, height, label].
    """
    data = data.data
    cloud_path = data['gt_bbx_path'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as file:
        json_data = json.load(file)
    bounding_boxes = extract_bounding_boxes_from_instance_segmentation_polygons(json_data)
    return bounding_boxes.astype(np.float32)


# ----------------------------------------------------------metadata----------------------------------------------------
@tensorleap_metadata("misc")
def misc_metadata(idx: int, data: PreprocessResponse) -> Dict[str, Union[str, int]]:
    img = encode_image(idx, data)
    misc_dict = {
        "filename": data.data['file_names'][idx],
        "city": data.data['cities'][idx],
        "idx": idx,
        "brightness": np.mean(img)
     }
    return misc_dict

def get_metadata_json(idx: int, data: PreprocessResponse) -> Dict[str, str]:
    cloud_path = data.data['metadata'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as f:
        metadata_dict = json.loads(f.read())
    return metadata_dict

@tensorleap_metadata("metadata_json")
def metadata_json(idx: int, data: PreprocessResponse):
    json_dict = get_metadata_json(idx, data)
    res = {
        "gps_heading": json_dict['gpsHeading'],
        "gps_latitude": json_dict['gpsLatitude'],
        "gps_longtitude": json_dict['gpsLongitude'],
        "outside_temperature": json_dict['outsideTemperature'],
        "speed": json_dict['speed'],
        "yaw_rate": json_dict['yawRate']
    }
    return res


#
@tensorleap_metadata("gt_all_bboxes_counts")
def gt_all_bbox_count(idx: int, data: PreprocessResponse):
    data = data.data
    cloud_path = data['gt_bbx_path'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as file:
        json_data = json.load(file)
    objects = filter_out_unknown_classes_id(json_data['objects'])
    return len(objects)


def category_percent(idx: int, data: PreprocessResponse, class_id: int) -> Tuple[np.ndarray, float]:
    bbs = np.array(ground_truth_bbox(idx, data))
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    category_bbs = valid_bbs[valid_bbs[..., -1] == class_id]
    return bbs, float(category_bbs.shape[0])


def category_avg_size(idx: int, data: PreprocessResponse, class_id: int) -> float:
    bbs, car_val = category_percent(idx, data, class_id)
    instances_cnt = number_of_bb(bbs)
    return np.float32(np.round(car_val / instances_cnt, 3) if instances_cnt > 0 else 0)

@tensorleap_metadata("metadata_category_avg_size")
def metadata_category_avg_size(idx: int, data: PreprocessResponse) -> Dict[str, float]:
    res = {
        "metadata_person_category_avg_size": np.float32(category_avg_size(idx, data, 24)),
        "metadata_car_category_avg_size": np.float32(category_avg_size(idx, data, 26))
    }
    return res


@tensorleap_metadata("metadata_bbs")
def metadata_bbs(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    bboxes = np.array(ground_truth_bbox(idx, data))
    valid_bbs = bboxes[bboxes[..., -1] != CONFIG['BACKGROUND_LABEL']]
    res = {
        "instances_number": int(instances_num(valid_bbs)),
        "bb_aspect_ratio": avg_bb_aspect_ratio(valid_bbs),
        "avg_bb_area": avg_bb_area_metadata(valid_bbs),
        "small_bbs": count_small_bbs(bboxes),
        "bbox_number": number_of_bb(bboxes)

    }
    for c_label in CATEGORIES_no_background:
        label = CATEGORIES.index(c_label)
        valid_bbs = bboxes[bboxes[..., -1] == label]
        num_bbs = valid_bbs.shape[0]
        res[f"{c_label}_count"] = num_bbs
        res[f"does_{c_label}_exist"] = num_bbs > 0
    class_id_veg = 21
    class_id_building = 11
    is_veg_exist = (bboxes[..., -1] == class_id_veg).any()
    is_building_exist = (bboxes[..., -1] == class_id_building).any()
    res['veg_building_exist'] = is_veg_exist and is_building_exist
    return res

# ---------------------------------------------------------metrics------------------------------------------------------

# set custom metrics
def class_mean_iou(y_true: tf.Tensor, y_pred: tf.Tensor, class_id: int) -> np.ndarray:
    iou = calculate_iou(y_true, y_pred, class_id)
    return np.array([iou], dtype=np.float32)

@tensorleap_custom_metric("ious", direction=MetricDirection.Upward)
def iou_dic(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[float, int]]:

    if len(y_true.shape) == 2:
        y_true = tf.expand_dims(y_true, 0)
    else:
        y_true = tf.convert_to_tensor(y_true)
    if len(y_pred.shape) == 2:
        y_pred = tf.expand_dims(y_pred, 0)
    else:
        y_pred = tf.convert_to_tensor(y_pred)
    res_dic = dict()
    mean_iou = list()
    for c_id in CATEGORIES_id_no_background:
        class_name = Cityscapes.get_class_name(c_id)
        res = class_mean_iou(y_true, y_pred, c_id)
        res_dic[f"{class_name}"] = res
        if tf.reduce_sum(res) > 0:
            mean_iou += [res]  # todo multiply by pixels
    res_dic["meanIOU"] = np.mean(mean_iou, axis=0) if len(mean_iou) > 0 else np.zeros(shape=(1,))
    return res_dic


#leap_binder.add_custom_metric(od_metrics_dict, 'od_metrics')
@tensorleap_custom_metric('od_metrics_dict', direction=MetricDirection.Downward)
def od_metrics_dict(bb_gt: np.ndarray, detection_pred: np.ndarray) -> Dict[str, np.ndarray]:
    losses = calc_od_losses(bb_gt, detection_pred)
    metric_functions = {
        "Regression_metric": losses[0],
        "Classification_metric": losses[1],
        "Objectness_metric": losses[2],
    }
    return metric_functions

@tensorleap_custom_metric("bus_cnt_bbox_pred", direction=MetricDirection.Upward)
def bus_bbox_cnt_pred(predictions: tf.Tensor) -> int:
    if len(predictions.shape) == 2:
        predictions = tf.expand_dims(predictions, 0)
    bb_object = get_predict_bbox_list(predictions[0, ...])
    bb_object = [bbox for bbox in bb_object if bbox.label == 'bus']
    return np.array(len(bb_object))[np.newaxis]

# ------------------------------------------------------visualizers---------------------------------------------------

@tensorleap_custom_visualizer("bb_gt_decoder", LeapDataType.ImageWithBBox)
def gt_bb_decoder(image: np.ndarray, bb_gt: np.ndarray) -> LeapImageWithBBox:
    """
    This function overlays ground truth bounding boxes (BBs) on the input image.

    Parameters:
    image (np.ndarray): The input image for which the ground truth bounding boxes need to be overlaid.
    bb_gt (np.ndarray): The ground truth bounding box array for the input image.

    Returns:
    An instance of LeapImageWithBBox containing the input image with ground truth bounding boxes overlaid.
    """
    image = np.squeeze(image)
    bb_object: List[BoundingBox] = bb_array_to_object(bb_gt, iscornercoded=False, bg_label=CONFIG['BACKGROUND_LABEL'],
                                                      is_gt=True)
    bb_object = [bbox for bbox in bb_object if bbox.label in CATEGORIES_no_background]
    return LeapImageWithBBox(data=(image * 255).astype(np.uint8), bounding_boxes=bb_object)

@tensorleap_custom_visualizer("bb_car_gt_decoder", LeapDataType.ImageWithBBox)
def bb_car_gt_decoder(image: np.ndarray, bb_gt: np.ndarray) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    image = np.squeeze(image)
    bb_object: List[BoundingBox] = bb_array_to_object(bb_gt, iscornercoded=False, bg_label=CONFIG['BACKGROUND_LABEL'],
                                                      is_gt=True)
    bb_object = [bbox for bbox in bb_object if bbox.label == 'car']
    return LeapImageWithBBox(data=(image * 255).astype(np.uint8), bounding_boxes=bb_object)

@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, predictions: np.ndarray) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    image = np.squeeze(image)
    predictions = np.squeeze(predictions)
    bb_object = get_predict_bbox_list(predictions)
    bb_object = [bbox for bbox in bb_object if bbox.label in CATEGORIES_no_background]
    return LeapImageWithBBox(data=(image * 255).astype(np.uint8), bounding_boxes=bb_object)

@tensorleap_custom_visualizer("bb_car_decoder", LeapDataType.ImageWithBBox)
def bb_car_decoder(image: np.ndarray, predictions: np.ndarray) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    image = np.squeeze(image)
    predictions = np.squeeze(predictions)
    bb_object = get_predict_bbox_list(predictions)
    bb_object = [bbox for bbox in bb_object if bbox.label == 'car']
    return LeapImageWithBBox(data=(image * 255).astype(np.uint8), bounding_boxes=bb_object)

# ---------------------------------------------------------binding------------------------------------------------------
# set prediction

if __name__ == '__main__':
    leap_binder.check()
