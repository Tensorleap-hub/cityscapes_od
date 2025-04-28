import os
import numpy as np
import tensorflow as tf

from cityscapes_od.data.preprocess import CATEGORIES_no_background, CATEGORIES_id_no_background, Cityscapes
from cityscapes_od.metrics import od_loss
from cityscapes_od.utils.general_utils import get_json, get_polygon
from cityscapes_od.utils.plots import plot_image_with_polygons, plot_image_with_bboxes
from leap_binder import load_cityscapes_data_leap, ground_truth_bbox, \
    od_metrics_dict, gt_bb_decoder, bb_decoder, bb_car_decoder, bb_car_gt_decoder, metadata_filename, metadata_city, \
    metadata_idx, metadata_brightness, metadata_json, metadata_category_avg_size, metadata_bbs, label_instances_num, \
    is_class_exist_gen, is_class_exist_veg_and_building, get_class_mean_iou, encode_image, bus_bbox_cnt_pred
from os import environ
from leap_binder import leap_binder
from code_loader.helpers import visualize

def check_custom_integration():
    plot_vis = True
    check_generic = True

    if check_generic:
        leap_binder.check()

    if environ.get('AUTH_SECRET') is None:
        print("The AUTH_SECRET system variable must be initialized with the relevant secret to run this test")
        exit(-1)
    print("started custom tests")
    # preprocess function
    responses = load_cityscapes_data_leap()
    train = responses[0]
    val = responses[1]
    test = responses[2]
    responses_set = train
    for idx in range(3):
        # model
        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = 'model/CSYolov7Retrained.h5'
        yolo = tf.keras.models.load_model(os.path.join(dir_path, model_path))

        # get input and gt
        image = encode_image(idx, responses_set)
        bounding_boxes_gt = ground_truth_bbox(idx, responses_set)

        json_data = get_json(idx, responses_set)
        image_height, image_width = json_data['imgHeight'], json_data['imgWidth']
        polygons = get_polygon(json_data)

        concat = np.expand_dims(image, axis=0)
        y_pred = yolo([concat])
        gt = np.expand_dims(bounding_boxes_gt, axis=0)

        # get visualizer
        bb_gt_decoder = gt_bb_decoder(concat, gt)
        bb__decoder = bb_decoder(concat, y_pred.numpy())
        bb_gt_car = bb_car_gt_decoder(concat, gt)
        bb_car = bb_car_decoder(concat, y_pred.numpy())

        if plot_vis:
            visualize(bb_gt_decoder, 'gt')
            visualize(bb__decoder, 'pred')
            visualize(bb_car, 'pred')
            visualize(bb_gt_car, 'gt')

            # plot_image_with_bboxes(image, bb_gt_decoder.bounding_boxes, 'gt')
            # plot_image_with_bboxes(image, bb__decoder.bounding_boxes, 'pred')
            # plot_image_with_bboxes(image, bb_car.bounding_boxes, 'pred')
            # plot_image_with_bboxes(image, bb_gt_car.bounding_boxes, 'gt')
            plot_image_with_polygons(image_height, image_width, polygons, image)

        # get loss and custom metrics
        ls = od_loss(gt, y_pred.numpy())
        metrics_all = od_metrics_dict(gt, y_pred.numpy())
        bus_bbox = bus_bbox_cnt_pred(y_pred.numpy())
        for id in CATEGORIES_id_no_background:
            iou_func = get_class_mean_iou(id)
            iou = iou_func(gt, y_pred.numpy())

        # get custom meta data
        filename = metadata_filename(idx, responses_set)
        city = metadata_city(idx, responses_set)
        idx = metadata_idx(idx, responses_set)
        brightness = metadata_brightness(idx, responses_set)
        json = metadata_json(idx, responses_set)
        category_avg_size = metadata_category_avg_size(idx, responses_set)
        bbs = metadata_bbs(idx, responses_set)
        for label in CATEGORIES_no_background:
            instances_count_func = label_instances_num(label)
            instance_count = instances_count_func(idx, responses_set)
        for id in CATEGORIES_id_no_background:
            class_name = Cityscapes.get_class_name(id)
            class_exist_gen_func = is_class_exist_gen(id)
            class_exist_gen = class_exist_gen_func(idx, responses_set)
        class_exist_veg_func = is_class_exist_veg_and_building(21, 11)
        class_exist_veg = class_exist_veg_func(idx, responses_set)


    print("Custom tests finished successfully")

if __name__ == '__main__':
    check_custom_integration()



