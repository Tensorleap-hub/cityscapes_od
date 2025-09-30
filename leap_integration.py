import os
import numpy as np
import tensorflow as tf
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.plot_functions.visualize import visualize

from cityscapes_od.config import CONFIG
from cityscapes_od.data.preprocess import CATEGORIES
from cityscapes_od.metrics import od_loss
from leap_binder import (load_cityscapes_data_leap, ground_truth_bbox, \
                         od_metrics_dict, gt_bb_decoder, bb_decoder, bb_car_decoder, bb_car_gt_decoder,
                         encode_image, iou_dic, bus_bbox_cnt_pred, leap_binder, misc_metadata, metadata_json,
                         gt_all_bbox_count, metadata_category_avg_size, metadata_bbs)
from os import environ
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, integration_test



prediction_type1 = PredictionTypeHandler('object detection', ["x", "y", "w", "h", "obj"] + [cl for cl in CATEGORIES])

@tensorleap_load_model([prediction_type1])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/CSYolov7.h5'
    return tf.keras.models.load_model(os.path.join(dir_path, model_path))

@integration_test()
def check_custom_test(idx, responses_set):
    yolo = load_model()

    image = encode_image(idx, responses_set)
    bounding_boxes_gt = ground_truth_bbox(idx, responses_set)
    y_pred = yolo([image])

    # vis
    bb_gt_decoder = gt_bb_decoder(image, bounding_boxes_gt)
    bb__decoder = bb_decoder(image, y_pred)
    bb_car = bb_car_decoder(image, y_pred)
    bb_gt_car = bb_car_gt_decoder(image, bounding_boxes_gt)

    visualize(bb_gt_decoder)
    visualize(bb__decoder)
    visualize(bb_car)
    visualize(bb_gt_car)

    # get loss and custom metrics
    ls = od_loss(bounding_boxes_gt, y_pred)
    metrics_all = od_metrics_dict(bounding_boxes_gt, y_pred)
    iou = iou_dic(bounding_boxes_gt, y_pred)
    bus_bbox = bus_bbox_cnt_pred(y_pred)

    misc_metadata(idx, responses_set)
    metadata_json(idx, responses_set)
    gt_all_bbox_count(idx, responses_set)
    metadata_category_avg_size(idx, responses_set)
    metadata_bbs(idx, responses_set)




if __name__ == '__main__':
    responses = load_cityscapes_data_leap()
    train = responses[0]
    check_custom_test(0, train)