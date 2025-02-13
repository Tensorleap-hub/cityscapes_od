import os
import numpy as np
import tensorflow as tf

from cityscapes_od.config import CONFIG
from cityscapes_od.metrics import od_loss
from leap_binder import (load_cityscapes_data_leap, ground_truth_bbox, \
    od_metrics_dict, gt_bb_decoder, bb_decoder, bb_car_decoder, bb_car_gt_decoder,
                         encode_image, iou_dic, bus_bbox_cnt_pred, leap_binder)
from os import environ
from code_loader.helpers.visualizer.visualize import visualize

def check_custom_integration():
    if not CONFIG['USE_LOCAL'] and environ.get('AUTH_SECRET') is None:
        print("The AUTH_SECRET system variable must be initialized with the relevant secret to run this test")
        exit(-1)
    print("started custom tests")
    # preprocess function
    check_generic = True
    plot_vis = False
    if check_generic:
        leap_binder.check()
    print("started custom tests")
    responses = load_cityscapes_data_leap()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = ('model/CSYolov7.h5')
    yolo = tf.keras.models.load_model(os.path.join(dir_path, model_path))
    for responses_set in responses:
        for idx in range(10):
            # get input and gt
            image = encode_image(idx, responses_set)
            bounding_boxes_gt = ground_truth_bbox(idx, responses_set)
            concat = np.expand_dims(image, axis=0)
            y_pred = yolo([concat]).numpy()
            gt = np.expand_dims(bounding_boxes_gt, axis=0)
            bb_gt_decoder = gt_bb_decoder(image, gt)
            bb__decoder = bb_decoder(image, y_pred[0, ...])
            bb_car = bb_car_decoder(image, y_pred[0, ...])
            bb_gt_car = bb_car_gt_decoder(image, gt)
            if plot_vis:
                visualize(bb_gt_decoder)
                visualize(bb__decoder)
                visualize(bb_car)
                visualize(bb_gt_car)
            # get loss and custom metrics
            ls = od_loss(gt, y_pred)
            metrices_all = od_metrics_dict(gt, y_pred)
            iou = iou_dic(gt, y_pred)
            bus_bbox = bus_bbox_cnt_pred(y_pred)
            for metadata_handler in leap_binder.setup_container.metadata:
                curr_metadata = metadata_handler.function(idx, responses_set)
                print(f"Metadata {metadata_handler.name}: {curr_metadata}")

    print("Custom tests finished successfully")


if __name__ == '__main__':
    check_custom_integration()
