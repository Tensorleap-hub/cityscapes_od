from typing import Dict
import numpy as np
import tensorflow as tf

from code_loader import leap_binder
from code_loader.helpers.detection.yolo.utils import jaccard, xywh_to_xyxy_format
from code_loader.helpers.detection.utils import xyxy_to_xywh_format
from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue

from cityscapes_od.utils.yolo_utils import DECODER
from cityscapes_od.data.preprocess import Cityscapes, CATEGORIES_id_no_background
from cityscapes_od.config import CONFIG


def confusion_matrix_metric(gt, cls, reg, image):
    # assumes we get predictions in xyxy format in gt AND reg
    # assumes gt is in xywh form
    reg, cls = tf.transpose(reg, (0, 2, 1)), tf.transpose(cls, (0, 2, 1))
    id_to_name: Dict[str, str] = dict({c_id: Cityscapes.get_class_name(c_id) for c_id in CATEGORIES_id_no_background})  # CONFIG['class_id_to_name']
    threshold = CONFIG['CM_IOU_THRESH']
    image_shape = CONFIG['IMAGE_SIZE']
    reg_fixed = xyxy_to_xywh_format(reg)
    reg_normalized = tf.concat([
        (reg_fixed[:, :, :2] - reg_fixed[:, :,  2:] / 2) / image_shape[::-1],  # Normalized (x1, y1)
        reg_fixed[:, :, 2:] / image_shape  # Normalized (w, h)
    ], axis=2)
    outputs = DECODER(loc_data=[reg_normalized], conf_data=[cls], prior_data=[None],
                      from_logits=False, decoded=True)
    ret = []
    for batch_i in range(len(outputs)):
        confusion_matrix_elements = []
        if len(outputs[batch_i]) != 0:
            ious = jaccard(outputs[batch_i][:, 1:5],
                           xywh_to_xyxy_format(tf.cast(gt[batch_i, :, :-1], tf.double))).numpy()  # (#bb_predicted,#gt)
            prediction_detected = np.any((ious > threshold), axis=1)
            max_iou_ind = np.argmax(ious, axis=1)
            for i, prediction in enumerate(prediction_detected):
                gt_idx = int(gt[batch_i, max_iou_ind[i], 4])
                class_name = id_to_name.get(gt_idx)
                gt_label = f"{class_name}"
                confidence = outputs[batch_i][i, 0]
                if prediction:  # TP
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(gt_label),
                        ConfusionMatrixValue.Positive,
                        float(confidence)
                    ))
                else:  # FP
                    class_name = id_to_name.get(int(outputs[batch_i][i, -1]))
                    pred_label = f"{class_name}"
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(pred_label),
                        ConfusionMatrixValue.Negative,
                        float(confidence)
                    ))
        else:  # No prediction
            ious = np.zeros((1, gt[batch_i, ...].shape[0]))
        gts_detected = np.any((ious > threshold), axis=0)
        for k, gt_detection in enumerate(gts_detected):
            label_idx = gt[batch_i, k, -1]
            if not gt_detection and label_idx != CONFIG['BACKGROUND_LABEL']:  # FN
                class_name = id_to_name.get(int(gt[batch_i, k, -1]))
                confusion_matrix_elements.append(ConfusionMatrixElement(
                    f"{class_name}",
                    ConfusionMatrixValue.Positive,
                    float(0)
                ))
        if all(~ gts_detected):
            confusion_matrix_elements.append(ConfusionMatrixElement(
                "background",
                ConfusionMatrixValue.Positive,
                float(0)
            ))
        ret.append(confusion_matrix_elements)
    return ret