import torch
import os
from PIL import Image
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plts
import cv2
from cv2 import dnn
import mmcv
import numpy as np


input_dir = "/home/chenzhen/code/detection/datasets/test/"
out_dir = "/home/chenzhen/code/detection/datasets/nms_save-4"
config_file = "/home/chenzhen/code/my_github_code/mmdetection/configs/datang_detection/yolox_s_8x8_300e_coco.py"
checkpoint_file = "/home/chenzhen/code/detection/mmdetection/checkpoint/800-origin-yolox.pth"
device = 'cuda:0'

EPS = 1e-2
model = init_detector(config_file, checkpoint_file, device=device)
CLASSES = model.CLASSES

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]

def dete_result(file):
    img = mmcv.imread(os.path.join(input_dir, file))
    result = inference_detector(model, img)
    return img, result


# 类间nms
def corner2center(bbox):
    left, top, right, bottom = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    _bbox = torch.stack((left, top, right - left, bottom - top), dim=-1)
    return _bbox


def get_nms(result):
    bbox_result = result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    labels = labels.reshape(-1, 1)
    pred = np.concatenate((bboxes, labels), axis=1)
    pred = torch.tensor(pred)

    return pred


def draw_label_type(draw_img, bbox, labels, label_color, xywh=False):
    if xywh:
        x0, y0, w, h = map(lambda x: int(round(x)), bbox)
        x1, y1 = x0 + w, y0 + h
    else:
        x0, y0, x1, y1 = map(lambda x: int(round(x)), bbox)
    label = labels

    cv2.rectangle(draw_img, [x0, y0], [x1, y1], color=label_color, thickness=2)
    labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

    if y0 - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img, (x0, y0 + 2), (x0 + labelSize[0], y0 + labelSize[1] + 3), color=label_color, thickness=-1)
        cv2.putText(draw_img, label, (x0, y0 + labelSize[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
    else:
        cv2.rectangle(draw_img, (x0, y0 - labelSize[1] - 3), (x0 + labelSize[0], y0 - 3), color=label_color, thickness=-1)
        cv2.putText(draw_img, label, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
    return draw_img


# bboxes, scores, labels = nms_result[:, :4], nms_result[:, 4], nms_result[:, 5]
def main():
    files = os.listdir(input_dir)
    if len(files) != 0:
        for file in files:
            name = os.path.splitext(file)[0]
            print('detecting: ' + name)
            img, result = dete_result(file)
            nms_result = get_nms(result)
            result = nms_result.clone()

            boxes_bboxes, boxes_scores, boxes_labels = result[:, :4], result[:, 4], result[:, 5]
            boxes_bboxes = corner2center(boxes_bboxes)
            bbox_indices = cv2.dnn.NMSBoxes(boxes_bboxes.numpy(), boxes_scores.numpy(), 0.3, 0.5)
            bbox_result = nms_result[bbox_indices]


            batch_pred = bbox_result.clone()
            batch_bboxes, batch_scores, batch_labels = batch_pred[:, :4], batch_pred[:, 4], batch_pred[:, 5]
            batch_bboxes = corner2center(batch_bboxes)
            batch_indices = cv2.dnn.NMSBoxesBatched(batch_bboxes.numpy(), batch_scores.numpy(), batch_labels.numpy().astype(int), 0.3, 0.8)

            # print(indices)
            pred_result = bbox_result[batch_indices]
            bbox, score, label = pred_result[:, :4], pred_result[:, 4], pred_result[:, 5]
            for i, (bbox, score, label) in enumerate(zip(bbox.numpy().astype(np.int32), score.numpy(), label.numpy())):
                label_text = CLASSES[int(label)]
                label_text += f'|{score:.03f}'
                img = draw_label_type(img, bbox, f'{label_text}', PALETTE[int(label)], xywh=False)
            path = os.path.join(out_dir, name + ".jpg")
            im = Image.fromarray(img)
            im.save(path)




















if __name__ == '__main__':
    main()