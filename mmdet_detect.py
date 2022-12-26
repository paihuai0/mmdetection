# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np
import mmcv
from tqdm import trange
import os
import torch
import torchvision
from mmdet.apis import inference_detector, init_detector
from pycocotools.coco import COCO

img_path = Path('/home/chenzhen/code/detection/datasets/coco20/train')
config_path = Path('/home/chenzhen/code/detection/mmdetection/configs/datang_detection/yolox_s_temp.py')
checkpoint = Path('/home/chenzhen/code/detection/mmdetection/checkpoint/800-v1.0-model.pth')
draw_path = Path('/home/chenzhen/code/detection/datasets/dair-and-dthangzhou-sub100/draw')
draw_path.mkdir(parents=True, exist_ok=True)
score_thr = 0.3


CLASSES = (
    "Car",
    "Bus",
    "Cycling",
    "Pedestrian",
    "driverless_Car",
    "Truck",
    "Animal",
    "Obstacle",
    "Special_Target",
    "Other_Objects",
    "Unmanned_riding"
)

def draw(image, bbox, name, color, xywh=True):
    if xywh:
        x0, y0, w, h = map(lambda x: int(round(x)), bbox)
        x1, y1 = x0 + w, y0 + h
    else:
        x0, y0, x1, y1 = map(lambda x: int(round(x)), bbox)
    cv2.rectangle(image, [x0, y0], [x1, y1], color, 2)
    cv2.putText(image, name, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, thickness=1)
    return image

def nms(bboxes, scores, thresh):
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score降序排序（保存的是索引）
    # values, indices = torch.sort(scores, descending=True)
    indices = scores.sort(descending=True)[0]  # torch

    indice_res = torch.randn([1, 4]).to(bboxes)
    while indices.size()[0] > 0:  # indices.size()是一个Size对象，我们要取第一个元素是int，才能比较
        save_idx, other_idx = indices[0], indices[1:]
        indice_res = torch.cat((indice_res, bboxes[save_idx].unsqueeze(0)),
                               dim=0)  # unsqueeze是添加一个维度，让bboxes.shape从[4]-->[1,4]

        inter_x1 = torch.max(x1[save_idx], x1[other_idx])
        inter_y1 = torch.max(y1[save_idx], y1[other_idx])
        inter_x2 = torch.min(x2[save_idx], x2[other_idx])
        inter_y2 = torch.min(y2[save_idx], y2[other_idx])
        inter_w = torch.max(inter_x2 - inter_x1 + 1, torch.tensor(0).to(bboxes))
        inter_h = torch.max(inter_y2 - inter_y1 + 1, torch.tensor(0).to(bboxes))

        inter_area = inter_w * inter_h
        union_area = areas[save_idx] + areas[other_idx] - inter_area + 1e-6
        iou = inter_area / union_area

        indices = other_idx[iou < thresh]
    return indice_res[1:]


# 类内nms，把不同类别的乘以一个偏移量，把不同类别的bboxes给偏移到不同位置。
def class_nms(bboxes, scores, cat_ids, iou_threshold):
    '''
    :param bboxes: torch.tensor([n, 4], dtype=torch.float32)
    :param scores: torch.tensor([n], dtype=torch.float32)
    :param cat_ids: torch.tensor([n], dtype=torch.int32)
    :param iou_threshold: float
    '''
    max_coordinate = bboxes.max()

    # 为每一个类别/每一层生成一个很大的偏移量
    offsets = cat_ids * (max_coordinate + 1)
    # bboxes加上对应类别的偏移量后，保证不同类别之间bboxes不会有重合的现象
    bboxes_for_nms = bboxes + offsets[:, None]
    indice_res = nms(bboxes_for_nms, scores, iou_threshold)
    return indice_res


def main():
    # build the model from a config file and a checkpoint file
    model = init_detector(str(config_path), str(checkpoint), device='cuda:0')

    # test a single image
    for filename in os.listdir(img_path):
        img = os.path.join(img_path, filename)

        img = mmcv.imread(img).astype(np.uint8)
        result = inference_detector(model, img)
        bbox_result = result
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        if score_thr > 0:
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        i = torchvision.ops.nms(torch.tensor(bboxes[:, :-1]), torch.tensor(bboxes[:, -1]), 0.3)  # NMS
        bboxes = bboxes[i]
        scores_xi, bboxes_xi = bboxes[:, -1], bboxes[:, :-1]
        xi = torchvision.ops.nms(torch.tensor(bboxes_xi), torch.tensor(scores_xi), 0.65)  # NMS
        bboxes = bboxes[xi]

        img = mmcv.bgr2rgb(img)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)






if __name__ == '__main__':
    main()
