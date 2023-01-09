import torch
import os
from PIL import Image
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plts
import cv2
import mmcv
import numpy as np


red = (255, 0, 0)
blue = (0, 0, 255)
orange = (255, 165, 0)
red = red[::-1]
blue = blue[::-1]
orange = orange[::-1]

input_dir = "/home/chenzhen/code/detection/datasets/new/"
out_dir = "/home/chenzhen/code/detection/datasets/nms_save"
config_file = "/home/chenzhen/code/my_github_code/mmdetection/configs/datang_detection/yolox_s_8x8_300e_coco.py"
checkpoint_file = "/home/chenzhen/code/detection/mmdetection/checkpoint/800-origin-yolox.pth"
device = 'cuda:0'

EPS = 1e-2
model = init_detector(config_file, checkpoint_file, device=device)
CLASSES = model.CLASSES

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def dete_result(file):
    img = mmcv.imread(os.path.join(input_dir, file))
    result = inference_detector(model, img)
    return img, result


# 类间nms
def nms(preds,thresh):
    bboxes, scores = preds[:, :4], preds[:, 4]

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score降序排序（保存的是索引）
    # values, indices = torch.sort(scores, descending=True)
    indices = scores.sort(descending=True)[1]  # torch

    indice_res = torch.randn([1, 6]).to(bboxes)
    while indices.size()[0] > 0:  # indices.size()是一个Size对象，我们要取第一个元素是int，才能比较
        save_idx, other_idx = indices[0], indices[1:]
        indice_res = torch.cat((indice_res, preds[save_idx].unsqueeze(0)),
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
def class_nms(preds, iou_threshold):
    '''
    :param bboxes: torch.tensor([n, 4], dtype=torch.float32)
    :param scores: torch.tensor([n], dtype=torch.float32)
    :param cat_ids: torch.tensor([n], dtype=torch.int32)
    :param iou_threshold: float
    '''
    bboxes,  score, cat_ids = preds[:, :4], preds[:, 4], preds[:, 5]
    max_coordinate = bboxes.max()

    # 为每一个类别/每一层生成一个很大的偏移量
    offsets = cat_ids * (max_coordinate + 1)
    # bboxes加上对应类别的偏移量后，保证不同类别之间bboxes不会有重合的现象
    bboxes_for_nms = bboxes + offsets[:, None]

    preds_con = torch.cat((bboxes_for_nms, score.unsqueeze(0).permute(1, 0), cat_ids.unsqueeze(0).permute(1, 0)), 1)

    indice_res = nms(preds_con, iou_threshold)

    return indice_res


def getnms(result=None,
           score_thr=0.3):
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
    labels = labels.reshape(-1, 1)
    pred = np.concatenate((bboxes, labels), axis=1)
    pred = torch.tensor(pred)

    return pred

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

def main():
    text_color = 'green',
    files = os.listdir(input_dir)
    if len(files) != 0:
        for file in files:
            name = os.path.splitext(file)[0]
            print('detecting: ' + name)
            img, result = dete_result(file)
            result = getnms(result=result)
            nms_result = class_nms(result, 0.65)
            # nms_result = nms_result.numpy().astype(np.int32)
            bboxes, scores, labels = nms_result[:, :4], nms_result[:, 4], nms_result[:, 5]
            for i, (bbox, score, label) in enumerate(zip(bboxes.numpy().astype(np.int32), scores.numpy(), labels.numpy())):
                label_text = CLASSES[int(label)]
                label_text += f'|{score:.03f}'
                img = draw(img, bbox, f'{label_text}', blue, xywh=False)
            path = os.path.join(out_dir, name + ".jpg")
            im = Image.fromarray(img)
            im.save(path)




















if __name__ == '__main__':
    main()