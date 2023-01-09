import mmcv
import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
from PIL import Image
import matplotlib.pyplot as plt
import cv2

input_dir = "/home/chenzhen/code/detection/datasets/img/"
out_dir = "/home/chenzhen/code/detection/datasets/crop_save"
config_file = "/home/chenzhen/code/my_github_code/mmdetection/configs/datang_detection/yolox_s_8x8_300e_coco.py"
checkpoint_file = "/home/chenzhen/code/detection/mmdetection/checkpoint/800-origin-yolox.pth"
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)
CLASSES = model.CLASSES

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def dete_result(file):
    img = mmcv.imread(os.path.join(input_dir, file))
    result = inference_detector(model, img)
    return result


def get_expend_box(box,
                   expend_ratio=0.25):
    """
    根据检测结果，按照0.25的比率放大检测框
    Args:
        box:
        expend_ratio:

    Returns:放大的检测框

    """
    height, width = box[3] - box[1], box[2] - box[0]
    box[0] = box[0] - expend_ratio*width
    box[2] = box[2] + expend_ratio*width
    box[1] = box[1] - expend_ratio*height
    box[3] = box[3] + expend_ratio*height

    return box


def get_crop_result(result=None,
                    save_calsses=None,
                    score_thr=0.1):
    crop_list = []
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

    positions = bboxes[:, :4].astype(np.int32)
    for i, (pos, label) in enumerate(zip(positions, labels)):
        label_text = CLASSES[
                label] if CLASSES is not None else f'class {label}'
        if label_text == save_calsses:
            crop_list.append(pos)

    return crop_list


def main():
    files = os.listdir(input_dir)
    if len(files) != 0:
        for file in files:
            img = Image.open(os.path.join(input_dir, file))
            name = os.path.splitext(file)[0]
            print('detecting: ' + name)
            result = dete_result(file)
            crop_list = get_crop_result(result=result, score_thr=0.7, save_calss='Car')
            for i, bbox in enumerate(crop_list):
                bbox = get_expend_box(box=bbox)
                img_crop = img.crop(bbox)
                path = os.path.join(out_dir, name+str(i) + ".jpg")
                img_crop.save(path)
                # plt.imshow(img_crop)
                # plt.show()


if __name__ == '__main__':
    main()