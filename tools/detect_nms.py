import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector
from tools.show_result import show_result

input_dir = "/home/chenzhen/code/detection/datasets/test/"
out_dir = "/home/chenzhen/code/detection/datasets/test_save"
config_file = "/home/chenzhen/code/my_github_code/mmdetection/configs/datang_detection/yolox_s_8x8_300e_coco.py"
checkpoint_file = "/home/chenzhen/code/detection/mmdetection/checkpoint/800-origin-yolox.pth"
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 0.000000000001)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nms_cpu(total_detections, classnames, thresh=0.5):
    for i in range(len(classnames)):
        keep = py_cpu_nms(total_detections[i], thresh)
        total_detections[i] = total_detections[i][keep]
    return total_detections


files = os.listdir(input_dir)
if len(files) != 0:
    for file in files:
        name = os.path.splitext(file)[0]
        print('detecting: ' + name)
        img = mmcv.imread(os.path.join(input_dir, file))
        # img_resize = mmcv.imresize(img, (800, 800))
        result = inference_detector(model, img)
        result = nms_cpu(result, model.CLASSES, 0.1)
        result_img = show_result(img, result, CLASSES=model.CLASSES, out_file=os.path.join(out_dir, name + '.jpg'))
