from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os



imagepath = "/home/chenzhen/code/detection/datasets/test/"
savepath = "/home/chenzhen/code/detection/mmdetection/runs/test_save"
config_file = "/home/chenzhen/code/detection/mmdetection/configs/datang_detection/yolox_s_8x8_300e_coco.py"
checkpoint_file = "/home/chenzhen/code/detection/mmdetection/checkpoint/800-origin-yolox.pth"
device = 'cuda:0'

# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image

for filename in os.listdir(imagepath):
    img = os.path.join(imagepath, filename)
    result = inference_detector(model, img)
    out_file = os.path.join(savepath, filename)
    show_result_pyplot(model, img, result, out_file=out_file, score_thr=0.4)
