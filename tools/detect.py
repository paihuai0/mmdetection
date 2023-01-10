from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os



input_dir = "/home/chenzhen/code/detection/datasets/test/"
out_dir = "/home/chenzhen/code/detection/datasets/nms_save-2"
config_file = "/home/chenzhen/code/my_github_code/mmdetection/configs/datang_detection/yolox_s_8x8_300e_coco.py"
checkpoint_file = "/home/chenzhen/code/detection/mmdetection/checkpoint/800-origin-yolox.pth"
device = 'cuda:0'

# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image

for filename in os.listdir(input_dir):
    img = os.path.join(input_dir, filename)
    result = inference_detector(model, img)
    out_file = os.path.join(out_dir, filename)
    show_result_pyplot(model, img, result, out_file=out_file, score_thr=0.3)
