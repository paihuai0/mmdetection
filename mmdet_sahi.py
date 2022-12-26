
from sahi.predict import  predict

model_path = "/home/chenzhen/code/detection/mmdetection/checkpoint/800-origin-yolox.pth"
config_path = "/home/chenzhen/code/detection/mmdetection/configs/datang_detection/yolox_s_temp.py"
dataset_json = "/home/chenzhen/code/detection/datasets/dt_hangzhou/coco_dt_with_date_captured/annotations/val-dthangzhou.json"
source_image_dir = "/home/chenzhen/code/detection/datasets/dt_hangzhou/coco_dt_with_date_captured/val/"

if __name__ == '__main__':

    model_type = "mmdet"
    model_path = model_path
    model_config_path = config_path
    model_device = "cuda:0"  # or 'cuda:0'
    model_confidence_threshold = 0.01

    slice_height = 1080
    slice_width = 1920
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2

    predict(
        model_type=model_type,
        model_path=model_path,
        model_config_path=config_path,
        model_device=model_device,
        model_confidence_threshold=model_confidence_threshold,
        source=source_image_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        dataset_json_path=dataset_json,
        postprocess_type='NMS',
        no_sliced_prediction=True,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

