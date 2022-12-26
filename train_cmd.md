# 1.如何使用制定卡训练

- 1、CUDA_VISIBLE_DEVICES=4,5 
- ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py 2
- 2、CUDA_VISIBLE_DEVICES=4,5 
- python -m torch.distributed.launch --nproc_per_node=2 - -master_port=$PORT ./tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py --launcher pytorch

# 2.
重要：配置文件中的默认学习率（lr=0.02）是8个GPU和samples_per_gpu=2（批大小= 8 * 2 = 16）。
根据线性缩放规则，如果您使用不同的GPU或每个GPU的有多少张图像，则需要按批大小设置学习率，
例如，对于4GPU* 2 img / gpu=8，lr =8/16 * 0.02 = 0.01 ；对于16GPU* 4 img / gpu=64，lr =64/16 *0.02 = 0.08 。

计算公式：批大小(gup-num * samples_per_gpu) / 16 * 0.02

# 3. 
In mmdet,
samples per gpu * GPUs = batch size
epoch is training epoch, that you are setting
total_iteration per epoch = total_sample / batch_size

# 4. 
172.217.215.90 translate.googleapis.com