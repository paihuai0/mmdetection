import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
unloader = transforms.ToPILImage()

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,save_dir = 'feature_map',name = None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                plt.imshow(superimposed_img, cmap='gray')
                plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                # i=i+1

def draw(image, bbox, xywh=True):
    if xywh:
        x0, y0, w, h = map(lambda x: int(round(x)), bbox)
        x1, y1 = x0 + w, y0 + h
    else:
        x0, y0, x1, y1 = map(lambda x: int(round(x)), bbox)
    cv2.rectangle(image, [x0, y0], [x1, y1], (0, 0, 255), 2)
    return image


def imshow(img, gt_bboxes):
    image = img.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    # gt_bboxes = gt_bboxes[0].cpu().numpy().tolist()
    # for gt_bbox in gt_bboxes:
    #     bbox = gt_bbox
    #     draw(image, bbox)
    #
    plt.imshow(image)

    plt.pause(5)  # pause a bit so that plots are updated