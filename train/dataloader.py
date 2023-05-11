import os
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms
from config.config import *

data_transforms = transforms.Compose([
    transforms.ToTensor()
])


class DataSets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # xy_path = self.load_path()
        xy_path = self.load_coco_path()
        self.len = len(xy_path)
        self.x = xy_path[:, 0]
        self.y = xy_path[:, 1]

    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        img = cv2.imread(self.x[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.y[index])
        if img is None or mask is None:
            return torch.Tensor([[], []])
        img = cv2.resize(img, resize) / 255.
        mask = cv2.resize(mask, resize) / 255.
        return data_transforms(img).float(), data_transforms(mask).float()

    def __len__(self):
        return self.len

    def load_path(self):
        img_dir = os.path.join(self.data_dir, 'imgs')
        mask_dir = os.path.join(self.data_dir, 'masks')
        xy_path = []
        for img in os.listdir(img_dir):
            name = img.split('.')[0]
            img_path = os.path.join(img_dir, img)
            mask_path = os.path.join(mask_dir, f'{name}.png')
            xy_path.append([img_path, mask_path])
        return np.array(xy_path)

    def load_coco_path(self):
        xy_path = []
        for file in os.listdir(self.data_dir):
            img_path = os.path.join(self.data_dir, file, 'img', f'{file}.jpg')
            mask_path = os.path.join(self.data_dir, file, 'mask', f'{file}_merged_mask.png')
            xy_path.append([img_path, mask_path])
        return np.array(xy_path)


# ### load coco data online by kaggle
# data_transforms = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# coco_dir = '/kaggle/input/coco-2017-dataset/coco2017'
# dt = 'train'
# annotation_file = os.path.join(coco_dir, 'annotations', f'instances_{dt}2017.json')
# coco_yaml = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
# coco = COCO(annotation_file)
# ### exp labels
# exp_labels = ['person', 'cat', 'dog']
# labels = [v for k, v in coco_yaml.items() if v in exp_labels]
# catIds = coco.getCatIds()
# for label in labels:
#     ins_ids = coco.getCatIds(label)[0]
#     # 获取某一类的所有图片集合, 比如获取包含dog的所有图片
#     imgIds = coco.getImgIds(catIds=ins_ids)
# class DataSets(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.img_path, self.imgIds = self.load_coco_img_mask()
#         self.len = len(self.imgIds)
#
#
#     def __getitem__(self, index):
#
#         img = coco.loadImgs(self.imgIds[index])[0]
#         mui_ins_ids = [coco.getCatIds(l)[0] for l in labels]
#         annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#         ori_anns = coco.loadAnns(annIds)
#         anns = [an for an in ori_anns if (an['category_id'] in mui_ins_ids) or (an['category_id'] == ins_ids)]
#         if len(annIds) > 0:
#             mask = coco.annToMask(anns[0]) * anns[0]['category_id']
#             for i in range(len(anns) - 1):
#                 mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
#             img_id = img['file_name'].split('.')[0]
#             img_path = os.path.join(coco_dir, f'{dt}2017', f'{img_id}.jpg')
#             image = cv2.imread(img_path)
#             mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
#             mask = np.array(mask)
#             mask = np.expand_dims(mask, 2).repeat(3, axis=2)
#
#
#
#
#         # return self.x[index], self.y[index]
#         #         img = cv2.imread(self.x[index])
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         #         mask = cv2.imread(self.y[index])
#         if img is None or mask is None:
#             return torch.Tensor([[], []])
#         img = cv2.resize(img, resize) / 255.
#         mask = cv2.resize(mask, resize) / 255.
#         return data_transforms(img).float(), data_transforms(mask).float()
#
#     def __len__(self):
#         return self.len
#
#     def load_coco_img_mask(self):
#         img = []
#
#         for file in os.listdir(self.data_dir):
#             if file.split('.')[0] in imgIds:
#                 img_path = os.path.join(self.data_dir, f'{file}'.jpg)
#                 img.append(img_path)
#         return img, imgIds



# if __name__ == '__main__':
#     path = '/opt/projects/unetplusplus/python/datas'
#     dataset = DataSets(path)
#     print(dataset[0][0])
#     print(dataset[0][1])
