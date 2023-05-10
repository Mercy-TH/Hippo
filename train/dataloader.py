import os
import cv2
import numpy as np
import torch
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

# if __name__ == '__main__':
#     path = '/opt/projects/unetplusplus/python/datas'
#     dataset = DataSets(path)
#     print(dataset[0][0])
#     print(dataset[0][1])
