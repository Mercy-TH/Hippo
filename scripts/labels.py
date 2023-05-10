import json
import os
from yaml_operate import read_yaml
from coco_labels import label2n, n2label
import tqdm
import shutil
import cv2
import numpy as np
from pycocotools.coco import COCO

"""
images
    image_id
        img
            xxx.jpg
        mask
            xxx.png
"""


write_dir_name = 'images'
write_dir = r'/media/th/Ubuntu 20.0/beifen/download/coco/cats'

class Labels:
    def __init__(self):
        self.date_type = 'train2017'
        self.config_yaml = '../config/config.yaml'
        self.coco_yaml = '../config/coco.yaml'
        self._init_coco()
        self.check_images_dir()
        self.label = self.config['exp_labels']

    def _init_coco(self):
        self.config = read_yaml(self.config_yaml)
        self.coco_dir = self.config['coco_dir']
        self.annotation_file = os.path.join(self.coco_dir, 'annotations', f'instances_train2017.json')
        self.coco = COCO(self.annotation_file)

    def check_images_dir(self):
        images_dir = os.path.join(self.coco_dir, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

    def write_img(self, img_file_name):
        img_name = img_file_name.split('.')[0]
        img_path = os.path.join(write_dir, str(img_name), 'img')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        source_image_file = os.path.join(self.coco_dir, self.date_type, img_file_name)
        target_image_file = os.path.join(img_path, img_file_name)
        shutil.copyfile(source_image_file, target_image_file)

    def write_mask(self, img_file_name, category_name, mask_file_name, mask):

        mask_dir = os.path.join(self.coco_dir, 'images', str(img_file_name), 'mask')
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        category_name_dir = os.path.join(self.coco_dir, 'images', str(img_file_name), 'mask', category_name)
        if not os.path.exists(category_name_dir):
            os.makedirs(category_name_dir)
        cv2.imwrite(os.path.join(category_name_dir, mask_file_name), mask)

    def mask2binary(self, mask):
        img = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
        img = np.array(img)
        img = np.expand_dims(img, 2).repeat(3, axis=2)
        return img

    def get_img_cats(self, imgId):
        """
        get all cats of one image
        :return:
        """
        img = self.coco.loadImgs(imgId)[0]
        img_name = img['file_name']
        img_file_name = img['file_name'].split('.')[0]
        self.write_img(img_name)
        ex_category_ids = [label2n()[l] for l in self.label]
        annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        # category_names = list(set([n2label()[i] for i in [ann['category_id'] for ann in anns]]))
        category_ids = list(set([ann['category_id'] for ann in anns if ann['category_id'] in ex_category_ids]))
        mask_dic = {}

        # label_names = [n for n in category_names if n in self.label]
        # label_ids = [i for i in category_ids if i in [label2n()[l] for l in self.label]]
        # print(label_names)
        # print(label_ids)
        for k in category_ids:
            mask_dic[k] = [v for v in anns if v['category_id'] == k]
        mask_ls = []
        mask = None
        for k, v in mask_dic.items():
            if len(v) == 1:
                mask = self.coco.annToMask(v[0])
            else:
                mask = self.coco.annToMask(v[0])
                for val in v[1:]:
                    mask += self.coco.annToMask(val)
            mask_ls.append(mask)
        mask_zero = np.zeros_like(mask, dtype='uint8')
        mask_zero = np.expand_dims(mask_zero, 2).repeat(3, axis=2)
        for im in mask_ls:
            mask_zero += self.mask2binary(im)
        merged_mask = cv2.threshold(mask_zero, 0, 255, cv2.THRESH_BINARY)[1]
        merged_mask_file_name = img_file_name + '_merged_mask.png'
        self.write_merged_mask(img_file_name, merged_mask_file_name, merged_mask)
        # cv2.imshow('mask_channel', merged_mask)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print()

    def write_merged_mask(self, img_file_name, merged_mask_file_name, merged_mask):
        mask_dir = os.path.join(write_dir, str(img_file_name), 'mask')
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        cv2.imwrite(os.path.join(mask_dir, merged_mask_file_name), merged_mask)

    def run(self):
        for label in self.label:
            ins_ids = self.coco.getCatIds(label)
            ins_ids = self.coco.getCatIds(label)[0]
            imgIds = self.coco.getImgIds(catIds=ins_ids)
            n = 0
            for imgId in tqdm.tqdm(imgIds, ncols=100):
                self.get_img_cats(imgId)
            #     if n == 3:
            #         break
            #     n += 1
            # break

    def derive_merged_file_name(self):
        def write_file_json(file_json):
            with open(os.path.join(self.coco_dir, 'merged_file.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(file_json, ensure_ascii=False))

        file_json = {'path': []}
        for img_id in os.listdir(os.path.join(self.coco_dir, 'images')):
            dic = {
                'img_path': os.path.join(os.path.join(self.coco_dir, 'images', img_id), 'img', img_id + '.jpg'),
                'mask_path': os.path.join(os.path.join(self.coco_dir, 'images', img_id), 'mask',
                                          img_id + '_merged_mask.png'),
            }
            file_json['path'].append(dic)
        write_file_json(file_json)
        return file_json

    def derive_file_json(self):
        def write_file_json(file_json):
            with open(os.path.join(self.coco_dir, 'file.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(file_json, ensure_ascii=False))

        file_json = {'path': []}
        for img_id in os.listdir(os.path.join(self.coco_dir, 'images')):
            dic = {
                'img_path': os.path.join(os.path.join(self.coco_dir, 'images', img_id), 'img', img_id + '.jpg'),
                'mask_path': {},
                'category_names': [],
                'category_name_dirs': {}
            }
            mask_dir = os.path.join(self.coco_dir, 'images', img_id, 'mask')
            for category_name in os.listdir(mask_dir):
                dic['category_names'].append(category_name)
                dic['category_name_dirs'][category_name] = os.path.join(self.coco_dir, 'images', img_id, 'mask',
                                                                        category_name)
                category_name_dir = os.path.join(self.coco_dir, 'images', img_id, 'mask', category_name)
                for c in os.listdir(category_name_dir):
                    dic['mask_path'][category_name] = {}
                    # dic['mask_path'][c]['1_channel'] = os.path.join(category_name_dir, img_id + '_1_channel.png')
                    dic['mask_path'][category_name]['1_channel'] = os.path.join(category_name_dir,
                                                                                img_id + '_1_channel.png')
                    dic['mask_path'][category_name]['3_channel'] = os.path.join(category_name_dir,
                                                                                img_id + '_3_channel.png')
            file_json['path'].append(dic)
        write_file_json(file_json)
        return file_json


if __name__ == '__main__':
    labels = Labels()
    labels.run()
    # res = labels.derive_merged_file_name()
    # print(res)
    # print(json.dumps(res, ensure_ascii=False))
