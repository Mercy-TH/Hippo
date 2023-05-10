import cv2
import shutil
import numpy as np
import yaml
from PIL import Image
import imgviz
import os
import tqdm
from pycocotools.coco import COCO
from yaml_operate import read_yaml
from multiprocessing import Process


class DataPreprocess:
    def __init__(self, labels, annotation_file, write_root_dir, date_type, binary=False):
        self.labels = labels
        self.annotation_file = annotation_file
        self.write_root_dir = write_root_dir
        self.config = self.get_config()
        self.coco_root = self.config['coco_dir']
        self.date_type = date_type
        self.binary = binary
        self.label_max_workers = 1
        self.imgid_max_workers = 2
        self.coco = COCO(annotation_file)
        self.check_root_dir()

    def check_root_dir(self):
        if os.path.exists(self.write_root_dir) is False:
            os.makedirs(self.write_root_dir)

    def get_config(self):
        filename = "../../../config/config.yaml"
        with open(filename, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        return cfg

    def mul_process_label(self, args):
        process = [Process(target=self.write_mask_label, args=(args[i],)) for i in range(self.label_max_workers)]
        [p.start() for p in process]
        [p.join() for p in process]

    def mul_write_mask(self):
        coco = self.coco
        mui_ins_ids = [coco.getCatIds(l)[0] for l in self.labels]
        labels_ls = self.split(self.labels, self.label_max_workers)
        args = [(coco, mui_ins_ids, labels_ls[i], self.label_max_workers) for i in range(self.label_max_workers)]
        self.mul_process_label(args)

    def mul_process_imgid(self, kwargs):
        process = [Process(target=self.write_mask_imgid, args=(kwargs,)) for i in range(self.imgid_max_workers)]
        [p.start() for p in process]
        [p.join() for p in process]

    def write_mask_imgid(self, kwargs):
        mui_ins_ids, labels, ins_ids, catIds, imgIds = kwargs[0], kwargs[1], kwargs[2], kwargs[3], kwargs[4]
        coco = self.coco
        for imgId in tqdm.tqdm(imgIds, ncols=100):
            img = coco.loadImgs(imgId)[0]
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            ori_anns = coco.loadAnns(annIds)
            anns = [an for an in ori_anns if (an['category_id'] in mui_ins_ids) or (an['category_id'] == ins_ids)]
            if len(annIds) > 0:
                mask = coco.annToMask(anns[0]) * anns[0]['category_id']
                for i in range(len(anns) - 1):
                    mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
                    img_id = img['file_name'].split('.')[0]
                    img_id_dir = os.path.join(self.write_root_dir, str(img_id))
                    has_img_id_dir = os.path.exists(img_id_dir)
                    image_dir = os.path.join(img_id_dir, 'image')
                    has_image_dir = os.path.exists(image_dir)
                    mask_dir = os.path.join(img_id_dir, 'mask')
                    has_mask_dir = os.path.exists(mask_dir)
                    if has_img_id_dir:
                        if has_image_dir is False:
                            try:
                                os.makedirs(image_dir)
                            except FileExistsError:
                                pass
                        if has_mask_dir is False:
                            try:
                                os.makedirs(mask_dir)
                            except FileExistsError:
                                pass
                    if has_img_id_dir is False:
                        try:
                            os.makedirs(img_id_dir)
                        except FileExistsError:
                            pass
                        try:
                            os.makedirs(os.path.join(img_id_dir, 'image'))
                        except FileExistsError:
                            pass
                        try:
                            os.makedirs(os.path.join(img_id_dir, 'mask'))
                        except FileExistsError:
                            pass
                    source_image_file = os.path.join(self.coco_root, self.date_type, img['file_name'])
                    target_image_file = os.path.join(image_dir, img['file_name'])
                    shutil.copyfile(source_image_file, target_image_file)
                    if self.binary:
                        cv2.imwrite(os.path.join(mask_dir, img['file_name'].replace('.jpg', '.png')), self.mask2binary(mask))
                        continue
                    mask_save_path = os.path.join(mask_dir, img['file_name'].replace('.jpg', '.png'))
                    self.save_colored_mask(mask, mask_save_path)

    def write_mask_label(self, args):
        coco, mui_ins_ids, labels = args[0], args[1], args[2]
        processes = self.imgid_max_workers
        for label in labels:
            ins_ids = coco.getCatIds(label)[0]
            # 获取某一类的所有图片集合, 比如获取包含dog的所有图片
            imgIds = coco.getImgIds(catIds=ins_ids)
            catIds = coco.getCatIds()
            imgIds_ls = self.split(imgIds, processes)
            kwargs = [(mui_ins_ids, labels, ins_ids, catIds, imgIds_ls[i]) for i in range(processes)]
            process = [Process(target=self.mul_process_imgid, args=(kwargs[i],)) for i in range(processes)]
            [p_.start() for p_ in process]
            [p_.join() for p_ in process]

    def split(self, a, n):
        k, m = divmod(len(a), n)
        tmp = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
        return list(tmp)

    def write_mask(self):
        coco = COCO(self.annotation_file)
        mui_ins_ids = [coco.getCatIds(l)[0] for l in labels]
        for label in self.labels:
            ins_ids = coco.getCatIds(label)[0]
            # 获取某一类的所有图片集合, 比如获取包含dog的所有图片
            imgIds = coco.getImgIds(catIds=ins_ids)
            catIds = coco.getCatIds()
            for imgId in tqdm.tqdm(imgIds, ncols=100):
                img = coco.loadImgs(imgId)[0]
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                ori_anns = coco.loadAnns(annIds)
                anns = [an for an in ori_anns if (an['category_id'] in mui_ins_ids) or (an['category_id'] == ins_ids)]
                if len(annIds) > 0:
                    mask = coco.annToMask(anns[0]) * anns[0]['category_id']
                    # mask = coco.annToMask(anns[0])
                    for i in range(len(anns) - 1):
                        mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
                        img_id = img['file_name'].split('.')[0]
                        img_id_dir = os.path.join(self.write_root_dir, str(img_id))
                        has_img_id_dir = os.path.exists(img_id_dir)
                        image_dir = os.path.join(img_id_dir, 'image')
                        has_image_dir = os.path.exists(image_dir)
                        mask_dir = os.path.join(img_id_dir, 'mask')
                        has_mask_dir = os.path.exists(mask_dir)
                        if os.path.exists(self.write_root_dir) is False:
                            os.makedirs(self.write_root_dir)
                        if has_img_id_dir:
                            if has_image_dir is False:
                                os.makedirs(image_dir)
                            if has_mask_dir is False:
                                os.makedirs(mask_dir)
                        if has_img_id_dir is False:
                            os.makedirs(img_id_dir)
                            os.makedirs(os.path.join(img_id_dir, 'image'))
                            os.makedirs(os.path.join(img_id_dir, 'mask'))
                        source_image_file = os.path.join(self.coco_root, self.date_type, img['file_name'])
                        target_image_file = os.path.join(image_dir, img['file_name'])
                        shutil.copyfile(source_image_file, target_image_file)
                        if self.binary:
                            cv2.imwrite(os.path.join(mask_dir, img['file_name'].replace('.jpg', '.png')),
                                        self.mask2binary(mask))
                            continue
                        mask_save_path = os.path.join(mask_dir, img['file_name'].replace('.jpg', '.png'))
                        self.save_colored_mask(mask, mask_save_path)

    def mask2binary(self, mask):
        img = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
        img = np.array(img)
        img = np.expand_dims(img, 2).repeat(3, axis=2)
        return img

    def save_colored_mask(self, mask, save_path):
        lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(save_path)


if __name__ == '__main__':
    config_yaml = '../config/config.yaml'
    coco_yaml = '../config/coco.yaml'
    coco_dir = read_yaml(config_yaml)['coco_dir']
    # labels
    labels = [v for k, v in read_yaml(coco_yaml).items()]

    data_type = ['val', 'train']
    data_type = ['train']
    for dt in data_type:
        annotation_file = os.path.join(coco_dir, 'annotations', f'instances_{dt}2017.json')
        write_root_dir = os.path.join(coco_dir, 'images', dt)
        dp = DataPreprocess(labels, annotation_file, write_root_dir, f'{dt}2017', True)
        if dt == 'val':
            dp.label_max_workers = 8
            dp.imgid_max_workers = 8
        # dp.muilty_write_mask()
        dp.mul_write_mask()
