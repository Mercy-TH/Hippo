import json
import os
import yaml
from pycocotools.coco import COCO

filename = "../config/config.yaml"
with open(filename, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
coco_root = cfg['coco_dir']


def write_img_msg(data_type):
    annFile = os.path.join(coco_root, f'annotations/instances_{data_type}.json')
    coco = COCO(annFile)

    imgs = coco.imgs
    anns = coco.imgToAnns
    cats = coco.cats

    data = []

    for img_id, img_v in imgs.items():
        anns_tmp = anns[img_id]
        for i in range(len(anns_tmp)):
            anns_tmp[i]['category'] = cats[anns_tmp[i]['category_id']]
        temp = {
            'img': {
                'img_id': img_id,
                'file_name': img_v['file_name'],
                'width': img_v['width'],
                'height': img_v['height']
            },
            'anns': anns_tmp
        }
        data.append(temp)
    for m in data:
        with open(f'../coco/{data_type}_data.txt', 'a+', encoding='utf-8') as f:
            f.write(json.dumps(m, ensure_ascii=False))
            f.write('\n')


def get_file_names(dir_name):
    files = os.listdir(dir_name)
    tp = dir_name.split('/')[-1]
    dic = json.dumps({'file_names': files})
    with open(f'../coco/{tp}_file_names.json', 'w', encoding='utf-8') as f:
        f.write(dic)


dir_name = f'{coco_root}/val2017'
get_file_names(dir_name)
dir_name = f'{coco_root}/train2017'
get_file_names(dir_name)

data_type = 'train2017'
write_img_msg(data_type)
data_type = 'val2017'
write_img_msg(data_type)
