from yaml_operate import read_yaml, write_yaml
from pycocotools.coco import COCO
import os

config_yaml = '../config/config.yaml'
config = read_yaml(config_yaml)
data_type = 'instances_val2017'
path = os.path.join(config['coco_dir'], 'annotations', f'{data_type}.json')
coco = COCO(path)
cats = coco.cats


def n2label():
    n2label = {k: v['name'] for k, v in cats.items()}
    return n2label


def label2n():
    label2n = {v['name']: k for k, v in cats.items()}
    return label2n


def n2superlabel():
    n2superlabel = {k: v['supercategory'] for k, v in cats.items()}
    return n2superlabel


def superlabel2n():
    superlabel2n = {v['supercategory']: k for k, v in cats.items()}
    return superlabel2n

# a = n2label()
# b = n2superlabel()
# c = label2n()
# d = superlabel2n()
# print(a)
# print(b)
# print(c)
# print(d)
# write_yaml(a, '../../../config/coco.yaml')
