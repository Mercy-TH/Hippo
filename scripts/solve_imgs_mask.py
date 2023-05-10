import os
import shutil


img_dir = '/py/datas'
for temp in os.listdir(img_dir):
    name = temp.split('.')[0]
    print(temp)
    if '.' in temp:
        if temp.split('.')[1] == 'jpg':
            img_file = os.path.join(img_dir, f'{name}.jpg')
            shutil.copyfile(img_file,os.path.join(img_dir,'imgs', f'{name}.jpg'))
        if temp.split('.')[1] == 'png':
            mask_file = os.path.join(img_dir, f'{name}.png')
            shutil.copyfile(mask_file, os.path.join(img_dir,'masks', f'{name}.png'))

