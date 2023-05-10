import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import DataSets
from torchvision.utils import save_image
from net.unetplusplus import UnetPlusPLus
from config.config import *
from tqdm import tqdm
from scripts.loss import *

# torch.multiprocessing.set_start_method("spawn")
mode_path = '../models/unetplusplus_new.pth'
data_path = '/opt/projects/image_algorithm/src/segment/coco/cats'
train_image_dir = 'train_image'
dataset = DataSets(data_path)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UnetPlusPLus(num_classes=num_classes, filters=filters, deep_supervision=deep_supervision).to(device)
if os.path.exists(mode_path):
    model.load_state_dict(torch.load(mode_path, map_location=device))
    print("Successful load weight.")
else:
    print("Not successful load weight.")
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = nn.BCELoss()
loss_func.to(device)

model.to(device)
model.train()

error = 1000.
loss = 0.
for e in range(1, epoch):

    for i, (img, mask) in enumerate(tqdm(dataloader)):
        if img.shape == 0 or mask.shape == 0:
            continue
        loss_dic = {}
        img = img.to(device)
        mask = mask.to(device)
        y_hat = model(img)
        loss = loss_func(y_hat, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < error:
            error = loss.item()
        if i % 50 == 0:
            torch.save(model.state_dict(), mode_path)
            print(
                f' ======= epoch =====>> {e} ======== train_loss ====>> {loss.item()} ====== error ======>> {error} =======')
            # _img = img[0]
            # _mask = mask[0]
            # _out_img = y_hat[0]
            # image = torch.stack([_img, _mask, _out_img], dim=0)
            # save_image(image, f'{train_image_dir}/{i}.png')
        if e % 5 == 0:
            e_mode_path = f'../models/unetplusplus_new_{e}.pth'
            torch.save(model.state_dict(), e_mode_path)

