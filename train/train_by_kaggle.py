import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import *
from net.unetplusplus import UnetPlusPLus
from train.dataloader import DataSets

mode_path = './unetplusplus.pth'
data_path = '/kaggle/input/coco-2017-dataset/coco2017'
dataset = DataSets(data_path)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UnetPlusPLus(num_classes=num_classes, filters=filters, deep_supervision=deep_supervision).to(device)
if os.path.exists(mode_path):
    model.load_state_dict(torch.load(mode_path, map_location=device))
    print("Successful load weight.")
else:
    print("Not successful load weight.")
# optimizer = optim.Adam(model.parameters(), lr=lr)
initial_lr = 1e-1
optimizer = torch.optim.Adam(model.parameters(),lr = initial_lr)
scheduler_1 = StepLR(optimizer, step_size=40000, gamma=0.1)
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
        scheduler_1.step()
        if loss.item() < error:
            error = loss.item()
            torch.save(model.state_dict(), f'./best_model.pth')
        if i % 50 == 0:
            torch.save(model.state_dict(), mode_path)
            print(
                f' ======= epoch =====>> {e} ====== lr ====>> {optimizer.param_groups[0]["lr"]} === train_loss ====>> {loss.item()} ====== error ======>> {error} =======')
        if e % 100 == 0:
            e_mode_path = f'./unetplusplus_new_{e}.pth'
            torch.save(model.state_dict(), e_mode_path)