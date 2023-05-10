import os
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import DataSets
from torchvision.utils import save_image
from py.unetplusplus.net.unetplusplus import UnetPlusPLus
from py.unetplusplus.config.config import *
from tqdm import tqdm
from py.unetplusplus.scripts.loss import *

# torch.multiprocessing.set_start_method("spawn")
mode_path = '../models/unetplusplus.pth'
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
# loss_func = nn.CrossEntropyLoss()
# loss_func = WceLoss()
loss_func = nn.BCELoss()
loss_func.to(device)

model.to(device)
model.train()

error = 1000.
loss = 0.
for e in range(1, epoch):

    for i, (img, mask) in enumerate(tqdm(dataloader)):
        if img is None and mask is None:
            continue
        # counts = torch.bincount(mask.long().flatten(), minlength=3)
        # w = (counts.sum() - counts) / counts.sum()
        # loss_func = WceLoss(w).to(device)

        # loss_func.weight = w
        loss_dic = {}
        img = img.to(device)
        mask = mask.to(device)
        y_hat = model(img)
        tmp = [tt.detach().cpu().numpy() for tt in y_hat]
        if deep_supervision is False:
            loss = loss_func(y_hat, mask)
        if deep_supervision is True:
            L1 = loss_func(y_hat[0], mask).sum()
            L2 = loss_func(y_hat[1], mask).sum()
            L3 = loss_func(y_hat[2], mask).sum()
            L4 = loss_func(y_hat[3], mask).sum()
            loss = alphas[0] * L1 + alphas[1] * L2 + alphas[2] * L3 + alphas[3] * L4
            loss_dic = {
                'L1': L1,
                'L2': L2,
                'L3': L3,
                'L4': L4,
                'loss': loss.item()
            }
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < error:
            error = loss.item()
        if i % 50 == 0:
            torch.save(model.state_dict(), mode_path)
            if deep_supervision is True:
                print(
                    f"""
                    epoch:      {e},
                    L1:         {loss_dic['L1'].data}
                    L2:         {loss_dic['L2'].data}
                    L3:         {loss_dic['L3'].data}
                    L4:         {loss_dic['L4'].data}
                    loss:       {loss.item()}
                    error:      {error}
                    """
                )
                for y_i, y_ in enumerate(y_hat):
                    y_hat_tensor = torch.split(y_, 1, dim=0)
                    for b_i, b in enumerate(y_hat_tensor):
                        _img = img[b_i]
                        _mask = mask[b_i]
                        _out_img = b[0]
                        image = torch.stack([_img, _mask, _out_img], dim=0)
                        save_image(image, f'{train_image_dir}/{i}_{y_i}_{b_i}.png')
            else:
                print(
                    f' ======= epoch =====>> {e} ======== train_loss ====>> {loss.item()} ====== error ======>> {error} =======')
                _img = img[0]
                _mask = mask[0]
                _out_img = y_hat[0]
                image = torch.stack([_img, _mask, _out_img], dim=0)
                save_image(image, f'{train_image_dir}/{i}.png')
        if e % 5 == 0:
            e_mode_path = f'../models/unetplusplus_{e}.pth'
            torch.save(model.state_dict(), e_mode_path)

