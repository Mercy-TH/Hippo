from torch import nn
from config.config import *


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, pre_batch_norm=True):
        super(ConvBlock, self).__init__()
        if pre_batch_norm:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_c),
                nn.LeakyReLU(),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),

                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(),

                nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(),
            )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        # self.down = nn.MaxPool2d(kernel_size=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)


class Final(nn.Module):
    def __init__(self, out_c):
        super(Final, self).__init__()
        self.final = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.Conv2d(out_c, num_classes, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.final(x)

