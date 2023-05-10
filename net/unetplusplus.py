import torch
from net.unetplusplusblocks import *


class UnetPlusPLus(nn.Module):
    def __init__(self, num_classes, filters, deep_supervision=False):
        super(UnetPlusPLus, self).__init__()
        self.num_classes = num_classes
        self.filters = filters
        self.deep_supervision = deep_supervision
        self.down = DownSample()

        self.conv0_0 = ConvBlock(filters[0], filters[1], pre_batch_norm=False)
        self.conv1_0 = ConvBlock(filters[1], filters[2], pre_batch_norm=False)
        self.conv2_0 = ConvBlock(filters[2], filters[3], pre_batch_norm=False)
        self.conv3_0 = ConvBlock(filters[3], filters[4], pre_batch_norm=False)
        self.conv4_0 = ConvBlock(filters[4], filters[5], pre_batch_norm=False)

        self.conv0_1 = ConvBlock(filters[1]*2, filters[1], pre_batch_norm=True)
        self.conv0_2 = ConvBlock(filters[1]*2, filters[1], pre_batch_norm=True)
        self.conv0_3 = ConvBlock(filters[1]*2, filters[1], pre_batch_norm=True)
        self.conv0_4 = ConvBlock(filters[1]*2, filters[1], pre_batch_norm=True)

        self.conv1_1 = ConvBlock(filters[2]*2, filters[2], pre_batch_norm=True)
        self.conv1_2 = ConvBlock(filters[2]*2, filters[2], pre_batch_norm=True)
        self.conv1_3 = ConvBlock(filters[2]*2, filters[2], pre_batch_norm=True)

        self.conv2_1 = ConvBlock(filters[3]*2, filters[3], pre_batch_norm=True)
        self.conv2_2 = ConvBlock(filters[3]*2, filters[3], pre_batch_norm=True)

        self.conv3_1 = ConvBlock(filters[4]*2, filters[4], pre_batch_norm=True)

        self.up1_0 = UpSample(filters[2], filters[1])
        self.up1_1 = UpSample(filters[2], filters[1])
        self.up1_2 = UpSample(filters[2], filters[1])
        self.up1_3 = UpSample(filters[2], filters[1])

        self.up2_0 = UpSample(filters[3], filters[2])
        self.up2_1 = UpSample(filters[3], filters[2])
        self.up2_2 = UpSample(filters[3], filters[2])

        self.up3_0 = UpSample(filters[4], filters[3])
        self.up3_1 = UpSample(filters[4], filters[3])

        self.up4_0 = UpSample(filters[5], filters[4])
        if self.deep_supervision:
            self.f = nn.Sequential(
                nn.BatchNorm2d(filters[1]),
                nn.LeakyReLU(),
                nn.Conv2d(filters[1], num_classes, 3, padding=1),
            )
            self.final0_1 = Final(filters[1])
            self.final0_2 = Final(filters[1])
            self.final0_3 = Final(filters[1])
            self.final0_4 = Final(filters[1])
            self.out = nn.Sequential(
                nn.BatchNorm2d(filters[0]*4),
                nn.LeakyReLU(),
                nn.Conv2d(filters[0]*4, num_classes, 3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.final = Final(filters[1])

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.down(x0_0))
        x2_0 = self.conv2_0(self.down(x1_0))
        x3_0 = self.conv3_0(self.down(x2_0))
        x4_0 = self.conv4_0(self.down(x3_0))

        up1_0 = self.up1_0(x1_0)
        x0_1 = torch.cat([up1_0, x0_0], 1)
        x0_1 = self.conv0_1(x0_1)

        up2_0 = self.up2_0(x2_0)
        x1_1 = torch.cat([up2_0, x1_0], 1)
        x1_1 = self.conv1_1(x1_1)
        up1_1 = self.up1_1(x1_1)
        x0_2 = torch.cat([up1_1, x0_1], 1)
        x0_2 = self.conv0_2(x0_2)

        up3_0 = self.up3_0(x3_0)
        x2_1 = torch.cat([up3_0, x2_0], 1)
        x2_1 = self.conv2_1(x2_1)
        up2_1 = self.up2_1(x2_1)
        x1_2 = torch.cat([up2_1, x1_1], 1)
        x1_2 = self.conv1_2(x1_2)
        up1_2 = self.up1_2(x1_2)
        x0_3 = torch.cat([up1_2, x0_2], 1)
        x0_3 = self.conv0_3(x0_3)

        up4_0 = self.up4_0(x4_0)
        x3_1 = torch.cat([up4_0, x3_0], 1)
        x3_1 = self.conv3_1(x3_1)
        up3_1 = self.up3_1(x3_1)
        x2_2 = torch.cat([up3_1, x2_1], 1)
        x2_2 = self.conv2_2(x2_2)
        up2_2 = self.up2_2(x2_2)
        x1_3 = torch.cat([up2_2, x1_2], 1)
        x1_3 = self.conv1_3(x1_3)
        up1_3 = self.up1_3(x1_3)
        x0_4 = torch.cat([up1_3, x0_3], 1)
        x0_4 = self.conv0_4(x0_4)

        if self.deep_supervision:
            out_0_1 = self.f(x0_1)
            out_0_2 = self.f(x0_2)
            out_0_3 = self.f(x0_3)
            out_0_4 = self.f(x0_4)
            out = torch.cat([out_0_1, out_0_2, out_0_3, out_0_4], 1)
            out = self.out(out)
            return out
        else:
            return self.final(x0_4)


# if __name__ == "__main__":
#     import time
#     from py.unetplusplus.config.config import *
#     print("deep_supervision: False")
#     deep_supervision = False
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     inputs = torch.randn((1, 3, 256, 256)).to(device)
#     model = UnetPlusPLus(num_classes=num_classes, filters=filters, deep_supervision=deep_supervision).to(device)
#     s_time = time.time()
#     outputs = model(inputs)
#     end_time = time.time()
#     print(end_time-s_time)
#     print(outputs.shape)
#     del outputs
#
#     print("deep_supervision: True")
#     deep_supervision = True
#     model = UnetPlusPLus(num_classes=num_classes, filters=filters, deep_supervision=deep_supervision).to(device)
#     s_time = time.time()
#     outputs = model(inputs)
#     end_time = time.time()
#     print(end_time-s_time)
#     for out in outputs:
#         print(out.shape)
