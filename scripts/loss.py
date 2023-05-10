import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WceLoss(nn.Module):
    """
    带权交叉熵
    :return:
    """
    def __init__(self, weight=None):
        super(WceLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, y, y_hat):
        return self.loss(y, y_hat)

    def aaa(self):
        y = torch.Tensor([1,0,0,0,0,1]).long()
        y_hat = torch.Tensor([1,0,1,0,0,1]).float()
        y = y.view((3,2))
        y_hat = y_hat.view((3,2))

        counts = torch.bincount(y.flatten(), minlength=2)
        w = (counts.sum() - counts) / counts.sum()
        w = WceLoss(w)
        loss = w(y.float(),y_hat)
        print(loss)
        print(nn.CrossEntropyLoss()(y.float(), y_hat))



def dice_loss():
    return


def iou_loss():
    return


def tversky_loss():
    return


# y_hat = torch.rand(size=(1,3,3))
# y = torch.randint(0,2,size=(1,3,3))
# print(y_hat)
# print(y)
# counts = torch.bincount(y.flatten(), minlength=3)
# w = (counts.sum() - counts) / counts.sum()
# wceloss = WceLoss(w)
# loss = wceloss(y.float(), y_hat)
# print(loss)
# loss = nn.CrossEntropyLoss(reduction='none')(y.float(), y_hat)
# print(loss)