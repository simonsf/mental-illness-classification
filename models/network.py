# Author: Kelei He (hkl@nju.edu.cn)
# Please contact author if you use
# Date: 24/04/2020

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.batchnorm import SynchronizedBatchNorm2d
import math

if torch.cuda.device_count() > 1:
    BatchNorm = SynchronizedBatchNorm2d
else:
    BatchNorm = nn.BatchNorm2d


cam_w = 100
cam_sigma = 0.4


def refine_cams(cam_original, image_shape, bag_size, sigmoid=True):
    cam_original = F.interpolate(
        cam_original, image_shape, mode="bilinear", align_corners=True
    )
    B, C, H, W = cam_original.size()
    cams = []
    max_pool = nn.MaxPool1d(kernel_size=bag_size, stride=bag_size)
    for idx in range(C):
        cam = cam_original[:, idx, :, :]
        cam = cam.view(B, -1)
        cam_min0 = cam.min(dim=1, keepdim=True)[0].unsqueeze(1)
        cam_max0 = cam.max(dim=1, keepdim=True)[0].unsqueeze(1)
        cam_max = max_pool(cam_max0.permute(2, 1, 0)).permute(2, 1, 0)
        cam_min = -1 * max_pool(-1 * cam_min0.permute(2, 1, 0)).permute(2, 1, 0)
        B0 = cam_max.shape[0]
        cam_max = cam_max.expand(B0, bag_size, 1).reshape(B, -1)
        cam_min = cam_min.expand(B0, bag_size, 1).reshape(B, -1)
        norm = cam_max - cam_min
        norm[norm == 0] = 1e-5
        cam = (cam - cam_min) / norm
        cam = cam.view(B, H, W).unsqueeze(1)
        cams.append(cam)
    cams = torch.cat(cams, dim=1)
    if sigmoid is True:
        cams = torch.sigmoid(cam_w * (cams - cam_sigma))
    return cams


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MILinMIL_Cls(nn.Module):
    def __init__(self, img_ch=3, out_cls_ch=2, bag_size=200, num_proto1=256, num_proto2=128):
        super(MILinMIL_Cls, self).__init__()
        self.bag_size = bag_size
        self.down_branch = Down_Branch(img_ch=img_ch)
        self.MIL = MILinMIL(in_ch=512, num_proto1=num_proto1, num_proto2=num_proto2,
                            out_cls_ch=out_cls_ch, bag_size=bag_size)


    def forward(self, x, return_feat=False):
        img_size = x.shape[2:]
        x5, _, _, _ = self.down_branch(x)
        #print(x5.shape)
        Pred_cls = self.MIL(x5, img_size, return_feat)
        return Pred_cls


class Down_Branch(nn.Module):
    def __init__(self, img_ch=3):
        super(Down_Branch, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=512)
        #self.Conv5 = conv_block(ch_in=64, ch_out=512)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        #x5 = self.Maxpool(x4)
        #x5 = self.Conv5(x5)

        return x4, x3, x2, x1


class Conv1d(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Conv1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias), self.weight

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MILinMIL(nn.Module):
    def __init__(self, in_ch=1024, out_cls_ch=2, num_proto1=256, num_proto2=128, bag_size=200):
        super(MILinMIL, self).__init__()
        self.num_proto1 = num_proto1
        self.num_proto2 = num_proto2
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=num_proto1, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=num_proto1, out_channels=num_proto1, kernel_size=1)
        self.fc = Conv1d(in_features=num_proto1, out_features=num_proto2, bias=True)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self。gap= nn.GlabalAveragePool2d()
        self.max = nn.MaxPool1d(kernel_size=bag_size, stride=bag_size, return_indices=True)
        self.classifier = Conv1d(in_features=num_proto2, out_features=out_cls_ch, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.bag_size = bag_size

    def forward(self, x, img_size, return_feat=False):  
        ins_feat1 = self.act(self.conv(x))
        ins_feat11 = self.conv1(ins_feat1)
        mean_feat1 = self.avg(ins_feat11)
        mean_feat1 = mean_feat1.squeeze(2)
        ins_feat2, fc_w0 = self.fc(mean_feat1)
        fc_w0 = fc_w0.detach().squeeze(2)

        max_feat2, indices = self.max(ins_feat2.permute(2, 1, 0))
        max_feat2 = max_feat2 + mean_feat1
        max_feat2 = max_feat2.permute(2, 1, 0)
        pred_cls, fc_w1 = self.classifier(max_feat2)
        fc_w1 = fc_w1.detach().squeeze(2)

        fc_w = fc_w1.mm(fc_w0)
        fc_w = fc_w.unsqueeze(2).unsqueeze(3)

        cam = self.act(F.conv2d(ins_feat11, fc_w, bias=None, stride=1, padding=0))
        new_cam = refine_cams(cam, img_size, sigmoid=False, bag_size=self.bag_size)

        if return_feat is not True:
            return pred_cls, new_cam, indices
        return pred_cls, new_cam, indices, max_feat2


if __name__ == '__main__':
    net = MILinMIL_Cls(img_ch=3, out_cls_ch=2, bag_size=23, num_proto1=256, num_proto2=128)
    print(net)