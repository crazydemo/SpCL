from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import random
import torchvision
import torch
from torchvision.models.resnet import Bottleneck
from .slot_attention_with_pos_emb import SoftPositionEmbed, SlotAttention

from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a

__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x

class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat1 = nn.Linear(out_planes, self.num_features)
                self.feat2 = nn.Linear(out_planes, self.num_features)
                self.feat_bn1 = nn.BatchNorm1d(self.num_features)
                self.feat_bn2 = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat1.weight, mode='fan_out')
                init.constant_(self.feat1.bias, 0)
                init.kaiming_normal_(self.feat2.weight, mode='fan_out')
                init.constant_(self.feat2.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn1 = nn.BatchNorm1d(self.num_features)
                self.feat_bn2 = nn.BatchNorm1d(self.num_features)
            self.feat_bn1.bias.requires_grad_(False)
            self.feat_bn2.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier1 = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier1.weight, std=0.001)
                self.classifier2 = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier2.weight, std=0.001)
        init.constant_(self.feat_bn1.weight, 1)
        init.constant_(self.feat_bn1.bias, 0)
        init.constant_(self.feat_bn2.weight, 1)
        init.constant_(self.feat_bn2.bias, 0)

        # self.posemb = SoftPositionEmbed(self.num_features, (16, 8))
        # self.slot_att = SlotAttention(8, self.num_features, hidden_dim=256)
        # self.layernorm = nn.LayerNorm(self.num_features)
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(self.num_features, self.num_features//4, 3),
        #     nn.BatchNorm2d(self.num_features//4),
        #     nn.ReLU(),
        #     nn.Conv2d(self.num_features//4, self.num_features, 3))
        # self.mlp.apply(weights_init_kaiming)
        # self.layernorm.apply(weights_init_kaiming)

        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_crop = BatchDrop(0.33, 1.0)

        self.res_part2 = Bottleneck(2048, 512)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        x_ = self.base(x)  # b, 2048, 16, 8
        b, c, h, w = x_.size()

        x = self.gap(x_)
        x = x.view(x.size(0), -1)  # b, 2048

        x_bdb = x_.detach()
        x_bdb = self.res_part2(x_bdb)
        x_bdb = self.batch_crop(x_bdb)
        # print(x_bdb.size())
        x_bdb = self.part_maxpool(x_bdb)
        x_bdb = x_bdb.view(x_bdb.size(0), -1)
        # print(x_bdb.size())

        if self.cut_at_pooling:
            return torch.cat([x, x_bdb], 1)

        if self.has_embedding:
            bn_x1 = self.feat_bn1(self.feat1(x))
            bn_x2 = self.feat_bn2(self.feat2(x_bdb))
        else:
            bn_x1 = self.feat_bn1(x)
            bn_x2 = self.feat_bn2(x_bdb)
            # bn_x2 = x_slot

        if self.training is False:
            bn_x1 = F.normalize(bn_x1)
            bn_x2 = F.normalize(bn_x2)
            return torch.cat([bn_x1, bn_x2], 1)

        if self.norm:
            bn_x1 = F.normalize(bn_x1)
            bn_x2 = F.normalize(bn_x2)
        elif self.has_embedding:
            bn_x1 = F.relu(bn_x1)
            bn_x2 = F.relu(bn_x2)

        if self.dropout > 0:
            bn_x1 = self.drop(bn_x1)
            bn_x2 = self.drop(bn_x2)

        if self.num_classes > 0:
            prob1 = self.classifier1(bn_x1)
            prob2 = self.classifier2(bn_x2)
        else:
            return torch.cat([bn_x1, bn_x2], 1)

        return torch.cat([prob1, prob2], -1)

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNetIBN.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[4].load_state_dict(resnet.layer1.state_dict())
        self.base[5].load_state_dict(resnet.layer2.state_dict())
        self.base[6].load_state_dict(resnet.layer3.state_dict())
        self.base[7].load_state_dict(resnet.layer4.state_dict())


def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)
