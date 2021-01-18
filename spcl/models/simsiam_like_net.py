from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
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

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
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

        self.projection_layer = projection_MLP(2048, 2048, 2048)
        self.prediction_layer = prediction_MLP(2048)

        if not pretrained:
            self.reset_params()

    # def forward(self, x1, x2):
    #     x1, x2 = self.base(x1), self.base(x2)  # b, 2048, 16, 8
    #     x1, x2 = self.gap(x1), self.gap(x2)
    #     x1 = x1.view(x1.size(0), -1)
    #     x2 = x2.view(x2.size(0), -1)
    #
    #     z1, z2 = self.projection_layer(x1), self.projection_layer(x2)
    #     p1, p2 = self.prediction_layer(z1), self.prediction_layer(z2)
    #
    #     z1, z2 = F.normalize(z1), F.normalize(z2)
    #     p1, p2 = F.normalize(p1), F.normalize(p2)
    #
    #     zs = torch.cat([z1, z2], 0)
    #     ps = torch.cat([p1, p2], 0)
    #
    #
    #     if self.training is False:
    #         return torch.cat([z1, p1], -1)
    #
    #     return torch.cat([zs, ps], -1)

    def forward(self, x):
        x = self.base(x) # b, 2048, 16, 8
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        z = self.projection_layer(x)
        p = self.prediction_layer(z)

        z = F.normalize(z)
        p = F.normalize(p)

        if self.training is False:
            return torch.cat([z, p], -1)[:x.size(0)//2, :]

        return torch.cat([z, p], -1)


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
