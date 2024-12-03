from base import BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.helpers import initialize_weights, set_trainable
from itertools import chain

''' 
-> ResNet BackBone
'''


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def norm_layer(channel, norm_name='bn'):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


# Multi-scale localization Block
class MSLB(nn.Module):
    def __init__(self, channel):
        super(MSLB, self).__init__()
        temp_c = channel // 4
        self.query_conv = nn.Conv2d(in_channels=channel, out_channels=temp_c, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channel, out_channels=temp_c, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.local1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.local2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.local3 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 2, dilation=2, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )

        # residual connection
        self.conv_res = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )

    def forward(self, x):
        # non-local
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        # local
        branch1 = self.local1(x)
        branch2 = self.local2(x)
        branch3 = self.local3(x)
        out2 = self.conv_cat(torch.cat([branch1, branch2, branch3], dim=1))

        out = F.relu(x + self.conv_res(out1 + out2))
        return out


def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=True)


class ChannelCompress(nn.Module):
    def __init__(self, in_c, out_c):
        super(ChannelCompress, self).__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.reduce(x)


# Global Localization Module
class GLM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GLM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.locate1 = MSLB(256)
        self.locate2 = MSLB(256)
        self.locate3 = MSLB(256)
        self.compress3 = ChannelCompress(2048, 256)
        self.compress2 = ChannelCompress(1024, 256)
        self.compress1 = ChannelCompress(512, 256)
        self.predict = nn.Conv2d(256, 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.branch0(x)
        x2 = self.branch1(x)
        x3 = self.branch2(x)
        x4 = self.branch3(x)

        # x2 = self.compress1(x2)
        # x3 = self.compress2(x3)
        # x4 = self.compress3(x4)

        # x4 = self.locate1(x4)
        x3 = x3 + upsample(x4, x3.shape[2:])
        # x3 = self.locate2(x3)
        x2 = x2 + upsample(x3, x2.shape[2:])
        # x2 = self.locate3(x2)

        attention_map = torch.sigmoid(self.predict(x2))
        edge = torch.abs(F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1) - attention_map)

        return attention_map, edge


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features


''' 
-> (Aligned) Xception BackBone
Pretrained model from https://github.com/Cadene/pretrained-models.pytorch
by Remi Cadene
'''

''' 
-> Decoder
'''


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48 + 64, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x


class RFB(nn.Module):
    """ receptive field block """

    def __init__(self, in_channel, out_channel=256):
        super(RFB, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)  # 当kernel=3，如果dilation=padding则shape不变
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        out = self.relu(x_cat + self.conv_res(x))
        return out


# Boundary Aware Module
class BAM(nn.Module):
    def __init__(self, in_c, out_c, groups=8):
        super(BAM, self).__init__()
        self.rfb = RFB(in_c, out_c)
        self.groups = groups
        sc_channel = (out_c // groups + 1) * groups  # split then concate channel

        self.foreground_conv = nn.Conv2d(sc_channel, sc_channel, 3, 1, 1, bias=False)
        self.foreground_bn = norm_layer(sc_channel)
        self.foreground_relu = nn.ReLU()
        self.background_conv = nn.Conv2d(sc_channel, sc_channel, 3, 1, 1, bias=False)
        self.background_bn = norm_layer(sc_channel)
        self.background_relu = nn.ReLU()
        self.compress1 = ChannelCompress(2048, 256)
        self.compress2 = ChannelCompress(256, 64)

        self.edge_conv = nn.Sequential(
            nn.Conv2d(sc_channel, out_c, 3, 1, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True)
        )
        self.mask_conv = nn.Sequential(
            nn.Conv2d(2 * sc_channel, out_c, 3, 1, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True)
        )
        self.mask_pred_conv = nn.Conv2d(out_c, 1, 3, 1, 1)
        self.edge_pred_conv = nn.Conv2d(out_c, 1, 3, 1, 1)

    def split_and_concate(self, x1, x2):
        N, C, H, W = x1.shape
        x2 = x2.repeat(1, self.groups, 1, 1)
        x1 = x1.reshape(N, self.groups, C // self.groups, H, W)
        x2 = x2.unsqueeze(2)
        x = torch.cat([x1, x2], dim=2)
        x = x.reshape(N, -1, H, W)
        return x

    def forward(self, low, high, mask_pred, edge_pred, sig=True):
        low = self.rfb(low)
        high = self.compress1(high)
        high = self.compress2(high)
        if high is not None:
            low += upsample(high, low.shape[2:])
        mask_pred = upsample(mask_pred, low.shape[2:])
        edge_pred = upsample(edge_pred, low.shape[2:])
        if sig:
            mask_pred = torch.sigmoid(mask_pred)
            edge_pred = torch.sigmoid(edge_pred)
        foreground = low * mask_pred
        background = low * (1 - mask_pred)

        foreground = self.foreground_conv(self.split_and_concate(foreground, edge_pred))
        background = self.background_conv(self.split_and_concate(background, edge_pred))

        edge_feature = (foreground - foreground.min()) / (foreground.max() - foreground.min()) * (
                background - background.min()) / (background.max() - background.min())

        foreground = self.foreground_relu(self.foreground_bn(foreground))
        background = self.background_relu(self.background_bn(background))
        mask_feature = torch.cat((foreground, background), dim=1)

        edge_feature = self.edge_conv(edge_feature)
        mask_feature = self.mask_conv(mask_feature)

        mask = self.mask_pred_conv(mask_feature)
        edge = self.edge_pred_conv(edge_feature)
        return mask_feature, mask, edge


'''
-> EdgeSeg
'''


class EdgeSeg(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='xception', pretrained=True,
                 output_stride=16, freeze_bn=False, freeze_backbone=False, **_):

        super(EdgeSeg, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256

        self.decoder = Decoder(low_level_channels, num_classes)
        self.GCM = GLM(2048, 256)
        self.cnn = nn.Conv2d(2048, 256, (1, 1))
        self.refine = BAM(256, 64, 1)

        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        attention_map, edge = self.GCM(x)
        x_refined, pred, edge = self.refine(low_level_features, x, attention_map, edge, sig=False)
        x = self.decoder(x_refined, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x, edge

    # Two functions to yield the parameters of the backbone
    # & Decoder / ASSP to use differentiable learning rates
    # FIXME: in xception, we use the parameters from xception and not aligned xception
    # better to have higher lr for this backbone

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.GCM.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
