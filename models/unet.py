import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchstat import stat
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain
from models import resnet


def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)

        if (x.size(2) != x_copy.size(2)) or (x.size(3) != x_copy.size(3)):
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                                  mode="bilinear", align_corners=True)
            else:
                # Padding in case the incomping volumes are of different sizes
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2))

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=True)


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


def change(x):
    x1 = torch.where(x == 1, torch.full_like(x, 1 - (1e-5)), x)
    x2 = torch.where(x1 == 0, torch.full_like(x1, (1e-5)), x1)

    return x2


class UNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(UNet, self).__init__()

        self.start_conv = x2conv(in_channels, 64)
        self.down1 = encoder(64, 128)
        self.down2 = encoder(128, 256)
        self.down3 = encoder(256, 512)
        self.down4 = encoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)

        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()
        self.predict_1 = nn.Conv2d(64, 1, 3, 1, 1)
        self.predict_2 = nn.Conv2d(64, 1, 3, 1, 1)
        self.predict_3 = nn.Conv2d(128, 1, 3, 1, 1)
        self.predict_4 = nn.Conv2d(256, 1, 3, 1, 1)

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.conv_a4 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_a3 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_a2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # EAO
        self.conv_epau = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 3, padding=1),
            nn.Sigmoid()
        )

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.middle_conv(self.down4(x4))

        # x = self.up1(x4, x)
        # x = self.up2(x3, x)
        # x = self.up3(x2, x)
        # x = self.up4(x1, x)
        #
        # x = self.final_conv(x)
        # return x

        # FFM
        x4 = self.up1(x4, x)
        p4 = x4
        x3 = self.up2(x3, x4)
        p3 = x3
        p4 = F.interpolate(p4, size=(90, 160), mode='bilinear', align_corners=True)
        p4 = self.conv1(p4)
        p3 = p3 + p4
        x2 = self.up3(x2, x3)
        p2 = x2
        p3 = F.interpolate(p3, size=(180, 320), mode='bilinear', align_corners=True)
        p3 = self.conv2(p3)
        p2 = p2 + p3
        x1 = self.up4(x1, x2)
        p1 = x1
        p2 = F.interpolate(p2, size=(360, 640), mode='bilinear', align_corners=True)
        p2 = self.conv3(p2)
        p1 = p1 + p2

        # MSFA
        x4_c = F.interpolate(self.conv_a4(x4), scale_factor=8, align_corners=False, mode='bilinear')
        x3_c = F.interpolate(self.conv_a3(x3), scale_factor=4, align_corners=False, mode='bilinear')
        x2_c = F.interpolate(self.conv_a2(x2), scale_factor=2, align_corners=False, mode='bilinear')
        x1_c = self.conv4(x1)
        x_c = x4_c + x3_c + x2_c + x1_c

        x = self.final_conv(x_c)
        # EAO
        x1 = self.conv_epau(x4_c)
        x2 = self.conv_epau(x3_c)
        x3 = self.conv_epau(x2_c)
        x4 = self.conv_epau(x1_c)
        attention_map_1 = torch.sigmoid(self.predict_1(p1))
        edge_1 = torch.abs(F.avg_pool2d(attention_map_1, kernel_size=3, stride=1, padding=1) - attention_map_1)
        attention_map_2 = torch.sigmoid(self.predict_2(p2))
        edge_2 = torch.abs(F.avg_pool2d(attention_map_2, kernel_size=3, stride=1, padding=1) - attention_map_2)
        attention_map_3 = torch.sigmoid(self.predict_3(p3))
        edge_3 = torch.abs(F.avg_pool2d(attention_map_3, kernel_size=3, stride=1, padding=1) - attention_map_3)
        attention_map_4 = torch.sigmoid(self.predict_4(p4))
        edge_4 = torch.abs(F.avg_pool2d(attention_map_4, kernel_size=3, stride=1, padding=1) - attention_map_4)
        return x, change(x1), change(x2), change(x3), change(x4), edge_1, edge_2, edge_3, edge_4
        # return x, edge_1
        # return x, change(x1), change(x2), change(x3), change(x4), edge_1, edge_2, edge_3, edge_4
        # return x, change(x1), change(x2), change(x3), change(x4), edge_1
        # return x, change(x1), change(x2), change(x3), change(x4)
        # return x, edge_1, edge_2, edge_3, edge_4

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


"""
-> Unet with a resnet backbone
"""


class UNetResnet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet50', pretrained=True, freeze_bn=False,
                 freeze_backbone=False, **_):
        super(UNetResnet, self).__init__()
        model = getattr(resnet, backbone)(pretrained, norm_layer=nn.BatchNorm2d)

        self.initial = list(model.children())[:4]
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # decoder
        self.conv1 = nn.Conv2d(2048, 192, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(192, 128, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(1152, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 96, 4, 2, 1, bias=False)

        self.conv3 = nn.Conv2d(608, 96, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 48, 4, 2, 1, bias=False)

        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.upconv5 = nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=False)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

        initialize_weights(self)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x1 = self.layer1(self.initial(x))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.upconv1(self.conv1(x4))
        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(self.conv2(x))

        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(self.conv3(x))

        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x1], dim=1)

        x = self.upconv4(self.conv4(x))

        x = self.upconv5(self.conv5(x))

        # if the input is not divisible by the output stride
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        x = self.conv7(self.conv6(x))
        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(),
                     self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.conv1.parameters(), self.upconv1.parameters(), self.conv2.parameters(),
                     self.upconv2.parameters(),
                     self.conv3.parameters(), self.upconv3.parameters(), self.conv4.parameters(),
                     self.upconv4.parameters(),
                     self.conv5.parameters(), self.upconv5.parameters(), self.conv6.parameters(),
                     self.conv7.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':
    model = UNet(num_classes=2)
    # stat(model, (3, 360, 640))