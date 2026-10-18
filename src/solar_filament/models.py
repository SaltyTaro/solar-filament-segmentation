"""Small segmentation models retaining native-resolution spatial features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, inputs, skip):
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([inputs, skip], dim=1))


class ResNetUNet(nn.Module):
    def __init__(self, pretrained: bool = True, input_channels: int = 1):
        super().__init__()
        encoder = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        gray_conv = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Summing reproduces a convolution on three replicated gray channels.
            gray_conv.weight.zero_()
            gray_conv.weight[:, :1].copy_(encoder.conv1.weight.sum(dim=1, keepdim=True))
        self.stem = nn.Sequential(gray_conv, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.decode4 = DecoderBlock(512, 256, 256)
        self.decode3 = DecoderBlock(256, 128, 128)
        self.decode2 = DecoderBlock(128, 64, 64)
        self.decode1 = DecoderBlock(64, 64, 32)
        self.full_resolution = ConvBlock(32 + input_channels, 24)
        self.head = nn.Conv2d(24, 1, 1)

    def forward(self, inputs):
        stem = self.stem(inputs)
        level1 = self.layer1(self.pool(stem))
        level2 = self.layer2(level1)
        level3 = self.layer3(level2)
        level4 = self.layer4(level3)
        decoded = self.decode4(level4, level3)
        decoded = self.decode3(decoded, level2)
        decoded = self.decode2(decoded, level1)
        decoded = self.decode1(decoded, stem)
        decoded = F.interpolate(decoded, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.full_resolution(torch.cat([decoded, inputs], dim=1)))
