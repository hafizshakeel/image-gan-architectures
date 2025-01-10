"""
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
(SRGAN) Implementation: https://arxiv.org/abs/1609.04802

This script contains the implementation of the SRGAN model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

import torch
from torch import nn
from torchsummary import summary


"""Convolutional Block with optional BatchNorm and Activation"""

# Conv-BN-PReLU/LeakyReLU
class _convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, BN=True, act="prelu", **kwargs):
        super().__init__()
        # Convolutional layer with or without BatchNorm and activation
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=True if BN else False)
        self.bn = nn.BatchNorm2d(out_channels) if BN else nn.Identity()

        # Activation handling: "prelu" for PReLU, "leaky_relu" for LeakyReLU, or Identity for no activation
        if act == "prelu":
            self.act = nn.PReLU(num_parameters=out_channels)
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


"""Residual Block with two convolutional layers and skip connection"""

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act="prelu"):
        super().__init__()
        self.res_blocks = nn.Sequential(
            _convBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=act),
            _convBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=None),
        )

    def forward(self, x):
        return self.res_blocks(x) + x  # Skip connection (residual learning)


"""Upsampling block using PixelShuffle"""

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        # Convolutional layer followed by PixelShuffle for upscaling
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
        self.ps = nn.PixelShuffle(upscale_factor=upscale_factor)  # PixelShuffle to upscale
        self.act = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


"""Generator network"""

class Generator(nn.Module):
    def __init__(self, in_channels, num_features, num_res_block, act="prelu"):
        super().__init__()
        self.initial = _convBlock(in_channels, num_features, BN=False, act=act, kernel_size=9, stride=1, padding=4)
        self.res_block = nn.Sequential(*[
            ResBlock(num_features, num_features, act=act) for _ in range(num_res_block)
        ])  # List comprehension to create residual blocks
        self.conv_block = _convBlock(num_features, num_features, BN=True, act=None, kernel_size=3, stride=1, padding=1)

        # Upsampling using multiple UpSample blocks
        self.up_sample = nn.Sequential(
            UpSample(num_features, num_features),
            UpSample(num_features, num_features)
        )

        # Final convolution to return to the input channel size
        self.conv_final = nn.Conv2d(num_features, in_channels, 9, 1, 4)

    def forward(self, x):
        initial = self.initial(x)  # Initial feature extraction
        x = self.res_block(initial)  # Pass through residual blocks
        x = self.conv_block(x) + initial  # Add skip connection from initial layer
        x = self.up_sample(x)  # Upsample the feature map
        return torch.tanh(self.conv_final(x))  # Final output after convolution and tanh activation


"""Discriminator network"""

class Discriminator(nn.Module):
    def __init__(self, in_channels, features, act="leaky_relu"):
        super().__init__()

        layers = []
        for idx, feature in enumerate(features):  # List of features: [64, 64, 128, 128, 256, 256, 512, 512]
            self.in_channels = in_channels
            layers.append(_convBlock(in_channels, feature, BN=False if idx == 0 else True, act=act,
                                     kernel_size=3, stride=1 + idx % 2, padding=1))  # Stride even or odd
            in_channels = feature

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),  # Adaptive pooling to a fixed size
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),  # Output for binary classification: real vs fake
            # nn.Sigmoid()  # No Sigmoid, use BCEWithLogitsLoss during training
        )

    def forward(self, x):
        x = self.layers(x)  # Pass through convolution layers
        return self.linear(x)  # Output the final binary classification score


# Weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

