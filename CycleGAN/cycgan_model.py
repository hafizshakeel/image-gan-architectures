"""
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
(CycleGAN) Implementation: https://arxiv.org/abs/1703.10593

This script contains the implementation of the CycleGAN model.

Author: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import sys

import torch
import torch.nn as nn
from torchsummary import summary


"""Discriminator part of CycleGAN"""
class DConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),  # cycleGAN use InstanceNorm2d everywhere
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]):  # inp 256 --> 1x30x30 out
        super(Discriminator, self).__init__()
        # Discriminator architecture is: C64-C128-C256-C512
        self.initial = nn.Sequential(
            # Here in_channels are not multiplied by 2 as we did in the Pix2Pix --> cycleGANs are unsupervised
            nn.Conv2d(in_channels, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:  # since first layer isn't included
            layers.append(DConvBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)),
            in_channels = feature
        # last conv layer to map to 1-dimensional output (real or fake) - see architecture diagram
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode="reflect"))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = torch.sigmoid(self.model(x))
        return x


"""Generator part of CycleGAN"""
# Block for down-sampling and up-sampling
class GConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):  # **kwargs for kernel, str, pad.
        super(GConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),  # inplace=True for additional performance
            # benefits. And use of Identity meaning pass it through and not do anything to what's input
        )

    def forward(self, x):
        return self.conv(x)


# Residual blocks
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            GConvBlock(channels, channels, kernel_size=3, padding=1),
            GConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)  # since residual and also same channels and same conv


class Generator(nn.Module):
    def __init__(self, img_channels, num_features, num_residuals):  # num_residual=9 for img size 256 x 256
        super(Generator, self).__init__()
        # 9 residual block c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128, u64,c7s1-3
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, 7, 1, 3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList([
            GConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
            GConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1)
        ])
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]  # * for unwrapping the list of all
            # residual blocks
        )
        self.up_blocks = nn.ModuleList([
            GConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            # output_padding controls the additional size added to one side of the output shape.
            # out_padding=1 gives error in my implementation
            GConvBlock(num_features * 2, num_features * 1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])
        self.last = nn.Sequential(
            nn.Conv2d(num_features * 1, img_channels, 7, 1, 3, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)  # already for loop is used.
        for layer in self.up_blocks:
            x = layer(x)
        return self.last(x)


def test():
    x = torch.randn(1, 3, 256, 256)  # for one example
    disc = Discriminator(in_channels=3)
    gen = Generator(3, 64)
    pred_1 = disc(x)
    pred_2 = gen(x)
    print(pred_1.shape, pred_2.shape)   # disc out: 1x30x30 meaning each value in the 30x30 grid sees
    # a 70x70 patch in the original image & gen out: 3x256x256
    print(summary(disc, (3, 256, 256)))
    print(summary(gen, (3, 256, 256)))


if __name__ == "__main__":
    # sys.exit()
    test()
