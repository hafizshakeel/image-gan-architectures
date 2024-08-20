"""
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
(SRGAN) Implementation: https://arxiv.org/abs/1609.04802

This script contains the implementation of the SRGAN model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries

import torch
import torch.nn as nn
from torchsummary import summary


class ConvBlock(nn.Module):
    # Conv-BN-PReLU/LeakyReLU
    def __init__(self, in_channels, out_channels, disc=False, use_act=True, use_bn=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)  # Bias will be false for use_bn=True
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()  # use_bn otherwise pass it through
        self.act = nn.LeakyReLU(0.2, inplace=True) if disc else nn.PReLU(num_parameters=out_channels)
        # nn.PRelu --> specify num of params such that each of these out channels have a separate slope

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch * scale_factor ** 2, 3, 1, 1)  # scale_factor = 2
        self.ps = nn.PixelShuffle(scale_factor)  # (in_ch * 4, H, W) --> in_c, H * 2, W * 2 make the W, H twice as large
        self.act = nn.PReLU(num_parameters=in_ch)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block1 = ConvBlock(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_ch, in_ch, use_act=False, kernel_size=3, stride=1, padding=1)  # fig.elementwise sum

    def forward(self, x):
        out = self.block1(x)
        out = self.block1(out)
        return out + x  # output + input


class Generator(nn.Module):
    def __init__(self, in_channels, num_features, num_blocks):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_features, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])  # list comprehension
        # to create all the 16 blocks and use * to unwrap the residual block in that list & turn into nn.Seq.
        # after B residual blocks
        self.convblock = ConvBlock(num_features, num_features, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_features, 2), UpsampleBlock(num_features, 2))
        self.final = nn.Conv2d(num_features, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 64, 128, 128, 256, 256, 512, 512]):  # see fig. discriminator part
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(ConvBlock(in_channels, feature,  disc=True, use_act=True, use_bn=False if idx == 0 else True,
                                    kernel_size=3, stride=1+idx % 2, padding=1))
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),  # if inp is 96x96 divide that by each of stride=2 --> result 6x6.
            # It will also run for inp shape >96x96
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)  # no sigmoid here because we'll use BCEWithLogitsLoss which include sigmoid


def test():
    low_resolution = 24  # Input resolution for Generator
    high_resolution = 96  # Output resolution for Generator and input resolution for Discriminator

    with torch.cuda.amp.autocast():
        x = torch.randn((1, 3, low_resolution, low_resolution))
        gen = Generator(in_channels=3, num_features=64, num_blocks=16)
        gen_out = gen(x)
        disc = Discriminator(in_channels=3)
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)

    # Print summaries for both models
    summary(gen, (3, low_resolution, low_resolution))
    summary(disc, (3, high_resolution, high_resolution))


if __name__ == "__main__":
    test()
