"""
Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks
(CycleGAN) Implementation: https://arxiv.org/abs/1703.10593

This script contains the implementation of the CycleGAN model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import sys

import torch
import torch.nn as nn
from torchsummary import summary


"""Discriminator part of CycleGAN"""

class _convBlockD(nn.Module):
    """
    A convolutional block for the Discriminator.
    Applies a Conv2d -> BatchNorm2d -> LeakyReLU sequence.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),  # CycleGAN uses InstanceNorm2d everywhere
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Output: 1x30x30 grid for input size 256x256, where each grid cell represents a 70x70 patch in the input image.
    """
    def __init__(self, in_channels, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:  # except the first layer
            layers.append(_convBlockD(in_channels, feature, stride=1 if feature == features[-1] else 2))  # Adjust stride for the last layer
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode="reflect"))  # Output shape: 1 x 30 x 30
        self.disc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = torch.sigmoid(self.disc(x))
        return x

"""Generator part of CycleGAN"""

class _convBlockG(nn.Module):
    """
    A block for down convolution and up convolution operations.
    Applies a Conv2d or ConvTranspose2d -> InstanceNorm2d -> ReLU sequence.
    """
    def __init__(self, in_channels, out_channels, use_act=True, down=True, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    """
    Residual block for the Generator.
    Contains two convolutional layers with InstanceNorm2d and ReLU.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            _convBlockG(channels, channels, use_act=True, kernel_size=3, padding=1),
            _convBlockG(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)  # Adding the input (residual connection)

class Generator(nn.Module):
    """
    CycleGAN Generator model.
    Uses downsampling, residual blocks, and upsampling layers to transform images.
    """
    def __init__(self, img_channels, num_features, num_residuals):
        super(Generator, self).__init__()
        self.init = nn.Sequential(
            nn.Conv2d(img_channels, num_features, 7, 1, 3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList([
            _convBlockG(num_features, num_features * 2, use_act=True, down=True, kernel_size=3, stride=2, padding=1),
            _convBlockG(num_features * 2, num_features * 4, use_act=True, down=True, kernel_size=3, stride=2, padding=1)
        ])
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList([
            _convBlockG(num_features * 4, num_features * 2, use_act=True, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            _convBlockG(num_features * 2, num_features, use_act=True, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        ])
        self.last = nn.Sequential(
            nn.Conv2d(num_features, img_channels, 7, 1, 3, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = self.last(x)
        return x


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
