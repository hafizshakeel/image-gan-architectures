"""
Image-to-Image Translation with Conditional Adversarial Networks
(Pix2Pix) Implementation: https://arxiv.org/abs/1611.07004

This script contains the implementation of the Pix2Pix model.

Author: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import sys
import torch
import torch.nn as nn
from torchsummary import summary


"""Discriminator part of Pix2Pix"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]):  # inp 256 --> 1x30x30 out
        super(Discriminator, self).__init__()
        # Discriminator architecture is: C64-C128-C256-C512
        self.initial = nn.Sequential(
            # in_channels*2 since input is (x + y) or (inp_img + target_img) <-- concatenate along the channels &
            # based on it tells the patch of specific region is real or fake
            nn.Conv2d(in_channels * 2, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:  # first layer isn't included
            layers.append(ConvBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)),
            in_channels = feature
        # last conv layer to map to 1-dimensional output (real or fake) - see architecture diagram
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode="reflect"))

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)  # dim=1 along channels
        x = self.initial(x)
        x = self.model(x)
        return x


"""Generator part of Pix2Pix"""
# Block for down-sampling and up-sampling
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels, features):
        super(Generator, self).__init__()
        # U-Net Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )  # out 128x128 after stride = 2
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)  # out 64x64
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)  # 32x32
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)  # 16x16
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)  # 8x8
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)  # 4x4
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)  # 2x2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),  # 1x1
            nn.ReLU(),
        )
        # U-Net Decoder with skips: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)  # out 2x2
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)  # 4x4; *2 for concat
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)  # 8x8
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)  # 16x16
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=True)  # 32x32
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=True)  # 64x64
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=True)  # 128x128
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, 4, 2, 1),
            nn.Tanh(),
        )  # 256x256

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def test():
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    disc = Discriminator(in_channels=3)
    gen = Generator(in_channels=3, features=64)
    preds_1 = disc(x, y)
    preds_2 = gen(x)
    print(preds_1.shape, preds_2.shape)  # disc out: 1x30x30 meaning each value in the 30x30 grid sees
    # a 70x70 patch in the original image & gen out: 3x256x256
    print(summary(disc, [(3, 256, 256), (3, 256, 256)]))
    print(summary(gen, (3, 256, 256)))


if __name__ == "__main__":
    # sys.exit()
    test()
