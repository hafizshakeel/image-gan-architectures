"""
Image-to-Image Translation with Conditional Adversarial Networks
(Pix2Pix) Implementation: https://arxiv.org/abs/1611.07004

This script contains the implementation of the Pix2Pix model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries

import torch
from torch import nn
from torchsummary import summary

class _convBlockD(nn.Module):
    """
    A convolutional block for the Discriminator.
    Applies a Conv2d -> BatchNorm2d -> LeakyReLU sequence.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
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
            nn.Conv2d(in_channels * 2, features[0], 4, 2, 1, padding_mode="reflect"),  # Output shape: features[0] x 128 x 128
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(_convBlockD(in_channels, feature, stride=1 if feature == features[-1] else 2))  # Adjust stride for the last layer
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode="reflect"))  # Output shape: 1 x 30 x 30
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)  # Concatenate input and target along channel dimension
        x = self.initial(x)
        x = self.model(x)
        return x

class _convBlockG(nn.Module):
    """
    A convolutional block for the Generator.
    Includes options for downsampling (Conv2d) or upsampling (ConvTranspose2d).
    """
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),  # Downsampling
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),  # Upsampling
                nn.BatchNorm2d(out_channels),
                nn.Dropout(p=0.5),
                nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    """
    U-Net-based Generator for image-to-image translation.
    """
    def __init__(self, in_channels, features):
        super(Generator, self).__init__()
        self.init_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),  # Output: features x 128 x 128
            nn.LeakyReLU(0.2)
        )
        # Encoder (Downsampling path)
        self.down1 = _convBlockG(features, features * 2, down=True)  # Output: (features*2) x 64 x 64
        self.down2 = _convBlockG(features * 2, features * 4, down=True)  # Output: (features*4) x 32 x 32
        self.down3 = _convBlockG(features * 4, features * 8, down=True)  # Output: (features*8) x 16 x 16
        self.down4 = _convBlockG(features * 8, features * 8, down=True)  # Output: (features*8) x 8 x 8
        self.down5 = _convBlockG(features * 8, features * 8, down=True)  # Output: (features*8) x 4 x 4
        self.down6 = _convBlockG(features * 8, features * 8, down=True)  # Output: (features*8) x 2 x 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),  # Output: (features*8) x 1 x 1
            nn.ReLU()
        )

        # Decoder (Upsampling path)
        self.up1 = _convBlockG(features * 8, features * 8, down=False)  # Output: (features*8) x 2 x 2
        self.up2 = _convBlockG(features * 8 * 2, features * 8, down=False)  # Output: (features*8) x 4 x 4
        self.up3 = _convBlockG(features * 8 * 2, features * 8, down=False)  # Output: (features*8) x 8 x 8
        self.up4 = _convBlockG(features * 8 * 2, features * 8, down=False)  # Output: (features*8) x 16 x 16
        self.up5 = _convBlockG(features * 8 * 2, features * 4, down=False)  # Output: (features*4) x 32 x 32
        self.up6 = _convBlockG(features * 4 * 2, features * 2, down=False)  # Output: (features*2) x 64 x 64
        self.up7 = _convBlockG(features * 2 * 2, features, down=False)  # Output: features x 128 x 128
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, 4, 2, 1),  # Output: in_channels x 256 x 256
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.init_down(x)
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
    print(preds_1.shape, preds_2.shape)  # disc out: 1x30x30, gen out: 3x256x256
    print(summary(disc, [(3, 256, 256), (3, 256, 256)]))
    print(summary(gen, (3, 256, 256)))

if __name__ == "__main__":
    test()
