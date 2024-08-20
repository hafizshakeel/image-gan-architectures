"""
Enhanced Super-Resolution Generative Adversarial Networks
(ESRGAN) Implementation: https://arxiv.org/abs/1809.00219

This script contains the implementation of the ESRGAN model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import torch
from torch import nn
from torchsummary import summary


class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super(convBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))

class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super(DenseResidualBlock, self).__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()  # when you want to do a for loop in forward part, it's good to use ModuleList()
        # because those parameters will then be maintained by PyTorch ModuleList
        # --> https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        for i in range(5):
            self.blocks.append(
                convBlock(in_channels + channels * i, channels if i <= 3 else in_channels,
                          use_act=True if i <= 3 else False, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x):
        # concatenate diff convBlock (see DenseBlock internal structure)
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)  # concatenate along the channel dimension
        return self.residual_beta * out + x  # Concatenation bw dense blocks


class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super(RRDB, self).__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x  # Concatenation out of dense blocks with x (down arrow in fig.)


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_blocks=23):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, num_features, 3, 1, 1, bias=True)
        self.residuals = nn.Sequential(*[RRDB(num_features) for _ in range(num_blocks)])
        # Behind self.residuals there are a lot of conv layers bc we're creating 23 of them and all of them
        # are creating 3 DenseResidualBlock and all DenseResidualBlock are creating 5 conv blocks
        self.conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_features), UpsampleBlock(num_features),
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, in_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial  # final concatenation (down arrow top img)
        x = self.upsamples(x)
        return self.final(x)


# Discriminator from SRGAN network
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):  # see fig. discriminator part
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(convBlock(in_channels, feature, use_act=True, kernel_size=3, stride=1 + idx % 2, padding=1))
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),  # if inp is 96x96 divide that by each of stride=2 --> result 6x6.
            # It will also run for inp shape >96x96 like 128x128
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)  # no sigmoid here because we'll use BCEWithLogitsLoss which include sigmoid


def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale


def test():
    gen = Generator()
    disc = Discriminator()
    low_res = 24
    high_res = 96
    x = torch.randn((5, 3, low_res, low_res))
    gen_out = gen(x)
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)

    # Print summaries for both models
    summary(gen, (3, low_res, low_res))
    summary(disc, (3, high_res, high_res))


if __name__ == "__main__":
    test()
