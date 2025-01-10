"""
Enhanced Super-Resolution Generative Adversarial Networks
(ESRGAN) Implementation: https://arxiv.org/abs/1809.00219

This script contains the implementation of the ESRGAN model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import torch
import torch.nn as nn
from torchsummary import summary


"""Convolutional Block with optional Activation"""

# Convolutional Block with optional activation (PReLU/LeakyReLU/Identity)
class _convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act="leaky_relu", **kwargs):
        super().__init__()
        # Convolutional layer with specified kernel size, stride, padding, etc.
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

        # Activation function selection
        if act == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == "prelu":
            self.act = nn.PReLU(num_parameters=out_channels)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


""" Deep Residual Block with five convolutional layers and skip connection"""

class DeepResBlock(nn.Module):
    def __init__(self, in_channels, features=32, act="leaky_relu", beta=0.2):
        super().__init__()
        self.beta = beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            # Adding convolutional blocks with increasing input channels for concatenation
            self.blocks.append(
                _convBlock(in_channels + features * i, features if i <= 3 else in_channels,
                           kernel_size=3, stride=1, padding=1, act=act if i <= 3 else False)
            )

    def forward(self, x):
        initial_input = x
        for block in self.blocks:
            out = block(initial_input)
            # Concatenate output with previous inputs along the channel dimension
            initial_input = torch.cat([initial_input, out], dim=1)
        return (self.beta * out) + x  # Apply skip connection with scaling


"""Residual-in-Residual Dense Block (RRDB)"""

class RRDB(nn.Module):
    def __init__(self, in_channels, beta=0.2):
        super().__init__()
        self.beta = beta
        # Three sequential DeepResBlocks within a Residual block - see fig.4
        self.res_dense_block = nn.Sequential(*[DeepResBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        out = self.res_dense_block(x)
        return (self.beta * out) + x  # Residual scaling and skip connection


"""Upsampling block using UpSample"""

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        # Convolution followed by nearest neighbor upsampling
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.ps = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


"""Generator network"""

class Generator(nn.Module):
    def __init__(self, in_channels, num_features, num_res_block, act="prelu"):
        super().__init__()
        # Initial convolutional block
        self.initial = _convBlock(in_channels, num_features, kernel_size=9, stride=1, padding=4, act=act)

        # Series of residual blocks
        self.res_blocks = nn.Sequential(*[
            DeepResBlock(num_features) for _ in range(num_res_block)
        ])

        # Convolutional block for skip connection
        self.conv_block = _convBlock(num_features, num_features, kernel_size=3, stride=1, padding=1, act=None)

        # Upsampling to higher resolution
        self.up_sample = nn.Sequential(
            UpSample(num_features), UpSample(num_features)
        )

        # Final convolutional layers to return to original channel size
        self.final = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, in_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        initial = self.initial(x)  # Initial feature extraction
        x = self.res_blocks(initial)  # Pass through residual blocks
        x = self.conv_block(x) + initial  # Add skip connection from initial layer
        x = self.up_sample(x)  # Upsample the feature map
        return self.final(x)  # Final output


"""Discriminator network"""

class Discriminator(nn.Module):
    def __init__(self, in_channels, features, act="leaky_relu"):
        super().__init__()
        layers = []
        for idx, feature in enumerate(features):
            # Adding convolutional blocks with varying stride for downsampling
            layers.append(_convBlock(in_channels, feature, act=act, kernel_size=3, stride=1 + idx % 2, padding=1))
            in_channels = feature  # Update input channels for the next layer

        self.layers = nn.Sequential(*layers)  # Sequential model of convolutional layers

        # Linear layers for classification
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),  # Pool to fixed 6x6 size
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),  # Fully connected layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),  # Output layer for binary classification (real vs fake)
        )

    def forward(self, x):
        x = self.layers(x)  # Pass through convolution layers
        return self.linear(x)  # Final binary classification output


# Function for initializing weights
# Uses Kaiming initialization for Conv2d layers and sets BatchNorm weights and biases
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# Test function to verify model outputs and shapes
def test():
    low_resolution = 24  # Input resolution for Generator
    high_resolution = 96  # Output resolution for Generator and input resolution for Discriminator

    with torch.no_grad():  # No gradient computation during testing
        x = torch.randn((1, 3, low_resolution, low_resolution))  # Random low-resolution input
        disc_features = [64, 64, 128, 128, 256, 256, 512, 512]  # Feature map sizes for Discriminator

        # Instantiate and initialize Generator
        gen = Generator(in_channels=3, num_features=64, num_res_block=8, act="prelu")
        gen.apply(weights_init)  # Apply custom weights initialization
        gen_out = gen(x)  # Generate super-resolved image

        # Instantiate and initialize Discriminator
        disc = Discriminator(in_channels=3, features=disc_features, act="leaky_relu")
        disc.apply(weights_init)  # Apply custom weights initialization
        disc_out = disc(gen_out)  # Discriminate between real and fake

        print(gen_out.shape)  # Expected Generator output shape: (1, 3, 96, 96)
        print(disc_out.shape)  # Expected Discriminator output shape: (1, 1)

    # Uncomment to print detailed summaries of both models
    summary(gen, (3, low_resolution, low_resolution))
    summary(disc, (3, high_resolution, high_resolution))


if __name__ == "__main__":
    test()


