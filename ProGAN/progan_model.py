"""
Progressive Growing of GANs for Improved Quality, Stability, and Variation
(ProGAN) Implementation: https://arxiv.org/abs/1710.10196

This script contains the implementation of the CycleGAN model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
import sys

# Factors for output shape based on architecture (see Architecture.png).
# 1 for 512x512, 1/2 for 256x256, and so on.
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

"""Class for weighted scale convolution layer using equalized learning rate (ELR)"""

# ELR Formula: W_f = W_i * sqrt( 2 / k * k * c ), where k is kernel size and c is in_channels.
# The second part of the expression is a scale (He's initialization W^_i = w_i/c)
class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # Calculating the scale based on the ELR formula
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5  # 0.5 for sqrt

        # Store bias from conv layer; we will manually adjust it later
        self.bias = self.conv.bias
        self.conv.bias = None  # Remove bias from convolution

        # Initialize the convolution weights and set the bias to zero
        nn.init.normal_(self.conv.weight)  # Normal initialization
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Apply scaling to the input and then add the bias after reshaping it for broadcasting
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


"""Class for Pixel Normalization"""

# PixelNorm Formula:
# b_{x,y} = a_{x,y} * ( 1/N * sum_j=0^(N-1) (c_{x+j,y})^2 + epsilon )^(-1/2),
# where epsilon = 1e-8.
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8  # Small constant to avoid division by zero

    def forward(self, x):
        # Normalize the input across channels and add epsilon for numerical stability
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


# To compact the code and avoid repetition, create a ConvBlock
# as we have multiple 3x3 conv layers in both generator and discriminator.
# (See Architecture.png for clarification)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pn=True):  # use_pn: use PixelNorm
        super(ConvBlock, self).__init__()
        self.use_pn = use_pn

        # Using equalized convolution layers for both conv1 and conv2
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)

        self.leaky = nn.LeakyReLU(0.2)  # LeakyReLU for activation
        self.pn = PixelNorm()  # PixelNorm layer (only if use_pn=True)

    def forward(self, x):
        # First convolution followed by LeakyReLU and optional PixelNorm
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x

        # Second convolution followed by LeakyReLU and optional PixelNorm
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x

        return x


"""Class for the Generator Network"""

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels):
        super().__init__()

        # Initial layer consisting of PixelNorm, ConvTranspose2d, WSConv2d, and LeakyReLU
        self.initial = nn.Sequential(
            PixelNorm(),  # Pixel normalization
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),  # Upsample from 1x1 to 4x4 (ConvTranspose)
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels),  # First convolution layer (3x3)
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        # Add an initial RGB layer (1x1 convolution) before progressive blocks
        self.initial_rgb = WSConv2d(in_channels, img_channels, 1, 1, 0)  # 1x1 convolution for RGB

        # List for progressive blocks and RGB layers
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])

        # Add progressive blocks and RGB layers for different image sizes
        for i in range(len(factors) - 1):  # -1 to prevent out-of-bounds error for factors[i+1]
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))  # Add ConvBlock
            self.rgb_layers.append(WSConv2d(conv_out_channels, img_channels, 1, 1, 0))  # Add RGB layer

    def fade_in(self, alpha, upscaled, generated):
        # Fade-in function for smoothing the transition between generated images and upscaled images
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        # Forward pass through the generator, with progressive upsampling based on the number of steps
        out = self.initial(x)  # Start with 4x4 resolution

        if steps == 0:
            return self.initial_rgb(out)  # Return RGB layer for 4x4 resolution

        # For higher resolutions, upsample and pass through progressive blocks
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)  # Get RGB layer after last progressive block
        final_out = self.rgb_layers[steps](out)  # Get RGB output after the final block

        # Perform fade-in between upscaled and generated image
        return self.fade_in(alpha, final_upscaled, final_out)

"""Class for the Discriminator (Critic) Network"""

class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels):
        super().__init__()

        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])  # Initialize layers
        self.leaky = nn.LeakyReLU(0.2)

        # In the discriminator, we mirror the generator's progressive blocks in reverse
        for i in range(len(factors) - 1, 0, -1):  # Reverse order for the discriminator
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pn=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_channels, 1, 1, 0))

        # Initial RGB layer (for the 4x4 input size)
        self.initial_rgb = WSConv2d(img_channels, in_channels, 1, 1, 0)
        self.rgb_layers.append(self.initial_rgb)  # Append RGB layer to the list
        self.avg_pool = nn.AvgPool2d(2, 2)  # Average pooling layer for downsampling

        # Final block for 4x4 input size
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, 3, 1, 1),  # Last Conv (3x3)
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, 4, 1, 0),  # Last Conv (4x4)
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, 1, 1, 1, 0)  # Output layer (1x1 Conv)
        )

    def fade_in(self, alpha, downscaled, out):
        # Fade-in function for the discriminator (similar to generator)
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        # Calculate the standard deviation across the batch for each feature map
        batch_statistics = torch.std(x, dim=0, unbiased=False).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)  # Concatenate the batch statistics to the input

    def forward(self, x, alpha, steps):
        # Forward pass through the discriminator
        cur_steps = len(self.prog_blocks) - steps  # Start from the corresponding step

        # Convert input from RGB using the appropriate RGB layer
        out = self.leaky(self.rgb_layers[cur_steps](x))

        if steps == 0:
            out = self.minibatch_std(out)  # Calculate minibatch std for 4x4 input
            return self.final_block(out).view(out.shape[0], -1)

        # Add fade-in for downscaled image and pass through progressive blocks
        downscaled = self.leaky(self.rgb_layers[cur_steps + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_steps](out))  # Process through current block

        out = self.fade_in(alpha, downscaled, out)

        # Continue through the remaining progressive blocks
        for step in range(cur_steps + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)  # Add minibatch standard deviation
        return self.final_block(out).view(out.shape[0], -1)


# Test function for verifying the Generator and Discriminator
if __name__ == "__main__":
    # sys.exit()
    Z_DIM = 100
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, 3)
    critic = Discriminator(IN_CHANNELS, 3)

    # Test the model with various image sizes
    for img_size in [4, 8, 16, 32, 64, 128, 256, 512]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")


