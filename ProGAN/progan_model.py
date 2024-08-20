"""
Progressive Growing of GANs for Improved Quality, Stability, and Variation
(ProGAN) Implementation: https://arxiv.org/abs/1710.10196

This script contains the implementation of the CycleGAN model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

from torchsummary import summary

# factors for output shape(see Architecture.png) where 1 for 512 & 1/2 for 256 and so on
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

"""class for weighted scale conv layer using equalized learning rate (ELR)"""


# ELR Formula:  W_f = W_i * sqrt( 2 / k * k * c ), where k is kernel size and c
# is in_channels. The second part of expression is sort of scale --> He's initialization W^_i = w_i/c
class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):  # gain for
        # initialization constant in ELR formula
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5  # 0.5 for sqrt
        # now we don't want the bias of self.conv layer to be scaled so
        self.bias = self.conv.bias  # sort of copying bias of the current conv layer that's been initialized all zeros
        self.conv.bias = None  # removing the bias

        # Initialize of conv layer
        nn.init.normal_(self.conv.weight)  # in-place normalization
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)  # reshaping the bias term so we
        # can add it to self.conv


"""class for pixel normalization"""


# PixelNorm Formula:
# [ b_{x,y} = a_{x,y} \left(\frac{1}{N} \sum_{j=0}^{N-1} (c_{x+j,y})^2 + \epsilon \right)^{-\frac{1}{2}},
# \text{ where } \epsilon = 10^{-8} ]

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)  # dim=1 for taking mean
        # across channels and keepdim=True, so we can do elementwise division


# To compact code and avoid repetition we'll create a ConvBlock as we've multiple (conv 3 x 3) layers with
# same convolution in both the generator and the discriminator (see Architecture.png)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pn=True):  # use_pn <--> use PixelNorm
        super(ConvBlock, self).__init__()
        self.use_pn = use_pn
        self.conv1 = WSConv2d(in_channels, out_channels)  # here we'll use equalized convolution layers
        self.conv2 = WSConv2d(out_channels, out_channels)  # same here
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


# class for Generator
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels):
        super().__init__()
        self.initial = nn.Sequential(
            # Here to up-sample the first conv layer let's use ConvTranspose2d instead of WSConv2d and see it's effect
            # Start with PixelNorm, ConvTranspose2d and then activation
            # See the actual implementation of this part
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),  # 1x1  --> 4x4 (Conv 4x4 layer)
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels),  # (first Conv 3x3 layer)
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
        # Now add an initial RGB which is needed after each sort of rectangular box (fig2.png) before prog_blocks
        self.initial_rgb = WSConv2d(in_channels, img_channels, 1, 1, 0)  # kernel must be 1x1
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])  # the rgb layer in
        # the beginning and the reason we add this here nn.ModuleList(self.initial_rgb) is we've
        # self.initial in the start so there is going to be one more rgb layer because we don't have
        # this inside of prog blocks --> initial_rgb then prog_blocks, initial_rgb then prog_blocks and so on.

        # Now for prog_blocks go through the factors created in the beginning
        for i in range(len(factors) - 1):  # -1 to prevent index error because of factors[i+1]
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(WSConv2d(conv_out_channels, img_channels, 1, 1, 0))

    # Now creat fade_in layer
    def fade_in(self, alpha, upscaled, generated):
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):  # if steps=0 (4x4 resolution img), if steps=1 (8x8 resolution img), ...
        # double at each of the steps
        out = self.initial(x)  # 4x4

        if steps == 0:
            return self.initial_rgb(out)
        for step in range(steps):  # up-sample, run it through a block, up-sample, run it through a block, ...
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)  # prog_blocks that corresponds to that resolution

        final_upscaled = self.rgb_layers[steps - 1](upscaled)  # The number of channels in upscale will stay the same,
        # while out which has moved through prog_blocks might change. To ensure we can convert both to rgb
        # we use different rgb_layers (steps-1) and steps for upscale and out respectively
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


# class for Discriminator / critic
class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # In discriminator prog-blocks are sort of mirror to the Generator --> factors in reverse order
        # In other words, here we work back ways from factors because the discriminator
        # should be mirrored from the generator. So the first prog_block and
        # rgb layer we append will work for input size 1024x1024, then 512->256-> etc
        for i in range(len(factors) - 1, 0, -1):  # start, end, stop
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i - 1])  # reverse order
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pn=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_channels, 1, 1, 0))

        # Now similar to the initial in the Generator, we've same for final block in the Discriminator
        self.initial_rgb = WSConv2d(img_channels, in_channels, 1, 1, 0)  # this initial_rgb is
        # mirror of the initial_rgb of the generator. In other words, this "initial_rgb"
        # is just the RGB layer for 4x4 input size
        self.rgb_layers.append(self.initial_rgb)  # appending bc prog_blocks can sometimes change number of channels
        self.avg_pool = nn.AvgPool2d(2, 2)  # down sampling using avg pool

        # block for 4x4 input size
        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1, in_channels, 3, 1, 1),  # (last Conv 3x3 layer)
            nn.LeakyReLU(0.2),
            PixelNorm(),
            WSConv2d(in_channels, in_channels, 4, 1, 0),  # (last Conv 4x4 layer)
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, 1, 1, 1, 0)  # use this instead of linear layer
        )

    def fade_in(self, alpha, downscaled, out):
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        # Used to fade in downscaled using avg pooling and output from CNN
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0, unbiased=False).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        # this will give std for example of x ( N x C x H x W --> N) and mean will give a single scalar value
        # which then will be repeated to be the exact same except it's just going to be one channel
        # which is 1 and x.shape[2], x.shape[3] is height and width

        # In other words, we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        #  will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)  # 512 --> 513

    def forward(self, x, alpha, steps):  # if steps=0 (4x4 img), if steps=1 (8x8 img), ... (steps last to the beginning)
        # where we should start in the list of prog_blocks, maybe a bit confusing but
        # the last is for the 4x4 starting from higher 1012 x 1024. So example let's say steps=1, then we should start
        # at the second to last because input_size will be 8x8. If steps==0 we just
        # use the final block
        cur_steps = len(self.prog_blocks) - steps

        # convert from rgb as initial step, this will depend on
        # the image size (each will have it's on rgb layer)
        out = self.leaky(self.rgb_layers[cur_steps](x))

        # to ensure that we don't index incorrectly in the prog_blocks
        if steps == 0:  # i.e, image is 4x4
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # in Discriminator first here we'll add fade_in layer see fig.2
        # Also, because prog_blocks might change the channels, for down scale we use rgb_layer
        # from previous/smaller size which in our case correlates to +1 in the indexing
        downscaled = self.leaky(self.rgb_layers[cur_steps + 1](self.avg_pool(x)))  # 0.5 mean avg_pool --> fig.2
        out = self.avg_pool(self.prog_blocks[cur_steps](out))  # first rgb, prog_blocks, then avg pool

        # now for fade-in since the fade_in is done first between the downscaled and the input
        # this is opposite to the generator
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_steps + 1, len(self.prog_blocks)):  # +1 because we already did current step
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


# test function
if __name__ == "__main__":
    # sys.exit()

    Z_DIM = 100
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, 3)
    critic = Discriminator(IN_CHANNELS, 3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512]:  # upgrade to 512 x 512 or more
        num_steps = int(log2(img_size / 4))
        print(num_steps)
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")


# Print the complete summary of the generator and the discriminator
# class GeneratorWrapper(nn.Module):
#     def __init__(self, gen):
#         super(GeneratorWrapper, self).__init__()
#         self.gen = gen
#
#     def forward(self, x):
#         alpha = 0.5  # Provide a default value for alpha
#         steps = 2  # Provide a default value for steps
#         return self.gen(x, alpha, steps)
#
#
# class DiscriminatorWrapper(nn.Module):
#     def __init__(self, critic):
#         super(DiscriminatorWrapper, self).__init__()
#         self.critic = critic
#
#     def forward(self, x):
#         alpha = 0.5  # Provide a default value for alpha
#         steps = 2  # Provide a default value for steps
#         return self.critic(x, alpha, steps)
#
#
# wrapped_gen = GeneratorWrapper(gen)
# wrapped_critic = DiscriminatorWrapper(critic)
#
# summary(wrapped_gen, (Z_DIM, 1, 1))  # latent vector input.
# summary(wrapped_critic, (3, 512, 512))  # image with 3 channels and the size of HQ-GT is 512 x 512
