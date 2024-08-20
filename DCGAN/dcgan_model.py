"""
DCGAN Implementation: https://arxiv.org/abs/1511.06434

This script contains the implementation of the DCGAN (Deep Convolutional Generative Adversarial Network) model.
It is designed for generating realistic images from random noise using a generator and distinguishing real from
fake images using a discriminator.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

# Import necessary libraries
import sys
import torch
import torch.nn as nn
from torchsummary import summary
import time
import torch.profiler

""" DCGAN Architecture """

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            # features_d = 64 (paper)
            # No BatchNorm to the discriminator input layer
            # See how to compute output of a layer: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),  # 32x32 -->  ((n+2p-f)/2 +1)
            nn.LeakyReLU(0.2),
            self._cblock(features_d, features_d * 2, 4, 2, 1),  # 16x16
            self._cblock(features_d * 2, features_d * 4, 4, 2, 1),  # 8x8
            self._cblock(features_d * 4, features_d * 8, 4, 2, 1),  # 4x4
            # After all _convBlock img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, 4, 2, 0),  # 1x1 (fake or real img)
            nn.Sigmoid(),
        )

    def _cblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channel_noise, channels_img, feature_g):
        super(Generator, self).__init__()
        # Input: N x channels_noise x 1 x 1
        # features_g = 64 (paper) --> 64x16=1024
        # See how we got 4x4 output in the first block.Formula to calculate output for ConvTranspose2d
        # is https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
        # H_out = (H_in−1)*stride[0] − 2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
        # H_out = (1-1) * 2 - 2 * 0 + 1 * (4-1)+ 0 + 1 = 0 + 3 + 1 = 4 (if value not given, default value will be taken)
        # W_out = (Win−1)×stride[1] − 2×padding[1] + dilation[1]×(kernel_size[1]−1) + output_padding[1] + 1
        self.net = nn.Sequential(
            self._cblock(channel_noise, feature_g * 16, 4, 2, 0),  # img: 4x4 after stride=2 for L1 in diag.
            self._cblock(feature_g * 16, feature_g * 8, 4, 2, 1),  # img: 8x8 after S=2 for L2 in diag.
            self._cblock(feature_g * 8, feature_g * 4, 4, 2, 1),   # img: 16x16 after S=2 for L3 in diag.
            self._cblock(feature_g * 4, feature_g * 2, 4, 2, 1),   # img: 32x32 after S=2 for L4 in diag.
            nn.ConvTranspose2d(feature_g * 2, channels_img, 4, 2, 1),  # img: 64x64 after S=2 for L4 in diag.
            # No BatchNorm to the generator output layer
            nn.Tanh()
        )

    def _cblock(self, in_channels, out_channels, kernel_size, stride, padding):
        # use ConvTranspose2d for up-sampling to increases the spatial resolution of an image.
        # Note: ConvTranspose2d also known as a fractionally-strided convolution
        # or simply deconvolution
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    d = torch.randn((N, in_channels, H, W))  # dim for original image
    disc = Discriminator(3, 8).to(device)
    initialize_weights(disc)
    assert disc(d.to(device)).shape == (N, 1, 1, 1), "Discriminator test failed"

    gen = Generator(100, 3, 8).to(device)
    initialize_weights(gen)
    g = torch.randn((N, noise_dim, 1, 1))
    assert gen(g.to(device)).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, test passed!")


# Use time module to compute the start and end times of the forward pass for performance evaluation of the models.
def measure_time(model, input_tensor):
    start_time = time.time()
    output = model(input_tensor)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time


# Get a more detailed breakdown of where the computation time is being spent more within the model.
def profile_model(model, input_tensor):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        model(input_tensor)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # sys.exit()
    test()

    disc = Discriminator(3, 64).to(device)
    gen = Generator(100, 3, 64).to(device)

    input_disc = torch.randn(32, 3, 64, 64).to(device)
    input_gen = torch.randn(32, 100, 1, 1).to(device)

    disc_time = measure_time(disc, input_disc)
    gen_time = measure_time(gen, input_gen)

    print(f"Discriminator forward pass time: {disc_time:.6f} seconds")
    print(f"Generator forward pass time: {gen_time:.6f} seconds")

    # profile_model(disc, input_disc)
    # profile_model(gen, input_gen)

    summary(disc, (3, 64, 64))
    summary(gen, (100, 1, 1))

