from torch import nn

"""
Simplest / Basic GANs using feed-forward networks

This script contains the implementation of a simple GAN (Generative Adversarial Network).
It includes both the Discriminator and Generator models using basic feed-forward neural networks.
The models are designed for generating and discriminating simple images, such as those from the MNIST dataset.

Author: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

class Discriminator(nn.Module):
    def __init__(self, in_features):
        """
        Initializes the Discriminator model.

        Args:
            in_features (int): Number of input features, typically the flattened size of the input image (e.g., 784 for MNIST).
        """
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),  # in_features --> 784 pixels for MNIST Dataset
            nn.LeakyReLU(0.01),  # LeakyReLU activation with a small negative slope (0.01) to avoid dead neurons
            nn.Linear(128, 1),  # output a single value, fake or real
            nn.Sigmoid(),  # output between 0-1
        )

    def forward(self, x):
        return self.disc(x)  # Pass the input through the Discriminator network


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):  # z_dim --> noise, latent noise
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),  # img_dim: 28x28x1 = 784
            nn.Tanh(),  # since we want to normalize inputs to [-1, 1], use Tanh() to make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)  # Pass the input noise through the Generator network
