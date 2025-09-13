# DCGAN (Deep Convolutional GAN)

Implementation of Deep Convolutional GAN based on the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).

## Overview

DCGAN introduces architectural guidelines for stable Deep Convolutional GANs:
- Using strided convolutions and transposed convolutions for downsampling/upsampling
- Batch Normalization in both networks
- Removing fully connected layers
- Using LeakyReLU activation in discriminator and ReLU in generator

## Components

- `dcgan_model.py`: Contains Generator and Discriminator architectures with proper initialization
- `dcgan_train.py`: Training implementation with hyperparameters and optimization
- `DCGAN.png`: Architecture visualization

## Model Architecture

Both Generator and Discriminator use:
- Convolutional layers with stride-2 for scaling
- Batch Normalization layers
- Appropriate activation functions (LeakyReLU/ReLU)