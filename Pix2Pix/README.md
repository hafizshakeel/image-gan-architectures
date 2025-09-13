# Pix2Pix Implementation

Implementation of Pix2Pix based on the paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004).

## Overview

Pix2Pix is a conditional GAN framework for image-to-image translation that learns a mapping between input and output images using paired training data.

## Components

- `p2p_model.py`: Generator and Discriminator architectures
- `p2p_train.py`: Training implementation
- `config.py`: Configuration and hyperparameters
- `dataset.py`: Dataset handling for paired images
- `utils.py`: Training utilities and helper functions
- Architecture visualizations:
  - `padding_mode.png`
  - `pix2pix implementation.png`

## Implementation Details

- U-Net generator architecture with skip connections
- PatchGAN discriminator
- Paired image training with L1 loss
- Mixed precision training support

## Architecture

Generator:
- Encoder-decoder architecture with skip connections
- Instance normalization
- ReLU/LeakyReLU activations

Discriminator:
- PatchGAN architecture for local texture matching
- 70x70 receptive field
- No batch normalization in first layer