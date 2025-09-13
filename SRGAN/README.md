# SRGAN Implementation

Implementation of SRGAN (Super-Resolution GAN) based on the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## Overview

SRGAN is designed for single image super-resolution, capable of inferring photo-realistic natural images with 4x upscaling factors.

## Components

- `srgan_model.py`: Generator and Discriminator architectures
- `srgan_train.py`: Training implementation
- `config.py`: Configuration and hyperparameters
- `dataset.py`: Dataset handling
- `utils.py`: Training utilities
- `VGGLoss.py`: Perceptual loss implementation
- `architecture.png`: Network architecture diagram

## Implementation Details

- Residual blocks in generator architecture
- VGG-based perceptual loss
- Discriminator with VGG-style architecture
- Image augmentation with albumentations

## Model Features

Generator:
- Residual architecture with skip connection
- Upsampling using convolutional layers
- Batch normalization and PReLU activation

Discriminator:
- VGG-style architecture
- LeakyReLU activation
- Binary classification output
- No batch normalization in first layer