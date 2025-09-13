# ESRGAN Implementation

Implementation of Enhanced SRGAN (ESRGAN) based on the paper [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219).

## Overview

ESRGAN improves upon SRGAN for single image super-resolution by introducing:
- Residual-in-Residual Dense Block (RRDB)
- Enhanced perceptual loss using VGG features
- Improved GAN architecture for better stability

## Components

- `esrgan_model.py`: Generator with RRDB and Discriminator architectures
- `esrgan_train.py`: Training implementation
- `config.py`: Configuration and hyperparameters
- `dataset.py`: Dataset handling for super-resolution
- `utils.py`: Training utilities and helper functions
- `VGGLoss.py`: Perceptual loss implementation
- Architecture diagrams in `architecture_details/`

## Implementation Details

- Upscaling factor: 4x
- RRDB network structure with dense connections
- Mixed precision training support
- Image augmentation using albumentations

## Model Architecture

Generator features:
- Initial feature extraction
- Multiple RRDB blocks
- Upsampling using nearest-neighbor interpolation
- Final reconstruction

Discriminator:
- VGG-style architecture
- LeakyReLU activation
- No batch normalization