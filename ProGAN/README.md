# ProGAN Implementation

Implementation of Progressive Growing of GANs based on the paper [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).

## Overview

ProGAN introduces progressive growing of both generator and discriminator, starting from low resolution and incrementally adding layers to reach higher resolutions.

## Components

- `progan_model.py`: Progressive growing Generator and Discriminator architectures
- `progan_train.py`: Training loop with progressive resolution scaling
- `config.py`: Configuration and training parameters
- `utils.py`: Helper functions and training utilities
- Architecture visualizations in `architecture_details/`

## Implementation Details

Key features implemented:
- Progressive growing from 4x4 to target resolution
- Equalized learning rate (Weight scaling)
- Pixel normalization
- Minibatch standard deviation
- Smooth transition between resolutions using alpha
- WGAN-GP loss function

## Architecture

Generator:
- Initial 4x4 resolution block
- Progressive upsampling blocks
- Weight scaling convolutions
- Pixel normalization
- Smooth fade-in transitions

Discriminator:
- Symmetric downsampling architecture
- Minibatch discrimination layer
- Progressive resolution handling
- No batch normalization

## Training

- Uses truncation trick for generating samples
- Dynamic batch size based on resolution
- Gradient penalty for WGAN loss
- Progressively transitions through resolutions