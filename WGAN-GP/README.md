# WGAN-GP Implementation

Implementation of Wasserstein GAN with Gradient Penalty based on the paper [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).

## Overview

WGAN-GP improves upon WGAN by replacing the weight clipping with a gradient penalty to enforce the Lipschitz constraint.

## Components

- `wgan_to_wgangp.py`: WGAN-GP model architectures
- `wgan_to_wgangp_train.py`: Training implementation
- `gradient_penalty.py`: Gradient penalty computation
- `Algo-WGAN-GP.png`: Algorithm visualization

## Implementation Details

- Gradient penalty instead of weight clipping
- Adam optimizer (β1=0, β2=0.9)
- No batch normalization in critic
- One-sided gradient penalty with λ=10
- Performance profiling capabilities

## Key Features

- Improved stability over original WGAN
- No mode collapse during training
- Better gradient flow
- Support for various architectures
- Performance timing and profiling tools