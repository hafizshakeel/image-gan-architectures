# WGAN Implementation

Implementation of Wasserstein GAN based on the paper [Wasserstein GAN](https://arxiv.org/abs/1701.07875).

## Overview

WGAN improves the stability of GAN training by using the Wasserstein distance instead of the Jensen-Shannon divergence.

## Components

- `dcgan_to_wgan.py`: WGAN architecture based on DCGAN structure
- `dcgan_to_wgan_train.py`: Training implementation with Wasserstein loss
- `Algo1 WGAN.png`: Algorithm visualization from the paper

## Implementation Details

- Weight clipping for Lipschitz constraint
- No sigmoid in the discriminator/critic
- Using RMSprop optimizer as suggested in the paper
- Modified loss function:
  - Critic maximizes Wasserstein distance
  - Generator minimizes Wasserstein distance
- Critic trained multiple times per generator update

## Architecture

Built upon DCGAN architecture with key modifications:
- Removed sigmoid from discriminator
- Implemented weight clipping
- Modified loss computation
- Changed optimizer settings