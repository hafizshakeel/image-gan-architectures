# CycleGAN Implementation

Implementation of CycleGAN based on the paper [Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).

## Overview

CycleGAN enables image-to-image translation without paired training data by introducing cycle consistency loss. This implementation includes:
- Bidirectional generation between two domains
- Cycle consistency loss to maintain content
- Identity loss for preserving color
- Patch-based discriminator architecture

## Components

- `cycgan_model.py`: Generator and Discriminator architectures
- `cycgan_train.py`: Training loop implementation
- `cyc_dataset.py`: Dataset class for unpaired image translation
- `config.py`: Configuration and hyperparameters
- `utils.py`: Utility functions for training and checkpointing
- Architecture diagrams:
  - `cycgan_arch_summary.png`
  - `cycgan_architectures.png`

## Implementation Details

- Uses instance normalization instead of batch normalization
- Implements residual blocks in the generator
- PatchGAN discriminator for realistic texture matching
- Flexible dataset class that works with any unpaired image domains


## Dataset Structure

The implementation expects data in the following format:
- Training images in `data/` directory
- Separate subdirectories for each domain (e.g., horses and zebras)