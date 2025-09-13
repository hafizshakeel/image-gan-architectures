# Basic GAN Implementation

This folder contains a basic implementation of a Generative Adversarial Network (GAN) using feed-forward neural networks.

## Overview

The implementation consists of:
- A simple Generator that transforms random noise into fake images
- A Discriminator that tries to distinguish between real and fake images
- Basic training loop with BCE loss
- Visualization using TensorBoard

## Components

- `basic_gan.py`: Contains the Generator and Discriminator model architectures
- `train.py`: Training script with hyperparameters and training loop

## Architecture Details

- **Generator**: Feed-forward network with LeakyReLU activation and Tanh output
- **Discriminator**: Feed-forward network designed for binary classification
- Input dimension: 28x28x1 (designed for MNIST-like datasets)
- Latent dimension: 64

## Training

The models are trained using:
- Adam optimizer with learning rate 3e-4
- Binary Cross Entropy loss
- Batch size of 32
- Progress monitoring through TensorBoard

This implementation serves as a foundation for understanding more complex GAN architectures.