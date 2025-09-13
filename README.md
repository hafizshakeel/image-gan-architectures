# Image GAN Architectures

A collection of GAN (Generative Adversarial Network) implementations in PyTorch, focusing on image generation and manipulation tasks.

## Implementations

This repository contains implementations of several key GAN architectures:

- [Basic GAN](Basic%20GAN/): Simple feed-forward GAN implementation
- [DCGAN](DCGAN/): Deep Convolutional GAN with architectural guidelines
- [WGAN](WGAN/): Wasserstein GAN using weight clipping
- [WGAN-GP](WGAN-GP/): Wasserstein GAN with gradient penalty
- [CycleGAN](CycleGAN/): Unpaired image-to-image translation
- [Pix2Pix](Pix2Pix/): Paired image-to-image translation
- [SRGAN](SRGAN/): Super-resolution GAN
- [ESRGAN](ESRGAN/): Enhanced super-resolution GAN
- [ProGAN](ProGAN/): Progressive growing of GANs

Each implementation includes the complete architecture, training code, and documentation.

## Requirements

The implementations use the following main dependencies:
- PyTorch
- torchvision
- numpy
- albumentations
- PIL
- tqdm
- tensorboard

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/hafizshakeel/image-gan-architectures.git
cd image-gan-architectures
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Navigate to the specific GAN implementation you want to use:
```bash
cd [implementation_folder]  # e.g., cd DCGAN
```

4. Each implementation folder contains:
- Model architecture implementation
- Training script
- Configuration file
- README with specific details
- Architecture visualizations (where applicable)

## Repository Structure

Each implementation is self-contained in its directory with consistent organization:
- Model architecture in `*_model.py`
- Training script in `*_train.py`
- Configuration in `config.py`
- Dataset handling in `dataset.py` (where applicable)
- Utility functions in `utils.py`
- Architecture visualizations in PNG format

## License

This project is licensed under the MIT License.

## Contact
  
Email: hafizshakeel1997@gmail.com
