import torch
from math import log2

# The starting image size for training, adjust based on the desired resolution. 
# If you want to train fully at higher resolutions, set this to 128 or 256.
START_TRAIN_AT_IMG_SIZE = 16  

# Path to the dataset. Ensure to use the correct subdirectory if using datasets.ImageFolder in get_loader().
DATASET = 'dataset'  

# Saving and loading model checkpoints.
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Flags for saving and loading the model. 
SAVE_MODEL = True
LOAD_MODEL = False

# Learning rate for the optimizer, adjust based on experimentation or dataset size.
LEARNING_RATE = 1e-3

# Batch sizes for different image resolutions, adjust based on available VRAM.
# Example: [32, 32, 32, 16, 16, 16, 16, 8, 4] for higher resolutions.
BATCH_SIZES = [4, 4, 2]  # Customize according to VRAM capacity.

IMAGE_SIZE = 1024  # Target image size to upscale to, usually a high-resolution like 1024x1024.
CHANNELS_IMG = 3  # Number of channels in the input images. Typically 3 for RGB images.

# Latent vector dimension (Z-dimension). In the original paper, it is set to 512.
# Reduced here to 256 to accommodate lower VRAM usage and speed up training.
Z_DIM = 256  

# Number of input channels for the generator and critic. 
# Originally 512 in the paper, reduced to 256 for efficiency.
IN_CHANNELS = 256  

CRITIC_ITERATIONS = 1  # Number of critic iterations per generator iteration.
LAMBDA_GP = 10  # Weight for the gradient penalty in the loss function, commonly set to 10.
NUM_STEPS = int(log2(IMAGE_SIZE / 4)) + 1  # Number of steps for progressively growing the image size.

# Number of epochs at each progressive stage. 
# This can be adjusted based on the desired quality of the output images.
PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)  

# Description from the original paper regarding the training process:
# "We start with 4×4 resolution and train the networks until we have shown the discriminator 
# 800k real images in total. We then alternate between two phases: fade in the first 3-layer block 
# during the next 800k images, stabilize the networks for 800k images, fade in the next 3-layer block 
# during 800k images, etc."

# Fixed noise vector for generating consistent images to monitor progress.
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
