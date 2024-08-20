import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 16  # change to 128 or 256 if you want complete training
DATASET = 'dataset'  # don't mention subdir if using datasets.ImageFolder in get_loader()
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
# BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]  # change depending on your vram
BATCH_SIZES = [4, 4, 2]  # change depending on your vram
IMAGE_SIZE = 1024
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper. I set 256 for less vram usage/speed up training
IN_CHANNELS = 256  # should be 512 in original paper. I set 256 for less vram usage/speed up training
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE / 4)) + 1
PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)  # [number] can be changes based on quality you output image you want.
# From paper,
# " We start with 4 × 4 resolution and train the networks until we
# have shown the discriminator 800k real images in total. We then alternate between two phases: fade
# in the first 3-layer block during the next 800k images, stabilize the networks for 800k images, fade
# in the next 3-layer block during 800k images, etc."
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4
