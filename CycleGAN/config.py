import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Device configuration: Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory paths for training and validation datasets
TRAIN_DIR = "data/"
VAL_DIR = "data/"

# Hyperparameters for training
BATCH_SIZE = 1  # Number of images per batch
LEARNING_RATE = 2e-5  # Learning rate for the optimizer
LAMBDA_CYCLE = 10  # Weight for cycle consistency loss
NUM_WORKERS = 4  # Number of subprocesses for data loading
NUM_EPOCHS = 10  # Number of epochs for training

# Flags to load and save model checkpoints
LOAD_MODEL = False 
SAVE_MODEL = True  

# Checkpoint file names for saving and loading models
CHECKPOINT_GEN_H = "gen_h.pth.tar"
CHECKPOINT_GEN_Z = "gen_z.pth.tar" 
CHECKPOINT_DISC_H = "critic_h.pth.tar"
CHECKPOINT_DISC_Z = "critic_z.pth.tar"

# Transformation pipeline for data preprocessing
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),  # Resize images
        A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),  # Normalize to [-1, 1]
        ToTensorV2(),  # Convert images to PyTorch tensors
    ],
    additional_targets={"image0": "image"},  # Define additional targets for transformation
    is_check_shapes=False,  # Disable shape check to allow flexible input sizes
)
