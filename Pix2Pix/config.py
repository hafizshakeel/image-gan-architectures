import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set device to GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths and hyperparameters
TRAIN_DIR = "data/"  # Training data directory
VAL_DIR = "data/"    # Validation data directory
LEARNING_RATE = 2e-4  # Learning rate
BATCH_SIZE = 2        # Batch size for training
NUM_WORKERS = 2       # Data loader worker threads
IMAGE_SIZE = 256      # Image size (height and width)
CHANNELS_IMG = 3      # Number of image channels (RGB)
L1_LAMBDA = 100       # Weight for L1 loss
NUM_EPOCHS = 500      # Number of training epochs
LOAD_MODEL = False    # Flag to load pre-trained model
SAVE_MODEL = True     # Flag to save trained model
CHECKPOINT_DISC = "disc.pth.tar"  # Path to discriminator checkpoint
CHECKPOINT_GEN = "gen.pth.tar"   # Path to generator checkpoint

# Transformations for both input and target images
both_transform = A.Compose(
    [A.Resize(width=256, height=256)], additional_targets={"image0": "image"}
)

# Transformations for input images only
transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.5] * 3, std=[0.5] * 3, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

# Transformations for target (mask) images only
transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5] * 3, std=[0.5] * 3, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)
