import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/"
VAL_DIR = "data/"
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "gen_h.pth.tar"
CHECKPOINT_GEN_Z = "gen_z.pth.tar"
CHECKPOINT_DISC_H = "critic_h.pth.tar"
CHECKPOINT_DISC_Z = "critic_z.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),  # [-1, 1]
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    is_check_shapes=False,  # Disable the shape check
)
