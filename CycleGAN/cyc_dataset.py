import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import config 

# The HorseZebraDataset class can be used for any two sets of images (e.g., summer <--> winter, Monet painting <--> photo) 
# by adjusting the directory paths in the config file. There is no need to define a separate dataset for every type of image pair. 
# In the future, I plan to add a more generalized version of this dataset class.

class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        super(HorseZebraDataset, self).__init__()
        self.root_zebra = root_zebra  # Directory containing zebra images
        self.root_horse = root_horse  # Directory containing horse images
        self.transform = transform  # Transformation pipeline to apply to images

        # List all image filenames in the respective directories
        self.zebra_images = os.listdir(self.root_zebra)
        self.horse_images = os.listdir(self.root_horse)

        # Use the maximum length of the two datasets since the lengths might not be equal
        self.dataset_length = max(len(self.zebra_images), len(self.horse_images))
        self.zebra_len = len(self.zebra_images)  # Number of zebra images
        self.horse_len = len(self.horse_images)  # Number of horse images

    def __len__(self):
        # Return the total length of the dataset
        return self.dataset_length

    def __getitem__(self, index):
        # Get zebra and horse images using modular indexing to avoid index errors
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]
        # Full path to zebra and horse images.
        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        # Open images, convert to RGB, and transform to numpy arrays for Albumentations
        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            # Apply transformations to both images
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img, horse_img


if __name__ == "__main__":
    sys.exit()
    # Instantiate the dataset and data loader
    dataset = HorseZebraDataset(root_zebra="data/zebra/", root_horse="data/horse/", transform=config.transforms)
    loader = DataLoader(dataset, batch_size=1)

    for idx, (x, y) in enumerate(loader):
        print(f"Original shapes: x={x.shape}, y={y.shape}")

        # Reverse normalization: Convert images from range [-1, 1] to [0, 1]
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5

        # Print the data types of the converted tensors
        print(f"Converted types: x={x.dtype}, y={y.dtype}")

        # Save the images if needed, ensuring they are in float32 format
        # save_image(x, "zebra.jpg")
        # save_image(y, "horse.jpg")
