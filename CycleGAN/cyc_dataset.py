import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import config


class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        super(HorseZebraDataset, self).__init__()
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(rself.root_zebra)
        self.horse_images = os.listdir(self.root_horse)

        self.dataset_length = max(len(self.zebra_images), len(self.horse_images))  # since length of datasets
        # is not equal; data is not in pair
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.zebra_len]  # since index could be greater than the dataset we've
        # because we're taking max of the two, so it can be solved using % to get correct range without index error
        horse_img = self.horse_images[index % self.horse_len]
        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        # convert to numpy array for albumentations
        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)  # check config file
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img, horse_img


if __name__ == "__main__":
    sys.exit()
    dataset = HorseZebraDataset(root_zebra="data/zebra/", root_horse="data/horse/", transform=config.transforms)
    loader = DataLoader(dataset, batch_size=1)
    for idx, (x, y) in enumerate(loader):
        print(f"Original shapes: x={x.shape}, y={y.shape}")

        # Reverse normalization: x and y are in range [-1, 1] due to normalization
        x = x * 0.5 + 0.5  # shift range to [0, 1]
        y = y * 0.5 + 0.5  # shift range to [0, 1]

        print(f"Converted types: x={x.dtype}, y={y.dtype}")

        # Ensure the tensors are of type float32 before saving
        # save_image(x, "zebra.jpg")
        # save_image(y, "horse.jpg")

