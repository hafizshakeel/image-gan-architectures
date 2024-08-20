import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import sys


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :256, :]  # first half of img for train & val (input)
        target_image = image[:, 256:, :]  # second half of img for train & val (target)

        augmentations = config.both_transform(image=input_image, image0=target_image)  # apply transformation on
        # both input and target image
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        # Some transformations are only applicable for input img or for target label.
        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    # sys.exit()
    dataset = MapDataset("data/dehaze/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
