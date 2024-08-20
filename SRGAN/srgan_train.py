import torch
import config
from torch import nn
from torch import optim
from utils import *
from VGGLoss import VGGLoss
from torch.utils.data import DataLoader
from srgan_model import Generator, Discriminator
from dataset import ImageFolder

torch.backends.cudnn.benchmark = True


def main():
    dataset = ImageFolder(root_dir="new_data/")
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True,
                        num_workers=config.NUM_WORKERS)
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(img_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))  # change betas for wgan-gp loss
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)  # utils

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()
