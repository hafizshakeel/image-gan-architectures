import torch
import torch.nn as nn
from p2p_model import Discriminator, Generator
import config
import torch.optim as optim
from utils import *
from dataset import MapDataset
from torch.utils.data import DataLoader


def main():
    # Models and Optimizers
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # Loss functions: BCE + L1. Not WGAN-GP loss bc it didn't work well with PatchGAN
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # Load Model
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, lr=config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, lr=config.LEARNING_RATE)

    # Load Train and Validation Dataset
    train_dataset = MapDataset(root_dir="data/")
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataset = MapDataset(root_dir="data/")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # float16 training
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    # Training Loop
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, BCE, L1_LOSS, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()
