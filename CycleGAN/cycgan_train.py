import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from cyc_dataset import HorseZebraDataset
from torch.utils.data import DataLoader
from cycgan_model import Discriminator, Generator
import config

# import logging
# logging.getLogger('albumentations.check_version').setLevel(logging.ERROR)


def main():
    # Models and Optimizers
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)  # disc_H for classifying real vs fake images of horses
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)  # disc_Z for classifying real vs fake images of zebras
    gen_Z = Generator(img_channels=3, num_features=64, num_residuals=9).to(config.DEVICE)  # gen horse --> zebra
    gen_H = Generator(img_channels=3, num_features=64, num_residuals=9).to(config.DEVICE)  # gen zebra --> horse
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )

    # Loss functions
    L1 = nn.L1Loss()  # for the cycle consistency loss and also the identity loss
    mse = nn.MSELoss()  # for adversarial loss

    # Load Model
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC_H, disc_H, opt_disc, lr=config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_Z, disc_Z, opt_disc, lr=config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, lr=config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, lr=config.LEARNING_RATE)

    # Load Train and Validation Dataset
    train_dataset = HorseZebraDataset(root_zebra=config.TRAIN_DIR + "/zebras", root_horse=config.TRAIN_DIR + "/horses", transform=config.transforms)
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    val_dataset = HorseZebraDataset(root_zebra=config.VAL_DIR + "/zebras", root_horse=config.VAL_DIR + "/horses", transform=config.transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # float16 training
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    # Training Loop
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_Z, gen_H, train_loader, opt_disc, opt_gen, L1, mse, g_scaler, d_scaler)  # utils

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_DISC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_DISC_Z)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)

        # save_some_examples(gen_Z, val_loader, epoch, folder="GenZebras")
        # save_some_examples(gen_H, val_loader, epoch, folder="GenHorses")


if __name__ == "__main__":
    main()
