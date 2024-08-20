import torch
import config
from torch import nn
from torch import optim
from utils import *
from VGGLoss import VGGLoss
from torch.utils.data import DataLoader
from esrgan_model import Generator, Discriminator, initialize_weights
from dataset import ImageFolder
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True


def main():
    dataset = ImageFolder(root_dir="data/")
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True,
                        num_workers=config.NUM_WORKERS)
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    initialize_weights(gen)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    writer = SummaryWriter("logs")
    tb_step = 0
    l1 = nn.L1Loss()
    gen.train()
    disc.train()
    vgg_loss = VGGLoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)


    for epoch in range(config.NUM_EPOCHS):
        tb_step = train_fn(loader, disc, gen, opt_gen, opt_disc, l1, vgg_loss, g_scaler, d_scaler, writer, tb_step)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    try_model = True

    if try_model:
        # Will just use pretrained weights and run on images
        # in test_images/ and save the ones to SR in saved/
        gen = Generator(in_channels=3).to(config.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        plot_examples("test_images/", gen)
    else:
        # This will train from scratch
        main()
