import numpy as np
import os
import random
import torch

from torchvision.utils import save_image
from tqdm import tqdm

import config


def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)
    for idx, (z, h) in enumerate(loop):  # z for zebra, h for horse
        z = z.to(config.DEVICE)
        h = h.to(config.DEVICE)

        # Train Discriminators H & Z
        with torch.cuda.amp.autocast():
            h_fake = gen_H(z)
            disc_h_real = disc_H(h)
            disc_h_real_loss = mse(disc_h_real, torch.ones_like(disc_h_real))
            disc_h_fake = disc_H(h_fake.detach())
            disc_h_fake_loss = mse(disc_h_fake, torch.zeros_like(disc_h_fake))
            disc_h_loss = disc_h_real_loss + disc_h_fake_loss

            z_fake = gen_H(h)
            disc_z_real = disc_H(z)
            disc_z_real_loss = mse(disc_z_real, torch.ones_like(disc_z_real))
            disc_z_fake = disc_H(z_fake.detach())
            disc_z_fake_loss = mse(disc_z_fake, torch.zeros_like(disc_z_fake))
            disc_z_loss = disc_z_real_loss + disc_z_fake_loss

            # put it together
            D_loss = (disc_h_loss + disc_z_loss) / 2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

        # Train Generators H & Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            gen_z_fake = disc_Z(z_fake)
            gen_z_fake_loss = mse(gen_z_fake, torch.ones_like(gen_z_fake))
            gen_h_fake = disc_H(h_fake)
            gen_h_fake_loss = mse(gen_h_fake, torch.ones_like(gen_h_fake))

            # cycle loss
            cycle_zebra = gen_Z(h_fake)
            cycle_horse = gen_H(z_fake)
            cycle_z_loss = L1(z, cycle_zebra)
            cycle_h_loss = L1(h, cycle_horse)

            # Identity loss -  shouldn't affect the real image
            # identity loss (better to remove these for efficiency if you set lambda_identity=0)
            # identity_z = gen_Z(z)
            # identity_h = gen_H(h)
            # identity_z_loss = L1(z, identity_z)
            # identity_h_loss = L1(h, identity_h)

            # add all together
            # G_loss = (gen_z_fake_loss + gen_h_fake_loss + cycle_z_loss * config.LAMBDA_IDENTITY +
            #           cycle_h_loss * config.LAMBDA_IDENTITY + identity_z_loss * config.LAMBDA_IDENTITY
            #           + identity_h_loss * config.LAMBDA_IDENTITY)

            G_loss = (gen_z_fake_loss + gen_h_fake_loss + cycle_z_loss * config.LAMBDA_CYCLE +
                      cycle_h_loss * config.LAMBDA_CYCLE)

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

        if idx % 2 == 0:
            save_image(h_fake * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(z_fake * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

            loop.set_postfix(
                disc_h_real=torch.sigmoid(disc_h_real).mean().item(),
                disc_z_real=torch.sigmoid(disc_z_real).mean().item(),
                disc_h_fake=torch.sigmoid(disc_h_fake).mean().item(),
                disc_z_fake=torch.sigmoid(disc_z_fake).mean().item(),
            )


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
