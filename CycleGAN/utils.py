import numpy as np
import os
import random
import torch

from torchvision.utils import save_image
from tqdm import tqdm

import config

# Training function for generators and discriminators
def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)
    for idx, (z, h) in enumerate(loop):  # z for zebra, h for horse
        z, h = z.to(config.DEVICE), h.to(config.DEVICE)  # Move to device

        # Train Discriminators H & Z
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            h_fake = gen_H(z)  # Generate fake horse
            disc_h_real_loss = mse(disc_H(h), torch.ones_like(disc_H(h)))  # Real horse loss
            disc_h_fake_loss = mse(disc_H(h_fake.detach()), torch.zeros_like(disc_H(h_fake)))  # Fake horse loss
            disc_h_loss = disc_h_real_loss + disc_h_fake_loss

            z_fake = gen_H(h)  # Generate fake zebra
            disc_z_real_loss = mse(disc_H(z), torch.ones_like(disc_H(z)))  # Real zebra loss
            disc_z_fake_loss = mse(disc_H(z_fake.detach()), torch.zeros_like(disc_H(z_fake)))  # Fake zebra loss
            disc_z_loss = disc_z_real_loss + disc_z_fake_loss

            D_loss = (disc_h_loss + disc_z_loss) / 2  # Total discriminator loss

            # Backpropagate and update discriminator
            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

        # Train Generators H & Z
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            gen_z_fake_loss = mse(disc_Z(gen_H(h)), torch.ones_like(disc_Z(gen_H(h))))  # Generator adversarial loss
            gen_h_fake_loss = mse(disc_H(gen_H(z)), torch.ones_like(disc_H(gen_H(z))))  # Generator adversarial loss

            cycle_zebra = gen_Z(gen_H(z))  # Cycle loss (zebra)
            cycle_horse = gen_H(gen_Z(h))  # Cycle loss (horse)
            cycle_z_loss = L1(z, cycle_zebra)
            cycle_h_loss = L1(h, cycle_horse)

            # Total generator loss
            G_loss = gen_z_fake_loss + gen_h_fake_loss + cycle_z_loss * config.LAMBDA_CYCLE + cycle_h_loss * config.LAMBDA_CYCLE

            # Backpropagate and update generator
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

        # Save generated images periodically
        if idx % 2 == 0:
            save_image(h_fake * 0.5 + 0.5, f"saved_images/horse_{idx}.png")  # Save horse image
            save_image(z_fake * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")  # Save zebra image

            # Update progress bar with discriminator performance
            loop.set_postfix(
                disc_h_real=torch.sigmoid(disc_H(h)).mean().item(),
                disc_z_real=torch.sigmoid(disc_H(z)).mean().item(),
                disc_h_fake=torch.sigmoid(disc_H(h_fake)).mean().item(),
                disc_z_fake=torch.sigmoid(disc_H(z_fake)).mean().item(),
            )


# Save model checkpoint
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),  # Save model weights
        "optimizer": optimizer.state_dict(),  # Save optimizer state
    }
    torch.save(checkpoint, filename)  # Save checkpoint


# Load model checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Adjust learning rate after loading checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Save example images generated during validation
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))  # Get a batch of validation images
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()  # Set to evaluation mode
    with torch.no_grad():
        y_fake = gen(x)  # Generate fake images
        y_fake = y_fake * 0.5 + 0.5  # Undo normalization
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")  # Save label image for epoch 1
    gen.train()  # Switch back to training mode


# Set random seed for reproducibility
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
