import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

"""Train using WGAN-GP discriminator instead of using relativistic discriminator"""

# Function to train the generator and discriminator
def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    """
    Args:
        loader: DataLoader providing batches of low-res and high-res images.
        disc: Discriminator model.
        gen: Generator model.
        opt_gen: Optimizer for the generator.
        opt_disc: Optimizer for the discriminator.
        mse: Mean Squared Error loss function.
        bce: Binary Cross Entropy loss function.
        vgg_loss: VGG perceptual loss function.
    """
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # Generate fake high-res images from the low-res input
        fake = gen(low_res)
        
        # Get discriminator outputs for real and fake images
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())  # detach to avoid affecting generator gradients
        
        # Calculate BCE loss for real and fake discriminator outputs
        # Label smoothing for real labels to help stabilize training
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        # Update discriminator weights
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # Here, you can train it first for only mse then for vgg & adversarial to see similar results (paper)
        disc_fake = disc(fake)
        # l2_loss = mse(fake, high_res)  # MSE loss for pixel-wise similarity (optional)
        
        # Adversarial loss scaled by a small factor to balance with VGG loss
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        # VGG loss to capture perceptual differences
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        
        # Combine VGG and adversarial loss for generator training
        gen_loss = loss_for_vgg + adversarial_loss

        # Update generator weights
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        # Save examples every 100 iterations for monitoring progress
        if idx % 100 == 0:
            plot_examples("test_images/", gen)


# Function to calculate gradient penalty for WGAN-GP
def gradient_penalty(critic, real, fake, device):
    """
    Compute the gradient penalty for the Wasserstein GAN with Gradient Penalty (WGAN-GP).
    """
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores for the interpolated images
    mixed_scores = critic(interpolated_images)

    # Calculate gradients of the scores with respect to the interpolated images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute the L2 norm of the gradients
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)

    # Compute the gradient penalty as the deviation from norm 1
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


# Function to save model and optimizer state
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Save model and optimizer state to a checkpoint file.
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


# Function to load model and optimizer state from a checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Load model and optimizer state from a checkpoint file and update learning rate.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Function to plot and save example images
def plot_examples(low_res_folder, gen):
    """
    Plot and save examples of the generator's output on test images.
    """
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open("test_images/" + file)
        with torch.no_grad():
            upscaled_img = gen(
                config.test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        save_image(upscaled_img * 0.5 + 0.5, f"saved/{file}")
    gen.train()
