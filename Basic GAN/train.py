import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from basic_gan import Discriminator, Generator

# Hyperparameters
# NOTE: GANs are very sensitive to hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4  # Learning rate for Adam optimizer
z_dim = 64  # Dimension of the noise vector (can be 128, 256, or smaller)
img_dim = 28 * 28 * 1  # Flattened size of the image (28x28 pixels)
batch_size = 32
num_epochs = 20

# Initialize networks
disc = Discriminator(in_features=img_dim).to(device)  # Discriminator model
gen = Generator(z_dim=z_dim, img_dim=img_dim).to(device)  # Generator model

# Initialize noise
fixed_noise = torch.randn((batch_size, z_dim)).to(device)  # Fixed noise for visualizing progress over epochs

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]  # Normalize images to [-1, 1]
)

# Dataset
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # DataLoader for batching and shuffling

# Optimizers
opt_disc = optim.Adam(disc.parameters(), lr=lr)  # Optimizer for Discriminator
opt_gen = optim.Adam(gen.parameters(), lr=lr)  # Optimizer for Generator

# Loss function
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

# TensorBoard
writer_fake = SummaryWriter(f"logs/fake")  # Log directory for fake images
writer_real = SummaryWriter(f"logs/real")  # Log directory for real images
step = 0

for epochs in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):  # Iterate through batches of real images
        real = real.view(-1, 784).to(device)  # Flatten images from [batch_size, 1, 28, 28] to [batch_size, 784]
        batch_size = real.shape[0]  # Get the number of images in the current batch

        # Train Discriminator
        # The goal is to maximize log(D(x)) + log(1-D(G(z))) where:
        # - D(x) is the probability that the Discriminator classifies real images as real
        # - D(G(z)) is the probability that the Discriminator classifies fake images as real

        noise = torch.randn(batch_size, z_dim).to(device)  # Generate random noise for the Generator
        fake = gen(noise)  # Generate fake images using the Generator with the random noise
        disc_real = disc(real).view(
            -1)  # Discriminator's output for real images; reshape to [batch_size] for loss computation

        # Compute loss for real images
        # We expect real images to be classified as real (label = 1)
        lossD_real = criterion(disc_real,
                               torch.ones_like(disc_real))  # BCE loss between Discriminator output and real labels

        # Compute loss for fake images
        disc_fake = disc(fake.detach()).view(
            -1)  # Discriminator's output for fake images; detach to avoid gradient propagation
        # Detaching the fake images avoids affecting the Generator's gradients during Discriminator's backward pass
        lossD_fake = criterion(disc_fake,
                               torch.zeros_like(disc_fake))  # BCE loss between Discriminator output and fake labels

        # Total Discriminator loss
        lossD = (lossD_real + lossD_fake) / 2  # Average of losses for real and fake images

        disc.zero_grad()  # Reset gradients for Discriminator
        lossD.backward()  # Compute gradients for Discriminator loss
        opt_disc.step()  # Update Discriminator weights

        # Train Generator
        # The goal is to maximize log(D(G(z))) where:
        # - D(G(z)) is the probability that the Discriminator classifies generated images as real

        output = disc(fake).view(-1)  # Discriminator's output for fake images; reshape for loss computation
        lossG = criterion(output, torch.ones_like(
            output))  # BCE loss between Discriminator output and real labels (for fake images)

        gen.zero_grad()  # Reset gradients for Generator
        lossG.backward()  # Compute gradients for Generator loss
        opt_gen.step()  # Update Generator weights

        # Log results to TensorBoard
        if batch_idx == 0:
            # Print progress for the current epoch
            print(f"Epoch [{epochs}/{num_epochs}] Batch {batch_size}/{len(loader)} \
                  loss D: {lossD:.4f}, loss G: {lossG:.4f}")

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)  # Generate images from fixed noise for visualization
                data = real.reshape(-1, 1, 28, 28)  # Reshape real images for visualization
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)  # Create grid of generated images
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)  # Create grid of real images

                # Add images to TensorBoard logs
                writer_fake.add_image("MNIST fake img", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST real img", img_grid_real, global_step=step)

                step += 1  # Increment step for TensorBoard logging
