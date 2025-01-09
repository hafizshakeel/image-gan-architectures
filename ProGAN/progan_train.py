""" Training of ProGAN using WGAN-GP loss """

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from progan_model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config

# Enabling cuDNN's benchmarking feature for performance optimization, especially on GPUs with a fixed input size.
torch.backends.cudnn.benchmark = True


# get_loader function to create a DataLoader for each image resolution during training
# This is necessary because the image size changes progressively during training.
def get_loader(image_size):
    # Apply transformations to resize images, normalize, and apply random horizontal flips.
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), 
         transforms.ToTensor(), 
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.Normalize([0.5 for _ in range(config.CHANNELS_IMG)], [0.5 for _ in range(config.CHANNELS_IMG)])]
    )
    # Choose batch size based on image size and VRAM capacity.
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    # Load dataset using ImageFolder.
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    # Create DataLoader with dataset and batch size, shuffle data and use multiple workers for efficiency.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    return loader, dataset


def main():
    # Initialize the generator and discriminator (called critic in WGAN-GP).
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)

    # Initialize optimizers for both networks (generator and critic) with the Adam optimizer.
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

    # Initialize scalers for mixed precision training (FP16 support).
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # Initialize TensorBoard for logging training metrics and visualizing progress.
    writer = SummaryWriter(f"logs/gan1")

    # Load pre-trained models if specified in the config.
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE)

    # Set both networks to training mode.
    gen.train()
    critic.train()

    tensorboard_step = 0
    # Start training at the image size defined in the configuration.
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    
    # Train progressively at different image sizes.
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1  # Initialize alpha for progressive growing.
        loader, dataset = get_loader(4 * 2 ** step)  # Update image size for current step (4, 8, 16, 32, 64...)
        print(f"Current image size: {4 * 2 ** step}")

        # Train for each epoch.
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            # Call training function to train the generator and critic.
            tensorboard_step, alpha = train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen,
                                               tensorboard_step, writer, scaler_gen, scaler_critic)

            # Save the model checkpoints periodically.
            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

        # Move to the next image size for the next training phase.
        step += 1


if __name__ == "__main__":
    main()
