import sys
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from wgan_to_wgan_gp_ptp import Discriminator, Generator, initialize_weights
from torch.utils.tensorboard import SummaryWriter
import torch.autograd
from tqdm import tqdm


# GRADIENT PENALTY -- https://arxiv.org/pdf/1704.00028
# Improved training of W-GAN - this paper is all about having a better way of enforcing the lipschitz constraint
# see implementation algorithm 1
def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    # print(f"Real shape: {real.shape}, Fake shape: {fake.shape}")  # Debug print

    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)  # one epsilon value for each example
    interpolated_images = real * epsilon + fake * (1 - epsilon)  # see algo line 6.
    # Create interpolated images by blending real and fake images. For a given epsilon value (e.g., 0.1),
    # this combines 10% of the real image with 90% of the fake image.
    # This interpolation generates images that are a weighted mixture of real and fake, with the weights determined
    # by epsilon. By varying epsilon between 0 and 1, we obtain a range of interpolated images
    # between the real and fake images.

    # calculate critic score
    mixed_score = critic(interpolated_images)  # called mixed score because scores from the interpolated images

    # calculate gradient
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True
    )[0]
    # we are computing gradient of the mixed score wrt the interpolated images, [0] first element of those

    gradient = gradient.view(gradient.shape[0], -1)  # examples, flatten all others dim
    gradient_norm = gradient.norm(2, dim=1)  # L2 norm, taking the norm across that dim that we just flatten
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)   # algo line 7 second part of expression
    return gradient_penalty


# HYPERPARAMETERS FROM PAPER - SEE ALGO1 WGAN
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 2  # 64 in paper
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 128
FEATURES_C = 64
FEATURES_G = 64
CRITIC_ITERATION = 5  # Critic here mean discriminator --> from paper
LAMBDA_GP = 10
BETAS = (0.0, 0.9)


# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Ensure both height and width are resized
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])  # mean, std
])


# DATASETS (MNIST --> CHANNELS=1 & CELEB --> CHANNELS=3)
# print(os.listdir())
dataset = datasets.ImageFolder(root="dataset", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODELS
critic = Discriminator(channels_img=CHANNELS_IMG, features_d=FEATURES_C).to(device=DEVICE)
gen = Generator(channel_noise=NOISE_DIM, channels_img=CHANNELS_IMG, feature_g=FEATURES_G).to(device=DEVICE)

# WEIGHT INITIALIZATION
initialize_weights(critic)
initialize_weights(gen)

# OPTIM AND LOSS
optim_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=BETAS)
optim_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=BETAS)

# FIXED_NOISE
fixed_noise = torch.randn((32, NOISE_DIM, 1, 1)).to(device=DEVICE)

# SAVE IMAGES
os.makedirs("generated_images", exist_ok=True)

# # for tensorboard plotting
writer_real = SummaryWriter(log_dir=f"logs/real")
writer_fake = SummaryWriter(log_dir=f"logs/fake")
step = 0

# Anomaly detection
torch.autograd.set_detect_anomaly(True)

# SET MODELS TO TRAIN MODE IF THEY AREN'T
critic.train()
gen.train()

for epochs in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device=DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, NOISE_DIM, 1, 1).to(device=DEVICE)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # Start from algo line 2
        for _ in range(CRITIC_ITERATION):
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=DEVICE)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp  # Algo line 7
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optim_critic.step()

        # Train Generator: min -E[critic(gen_fake)]
        # OR max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        # Start from algo line 9
        get_out = critic(fake).reshape(-1)
        loss_gen = -torch.mean(get_out)  # algo line 10
        gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        # print to tensorboard
        if batch_idx % 2 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(
                f" Epochs:{epochs}/{NUM_EPOCHS}, Batch: {batch_idx}/{len(loader)},"
                f" LossD: {loss_critic:.2f}, LossG: {loss_gen:.2f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)

                for i in range(5):
                    torchvision.utils.save_image(fake[i], f"generated_images/epoch_{epochs}_image_{i}.png", normalize=True)

                # take out (up to) 32 examples
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Actual Image", img_grid_real, global_step=step)

                step += 1

                gen.train()
                critic.train()

# Close the writers
# writer_fake.close()
# writer_real.close()