import sys
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from dcgan_to_wgan import Discriminator, Generator, initialize_weights
from torch.utils.tensorboard import SummaryWriter
import torch.autograd
from tqdm import tqdm

# HYPERPARAMETERS FROM PAPER - SEE ALGO1 WGAN
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.00005
NUM_EPOCHS = 3
BATCH_SIZE = 8  # 64 in paper
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 128
FEATURES_C = 64
FEATURES_G = 64
CRITIC_ITERATION = 5  # Critic here mean discriminator --> from paper
WEIGHT_CLIP = 0.01

# TRANSFORM
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])  # mean, std
])


# DATASETS (MNIST --> CHANNELS=1 & CELEB --> CHANNELS=3)
# print(os.listdir())
dataset = datasets.ImageFolder(root="dataset", transform=transform)  # Here, images are first downloaded
# and store in root folder. there is no need to write the whole dataset class. Images should be in subfolder.
# dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODELS
critic = Discriminator(channels_img=CHANNELS_IMG, features_d=FEATURES_C).to(device=DEVICE)
gen = Generator(channel_noise=NOISE_DIM, channels_img=CHANNELS_IMG, feature_g=FEATURES_G).to(device=DEVICE)

# WEIGHT INITIALIZATION
initialize_weights(critic)
initialize_weights(gen)

# OPTIM AND LOSS
optim_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
optim_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)

# FIXED_NOISE
fixed_noise = torch.randn((32, NOISE_DIM, 1, 1)).to(device=DEVICE)

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

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # Start from algo line 2
        for _ in range(CRITIC_ITERATION):
            noise = torch.randn(cur_batch_size, NOISE_DIM, 1, 1).to(device=DEVICE)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real)-torch.mean(critic_fake))  # See algo line 5 and formula in
            # rectangular box. Optimization algo like RMSprop are designed to minimize a loss function. Since we want to
            # maximize acc. to algo. (line 6), we transform this maximization problem into a minimization
            # problem by negating the loss. This approach aligns with the optimization strategy
            # where maximizing a value is equivalent to minimizing its negative.

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optim_critic.step()

            # Algo line 7 --> clip critic weights between -0.01, 0.01
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Train Generator: min -E[critic(gen_fake)]
        # OR max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        # Start from algo line 9
        get_out = critic(fake).reshape(-1)
        loss_gen = -torch.mean(get_out)  # algo line 10
        gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        # print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(
                f" Epochs:{epochs}/{NUM_EPOCHS}, Batch: {batch_idx}/{len(loader)},"
                f" LossD: {loss_critic:.2f}, LossG: {loss_gen:.2f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Actual Image", img_grid_real, global_step=step)

                step += 1
                gen.train()
                critic.train()

# Close the writers
writer_fake.close()
writer_real.close()

""" To use tensorboard in https://colab.research.google.com/, run the following commands:
%load_ext tensorboard
%tensorboard --logdir logs
command to create and download folder as zip file -->  !zip -r log.zip logs/ 
"""

""" NOTE: THIS ARCHITECTURE IS ALSO SENSITIVE TO HYPERPARAMETERS """
