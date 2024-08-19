import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from dcgan_model import Discriminator, Generator, initialize_weights
from torch.utils.tensorboard import SummaryWriter

# HYPERPARAMETERS FROM PAPER - THIS ARCHITECTURE IS ALSO SENSITIVE TO HYPERPARAMETERS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1  # 3 for RGB images
NOISE_DIM = 100
FEATURES_D = 64
FEATURES_G = 64
BETAS = (0.5, 0.9)
ROOT_DIR = "dataset/"

# TRANSFORM
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])  # mean, std
])

# DATASETS (MNIST --> CHANNELS=1 & CELEB --> CHANNELS=3)
# dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms) # Here, images are first downlaoed
# and store in root folder. there is no need to write the whole dataset class.
dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODELS
disc = Discriminator(channels_img=CHANNELS_IMG, features_d=FEATURES_D).to(device=DEVICE)
gen = Generator(channel_noise=NOISE_DIM, channels_img=CHANNELS_IMG, feature_g=FEATURES_G).to(device=DEVICE)

# WEIGHT INITIALIZATION
initialize_weights(disc)
initialize_weights(gen)

# OPTIM AND LOSS
optim_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=BETAS)
optim_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=BETAS)
criterion = nn.BCELoss()

# FIXED_NOISE
fixed_noise = torch.randn((32, NOISE_DIM, 1, 1)).to(device=DEVICE)

# TRAINING LOOP
writer_real = SummaryWriter(log_dir=f"logs/real")
writer_fake = SummaryWriter(log_dir=f"logs/fake")
step = 0

# SET MODELS TO TRAIN MODE IF THEY AREN'T
disc.train()
gen.train()

for epochs in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device=DEVICE)
        noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1)).to(device=DEVICE)
        fake = gen(noise)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)  # since D(x) and Nx1x1x1 --> N
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)  # since D(G(z))
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)  # To use fake in generator training without redefining
        optim_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)  # since D(G(z)) & Output just batch images
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        optim_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f" Epochs:{epochs}/{NUM_EPOCHS}, Batch: {batch_idx}/{len(loader)}, LossD: {lossD:.2f}, LossG: {lossG:.2f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Actual Image", img_grid_real, global_step=step)

                step += 1

# Close the writers
writer_fake.close()
writer_real.close()

""" To use tensorboard in https://colab.research.google.com/, run the following commands:
%load_ext tensorboard
%tensorboard --logdir logs
command to create and download folder as zip file -->  !zip -r log.zip logs/ 
"""

""" NOTE: THIS ARCHITECTURE IS ALSO SENSITIVE TO HYPERPARAMETERS """
