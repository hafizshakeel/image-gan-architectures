import torch
import torch.nn as nn

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