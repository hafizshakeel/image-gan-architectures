import torch
import config
from tqdm import tqdm
from torchvision.utils import save_image


def train_fn(disc, gen, loader, opt_disc, opt_gen, bce, l1, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            disc_real = disc(x, y)
            disc_real_loss = bce(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(x, y_fake.detach())
            disc_fake_loss = bce(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            disc.zero_grad()
            d_scaler.scale(disc_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            gen_fake = disc(x, y_fake)
            gen_fake_loss = bce(gen_fake, torch.ones_like(gen_fake))  # trick disc to believe these are real ones
            L1 = l1(y_fake, y) * config.L1_LAMBDA  # detach y_fake in discriminator
            gen_loss = gen_fake_loss + L1

            gen.zero_grad()
            g_scaler.scale(gen_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

        if batch_idx % 10 == 0:
            loop.set_postfix(
                disc_real=torch.sigmoid(disc_real).mean().item(),
                disc_fake=torch.sigmoid(disc_fake).mean().item(),
            )

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

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
