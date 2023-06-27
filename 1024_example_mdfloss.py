'''
    Get latent code of targte images
    using MDF loss
    with pretrained network pickle. [1024x1024]
'''

import argparse
import math
import os
import sys
import pickle
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import scipy.io as sio
import numpy as np
import csv
from mdfloss import MDFLoss
import misc
from misc import crop_max_rectangle as crop
import lpips
import loader

torch.cuda.empty_cache()

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)

# set lr
def get_lr(t, initial_lr,  rampup, rampdown):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp

# set noise
def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

# transform image to 1024x1024
def image_transform(file_path):
    resize = 1024
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []
    I = Image.open(file_path)
    I1 = I.convert("RGB")
    img = transform(I1)
    imgs.append(img)
    imgs = torch.stack(imgs, 0).to(device)

    return imgs

def image_transform2(args, file_path, size):
    resize = min(args.size, size)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    img = transform(Image.open(file_path).convert("RGB").resize((size,size),Image.BILINEAR))
    imgs.append(img)
    imgs = torch.stack(imgs, 0).to(device)

    return imgs


def projection(args, path_img1, criterion, G, latent_mean, latent_std):
    imgsr = image_transform(path_img1)
    imgsr = imgsr.cuda()
    imgs = Variable(imgsr, requires_grad=False)

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    latent_in.requires_grad = True
    optimizer = optim.Adam([latent_in], lr=args.lr)

    loss_list = []
    pbar = tqdm(range(args.step))
    latent_path = []
    min_loss = 1000.0

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr, args.lr_rampup, args.lr_rampdown)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen_raw = G(latent_n, args.truncation_psi)[0].cpu().detach().numpy()
        batch, channel, height, width = img_gen_raw.shape

        # if height > 256:
        #     factor = height // 256
        #     img_gen_raw = img_gen_raw.reshape(
        #         batch, channel, height // factor, factor, width // factor, factor
        #     )
        #     img_gen_raw = img_gen_raw.mean([3, 5])

        img_gen = torch.from_numpy(img_gen_raw)
        img_gen = img_gen.cuda()
        # Convert images to variables to support gradients
        img_gen = Variable(img_gen, requires_grad=True)

        # optimizer.zero_grad()
        p_loss = criterion(imgs, img_gen)
        loss_list.append(p_loss.item())

        loss = p_loss
        optimizer.zero_grad()  # ?
        loss.backward()
        optimizer.step()

        num_loss = p_loss.detach().cpu().numpy()
        if num_loss < min_loss:
            min_loss = num_loss
            latent_path.append(latent_n.detach().clone())
            # Save the image
            output_dir = args.path_to_gen
            if os.path.exists(output_dir) is False:
                os.makedirs(output_dir)
            pattern = "{}/sample_{{:06d}}_{{:04f}}.png".format(output_dir)
            dst = crop(misc.to_pil(img_gen_raw[0]), args.ratio).save(pattern.format(i, min_loss))

        pbar.set_description(
            (
                f" mdf: {p_loss.item():.4f};"
                f" min_loss: {min_loss.item():.4f}; lr: {lr:.4f}"
            )
        )


    return latent_path[-1]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda")
    out_dir = 'images/2_frgc_data/frgc_exp_1024/mdf_denoising/'
    model = 'models/ffhq-snapshot-1024.pkl'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("--path_to_gen", type=str, default=out_dir)
    # parser.add_argument("--path_to_latent", type=str, default=dst_path_latent)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--step", type=int, default=5000) # cal W

    # parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_mean_latent", type=int, default=10000)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    # mdf
    parser.add_argument("--mdfapp", type=str, default='Denoising')
    # application == 'SISR', 'Denoising', 'JPEG'


    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--truncation_psi", type=float, default=0.7)
    parser.add_argument("--noise_regularize", type=float, default=1e5)
    parser.add_argument("--w_plus", action="store_true")
    args = parser.parse_args()

    # Load pre-trained network
    print("Loading networks...")
    G = loader.load_network(args.model)["Gs"].to(device)

    application = args.mdfapp
    if application == 'SISR':
        path_disc = "mdf-main/weights/Ds_SISR.pth"
    elif application == 'Denoising':
        path_disc = "mdf-main/weights/Ds_Denoising.pth"
    elif application == 'JPEG':
        path_disc = "mdf-main/weights/Ds_JPEG.pth"

    with torch.no_grad():
        # Sample latent vector
        noise_sample = torch.randn(args.n_mean_latent, *G.input_shape[1:], device=device)
        latent_mean = noise_sample.mean(0)
        latent_std = ((noise_sample - latent_mean).pow(2).sum() / args.n_mean_latent) ** 0.5


    criterion = MDFLoss(path_disc, cuda_available=True)

    # one image
    path_img1 = 'images/2_frgc_data/frgc_exp_1024/04827d02.png'
    w1 = projection(args, path_img1, criterion, G, latent_mean, latent_std)

    # # Generate an image
    imgs = G(w1, args.truncation_psi)[0].cpu().numpy()
    # Save the image
    dst_img = 'images/2_frgc_data/frgc_exp_1024/04827d02_mdf_denoise.png'
    img = crop(misc.to_pil(imgs[0]), args.ratio).save(dst_img)

