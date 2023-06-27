'''
    Get latent code of targte images
    using LBP based matching score loss
    with pretrained network pickle. [1024x1024]

    latent code: (17,32)
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
from PIL import Image
from tqdm import tqdm
import numpy as np
from numpy import dot, sqrt
import csv
import cv2

import skimage.feature as sf
from PIL import Image
import scipy.io as sio

import misc
from misc import crop_max_rectangle as crop
import lpips
import loader

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'


def LBP_feature(I):
    image = cv2.imread(I, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    lbp = sf.local_binary_pattern(image, n_points, radius, METHOD)
    fea = lbp.flatten()
    return fea

def LBP_feature_im(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray, (224, 224))
    lbp = sf.local_binary_pattern(image, n_points, radius, METHOD)
    fea = lbp.flatten()
    return fea

def cosine_distance(x, y):
    return 1 - dot(x, y) / (sqrt(dot(x, x)) * sqrt(dot(y, y)))

def LBPLoss(qry_fea, trg_fea):
    match_dst = cosine_distance(qry_fea, trg_fea)
    distance = torch.from_numpy(np.asarray(match_dst))
    return torch.sum(distance)



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

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

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


def projection(args, path_img1, G, latent_mean, latent_std):
    trg_fea = LBP_feature(path_img1)

    imgs = image_transform(path_img1)
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    latent_in.requires_grad = True
    optimizer = torch.optim.Adam([latent_in], lr=args.lr)
    # optimizer = torch.optim.SGD([latent_in], lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,, weight_decay=0, amsgrad=False

    pbar = tqdm(range(args.step))
    latent_path = []
    min_distance = 1.0

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr, args.lr_rampdown, args.lr_rampup)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen_raw = G(latent_n, args.truncation_psi)[0].cpu().detach().numpy()
        # img_gen_raw = G(latent_n, args.truncation_psi)[0].cpu().detach()
        img_gen_255 = misc.to_pil(img_gen_raw[0])
        img_gen_2552 = np.asarray(img_gen_255)
        qry_fea = LBP_feature_im(img_gen_2552)

        match_distance = LBPLoss(qry_fea, trg_fea)
        match_distance.requires_grad = True

        optimizer.zero_grad()
        match_distance.backward()
        optimizer.step()

        distance = match_distance.detach().cpu().numpy()
        if distance < min_distance:
            min_distance = distance
            latent_path.append(latent_n.detach().clone())
            # Save the image
            output_dir = args.path_to_gen
            if os.path.exists(output_dir) is False:
                os.makedirs(output_dir)
            pattern = "{}/{{:06d}}_{{:08f}}.png".format(output_dir)
            im = img_gen_raw
            dst = crop(misc.to_pil(im[0]), args.ratio).save(pattern.format(i, min_distance))

        pbar.set_description(
            (
                f" distance: {match_distance.item():.8f};"
                f" min_distance: {min_distance.item():.8f}; lr: {lr:.6f}"
            )
        )

    return latent_path[-1]




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda")

    ro = '/home/na/1_Face_morphing/2_data/99_exp_ganformer/2_improve_loss/'
    img = ro + '04827d02.png'
    dst_path_morph = ro + 'LBP_loss/'
    if os.path.exists(dst_path_morph) is False:
        os.makedirs(dst_path_morph)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='models/ffhq-snapshot-1024_v2.pkl')
    parser.add_argument("--path_to_gen", type=str, default=dst_path_morph)
    # parser.add_argument("--path_to_latent", type=str, default=dst_path_latent)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--n_mean_latent", type=int, default=10000)
    parser.add_argument("--step", type=int, default=5000)  # cal W

    # parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--truncation_psi", type=float, default=0.7)

    parser.add_argument("--noise_regularize", type=float, default=1e5)
    parser.add_argument("--w_plus", action="store_true")
    args = parser.parse_args()

    # Load pre-trained network
    print("Loading networks...")
    G = loader.load_network(args.model)["Gs"].to(device)

    with torch.no_grad():
        # Sample latent vector
        noise_sample = torch.randn(args.n_mean_latent, *G.input_shape[1:], device=device)
        latent_mean = noise_sample.mean(0)
        latent_std = ((noise_sample - latent_mean).pow(2).sum() / args.n_mean_latent) ** 0.5


    # percept = lpips.PerceptualLoss(
    #     model="net-lin", net="squeeze", use_gpu=True  #device.startswith("cuda")
    # )
    # # 'vgg', 'alex', 'squeeze'
    # MSE = torch.nn.MSELoss().to(device)

    w1 = projection(args, img, G, latent_mean, latent_std)

    # # # Generate an image
    # imgs = G(w1, args.truncation_psi)[0].cpu().numpy()
    # dst_img = 'images/2_frgc_data/frgc_exp_1024/04827d02_latent_vgg.png'
    # img = crop(misc.to_pil(imgs[0]), args.ratio).save(dst_img)

