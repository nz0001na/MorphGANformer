'''
    Get latent code of targte images
    using perceptual loss
    with pretrained network pickle. [256x256]

    latent code: (1, 16, 512)
'''

import argparse
import math
import os
import cv2
import sys
import pickle
import torch
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import scipy.io as sio
import numpy as np
import csv

import misc
from misc import crop_max_rectangle as crop
import lpips
import loader
# from model import Generator
import csv

def get_lr(t, initial_lr, rampdown, rampup):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise

# transform image to 256x256
def image_transform(file_path):
    resize = 256
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
        ]
    )

    # imgs = []
    # I = Image.open(file_path)
    # I1 = I.convert("RGB")
    # img = transform(I1)
    # # I2 = I1 / 255.0
    # imgs.append(img)
    # imgs = torch.stack(imgs, 0).to(device)

    imgs = []
    img = transform(Image.open(file_path).convert("RGB"))
    imgs.append(img)
    imgs = torch.stack(imgs, 0).to(device)

    return imgs

def normalize(a):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = transform(a)
    return img

def image_transform_our(img_path):
    img1 = np.array(cv2.imread(img_path))
    img1 = cv2.resize(img1, (256,256))
    img2 = img1[..., ::-1]
    img = img2 / 255.0
    # converted to a torch tensor of appropriate dimensions and normalized to
    # be given as input to the VGG - 16 network.
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
    x1 = img[np.newaxis]
    x2 = x1.transpose([0, 3, 1, 2]) * 1
    X = (torch.FloatTensor(x2) - mean) / std
    return X


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


def projection(args, path_img1, percept, G, latent_mean, latent_std):
    imgs = image_transform(path_img1)
    # pp = imgs.cpu().detach().numpy()

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    latent_in.requires_grad = True
    optimizer = optim.Adam([latent_in], lr=args.lr)

    # loss_list = []
    pbar = tqdm(range(args.step))
    latent_path = []
    min_loss = 1.0

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr, args.lr_rampdown, args.lr_rampup)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen_raw = G(latent_n, args.truncation_psi)[0].cpu().detach().numpy()
        # img_gen_raw = normalize(img_gen_raw0)
        # img_gen_raw = (img_gen_raw0 - np.min(img_gen_raw0)) / (np.max(img_gen_raw0) - np.min(img_gen_raw0))
        # print(img_gen_raw)
        c = img_gen_raw[0]
        z = misc.to_pil(img_gen_raw[0])
        d = normalize(z)
        img_gen = d[None,:]
        dd = d.cpu().detach().numpy()
        img_gen = img_gen.cuda()
        # Convert images to variables to support gradients
        img_gen = Variable(img_gen, requires_grad=True)

        batch, channel, height, width = img_gen_raw.shape

        if height > 256:
            factor = height // 256
            img_gen_raw = img_gen_raw.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen_raw = img_gen_raw.mean([3, 5])


        optimizer.zero_grad() # ?
        p_loss = percept(img_gen, imgs, normalize=False).sum()
        # loss_list.append(str(p_loss.item()))
        # p_loss.detach().cpu().numpy()

        loss = p_loss
        # optimizer.zero_grad()  # ?
        loss.backward()
        optimizer.step()

        num_loss = p_loss.detach().cpu().numpy()
        # print(num_loss)
        if num_loss < min_loss:
            min_loss = num_loss
            latent_path.append(latent_n.detach().clone())
            # Save the image
            # output_dir = 'images/example/P_stepV1_percept2/'
            # if os.path.exists(output_dir) is False:
            #     os.makedirs(output_dir)
            # pattern = "{}/sample_{{:06d}}_{{:04f}}.png".format(output_dir)
            # dst = crop(misc.to_pil(img_gen_raw[0]), args.ratio).save(pattern.format(i, min_loss))

        pbar.set_description(
            (
                f" perceptual: {p_loss.item():.4f};"
                #f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )

    return latent_path[-1]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt", type=str, default='stylegan2-ffhq-config-f.pt')
    # parser.add_argument("--path_to_morph", type=str, default=dst_path_morph)
    # parser.add_argument("--path_to_latent", type=str, default=dst_path_latent)
    parser.add_argument("--size", type=int, default=256)
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
    ro = '/home/na/1_Face_morphing/1_code/2_morphing/5_gansformer-main_V2_256/pytorch_version/'
    model = ro + 'models/ffhq-snapshot.pkl'
    print("Loading networks...")
    G = loader.load_network(model)["Gs"].to(device)

    with torch.no_grad():
        # Sample latent vector
        noise_sample = torch.randn(args.n_mean_latent, *G.input_shape[1:], device=device)
        latent_mean = noise_sample.mean(0)
        latent_std = ((noise_sample - latent_mean).pow(2).sum() / args.n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="squeeze", use_gpu=True  #device.startswith("cuda")
    )

    path_img1 = 'images/example/2.png'
    w1 = projection(args, path_img1, percept, G, latent_mean, latent_std)

    # # Generate an image
    imgs = G(w1, args.truncation_psi)[0].cpu().numpy()
    # Save the image
    # pattern = "{}/sample_{{:06d}}.png".format(output_dir)
    dst_img = 'images/example/v1_morph02.png'
    img = crop(misc.to_pil(imgs[0]), args.ratio).save(dst_img)

