'''
    Do morphing for raw data pairs
    using Perceptual loss
    with pretrained network pickle. [256x256]
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
import scipy.io as sio
import numpy as np
import csv

import misc
from misc import crop_max_rectangle as crop
import lpips
import loader
# from model import Generator
import csv

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


# def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
#     lr_ramp = min(1, (1 - t) / rampdown)
#     lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
#     lr_ramp = lr_ramp * min(1, t / rampup)
#
#     return initial_lr * lr_ramp

def get_lr(t, initial_lr,  rampup, rampdown):
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

# transform image to 256x256
def image_transform(file_path):
    resize = 256
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



def projection2(args, path_img1, percept, G, latent_mean, latent_std):
    imgs = image_transform(path_img1)
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    # latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
    # latent_in = latent_mean[None, :]
    # log_size = int(math.log(args.size, 2))
    # n_latent = log_size * 2 - 2
    # latent_in = latent_in.unsqueeze(1).repeat(1, n_latent, 1, 1)
    # latent_in = latent_in.unsqueeze(1).repeat(1, n_latent, 1)
    latent_in.requires_grad = True
    optimizer = optim.Adam([latent_in], lr=args.lr)


    # loss_list = []
    pbar = tqdm(range(args.step))
    latent_path = []

    min_loss = 1.0

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        # print(lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        img_gen_raw = G(latent_n, args.truncation_psi)[0].cpu().detach().numpy()
        # print(img_gen_raw)
        batch, channel, height, width = img_gen_raw.shape




        if height > 256:
            factor = height // 256
            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        img_gen = torch.from_numpy(img_gen_raw)
        p_loss = percept(img_gen, imgs).sum()
        # loss_list.append(np.ndarray(p_loss.detach().cpu().numpy()))

        loss = p_loss
        optimizer.zero_grad()  # ?
        loss.backward()
        optimizer.step()

        num_loss = p_loss.detach().cpu().numpy()
        # if (i + 1) % 100 == 0:
        if num_loss < min_loss:
            min_loss = num_loss
            latent_path.append(latent_n.detach().clone())
            # Save the image
            # output_dir = 'images/example/P_stepV2_01/'
            # if os.path.exists(output_dir) is False:
            #     os.makedirs(output_dir)
            # pattern = "{}/sample_{{:06d}}_{{:04f}}.png".format(output_dir)
            # dst = crop(misc.to_pil(img_gen_raw[0]), args.ratio).save(pattern.format(i, min_loss))

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f};"
                #f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )
    # with open('images/example/P_v2_im1.csv', 'w') as f:
    #     ft = csv.writer(f)
    #     ft.writerows(loss_list)

    return latent_path[-1]


def projection(args, path_img1, percept, G, latent_mean, latent_std):
    imgs = image_transform(path_img1)
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    latent_in.requires_grad = True
    optimizer = optim.Adam([latent_in], lr=args.lr)

    loss_list = []
    pbar = tqdm(range(args.step))
    latent_path = []
    min_loss = 1.0

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr, args.lr_rampdown, args.lr_rampup)
        # print('*************************')
        # print(lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        # print(str(noise_strength.item()))
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen_raw = G(latent_n, args.truncation_psi)[0].cpu().detach().numpy()
        # print(img_gen_raw)
        batch, channel, height, width = img_gen_raw.shape

        if height > 256:
            factor = height // 256
            img_gen_raw = img_gen_raw.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen_raw = img_gen_raw.mean([3, 5])

        img_gen = torch.from_numpy(img_gen_raw)
        optimizer.zero_grad() # ?
        p_loss = percept(img_gen, imgs).sum()
        loss_list.append(p_loss.detach().cpu().numpy())

        loss = p_loss
        # optimizer.zero_grad()  # ?
        loss.backward()
        optimizer.step()

        num_loss = p_loss.detach().cpu().numpy()
        # print(num_loss)
        if num_loss < min_loss:
            min_loss = num_loss
            latent_path.append(latent_n.detach().clone())
            # # Save the image
            # output_dir = 'images/example/P_stepV1_im1/'
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


# python morphing.py --ckpt  stylegan2-ffhq-config-f.pt --size 1024 --path_to_img1 /path/to/img1/ --path_to_img2 /path/to/img2/

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda")

    ro = '/home/na/1_Face_morphing/2_data/1_self_collect/AA_real_raw_v2/'
    src_path = ro + '3_raw_aligned_1024_rename_V2/'
    dst_path_morph = ro + '3_raw_aligned_1024_rename_V2_ganformer_percept/'
    dst_path_raw = ro + '3_raw_aligned_1024_rename_V2_ganformer_percept_raw/'
    fil_path = ro + '0_raw_aligned_1024_rename_V2_crop_ArcFace/'
    if os.path.exists(dst_path_morph) is False:
        os.makedirs(dst_path_morph)
    if os.path.exists(dst_path_raw) is False:
        os.makedirs(dst_path_raw)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt", type=str, default='stylegan2-ffhq-config-f.pt')
    # parser.add_argument("--path_to_morph", type=str, default=dst_path_morph)
    # parser.add_argument("--path_to_latent", type=str, default=dst_path_latent)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--n_mean_latent", type=int, default=10000)
    parser.add_argument("--step", type=int, default=1000)  # cal W

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

    n_mean_latent = 10000
    with torch.no_grad():
        # Sample latent vector
        noise_sample = torch.randn(args.n_mean_latent, *G.input_shape[1:], device=device)
        latent_mean = noise_sample.mean(0)
        latent_std = ((noise_sample - latent_mean).pow(2).sum() / args.n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=True  #device.startswith("cuda")
    )

    items = ['male', 'female']
    for it in items:
        src_path2 = src_path + it + '/'
        dst_path_morph2 = dst_path_morph + it + '/'
        if os.path.exists(dst_path_morph2) is False:
            os.makedirs(dst_path_morph2)
        dst_path_raw2 = dst_path_raw + it + '/'
        if os.path.exists(dst_path_raw2) is False:
            os.makedirs(dst_path_raw2)

        fil = fil_path + it + '_simi.csv'
        f = csv.reader(open(fil, 'r'))
        for row in f:
            if row[0] == 'img1': continue
            re = float(row[2])
            if re < 0.5: continue
            print(row)
            img1 = row[0]
            img2 = row[1]
            path_img1 = src_path2 + img1
            path_img2 = src_path2 + img2
            final_name = img1.split('.')[0] + '_' + img2.split('.')[0]
            dst_img1 = dst_path_raw2 + final_name + '_A.png'
            dst_img2 = dst_path_raw2 + final_name + '_B.png'
            dst_img = dst_path_morph2 + final_name + '.png'
            if os.path.exists(dst_img) is True: continue

            w1 = projection(args, path_img1, percept, G, latent_mean, latent_std)
            w2 = projection(args, path_img2, percept, G, latent_mean, latent_std)
            im1 = G(w1, args.truncation_psi)[0].cpu().numpy()
            im2 = G(w2, args.truncation_psi)[0].cpu().numpy()
            imgv1 = crop(misc.to_pil(im1[0]), args.ratio).save(dst_img1)
            imgv2 = crop(misc.to_pil(im2[0]), args.ratio).save(dst_img2)
            W = 0.5 * w1 + 0.5 * w2
            imgs = G(W, args.truncation_psi)[0].cpu().numpy()
            # # Save the image
            img = crop(misc.to_pil(imgs[0]), args.ratio).save(dst_img)

