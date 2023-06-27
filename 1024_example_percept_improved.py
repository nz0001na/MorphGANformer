'''
    Get latent code of targte images
    using perceptual loss
    with pretrained network pickle. [1024x1024]

    latent code: (17,32)

    improve performance?
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
from sklearn import preprocessing
import csv
import cv2

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
    resize = args.size
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


def projection(args, path_img1, percept, G, latent_mean, latent_std):
    resize = args.size
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Save the image
    output_dir = args.path_to_gen
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    imgs = image_transform(path_img1)
    imgs2 = cv2.imread(path_img1)
    min_trg = np.amin(imgs2)  # imgs.cpu().detach().numpy()
    max_trg = np.amax(imgs2)
    print('trg: ' + str(min_trg) + '  :  ' + str(max_trg))

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    latent_in.requires_grad = True
    optimizer = torch.optim.Adam([latent_in], lr=args.lr) #, weight_decay=0.0001
    # optimizer = torch.optim.SGD([latent_in], lr=args.lr)  #, momentum=0.9, weight_decay=1e-4
    # learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,, weight_decay=0, amsgrad=False

    pbar = tqdm(range(args.step))
    latent_path = []
    min_loss = 100.0

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr, args.lr_rampdown, args.lr_rampup)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        # noise_strength.requires_grad = True
        lmin = np.amin(latent_n.cpu().detach().numpy())
        lmax = np.amax(latent_n.cpu().detach().numpy())

        # 1 3  1024  1024
        img_gen_raw = G(latent_n, args.truncation_psi)[0].cpu().detach().numpy()
        rmin = np.amin(img_gen_raw)
        rmax = np.amax(img_gen_raw)
        p1_l = percept(torch.from_numpy(img_gen_raw), imgs).sum()
        p1_loss = p1_l.detach().cpu().numpy()
        # print(str(rmin) + ' : ' + str(rmax) + ' : ' + str(p1_loss))

        img_gen_norm0 = img_gen_raw[0].reshape(-1, 1)
        img_gen_norm = min_max_scaler.fit_transform(img_gen_norm0)
        nmin = np.amin(img_gen_norm)
        nmax = np.amax(img_gen_norm)
        img_gen_norm = img_gen_norm.reshape(1, 3, 1024, 1024)
        p3_l = percept(torch.from_numpy(img_gen_norm), imgs).sum()
        p3_loss = p3_l.detach().cpu().numpy()
        # print(str(nmin) + ' : ' + str(nmax) + ' : ' + str(p3_loss))

        img_gen_pil = misc.to_pil(img_gen_raw[0])
        img_gen_tsr = transform(img_gen_pil)
        img_gen_tsr = img_gen_tsr[None, :]
        tmin = np.amin(img_gen_tsr.cpu().detach().numpy())
        tmax = np.amax(img_gen_tsr.cpu().detach().numpy())
        # print(tmin)
        # print(tmax)
        p2_l = percept(img_gen_tsr, imgs).sum()
        p2_loss = p2_l.detach().cpu().numpy()
        # print(p2_loss)
        # print(str(tmin) + ' : ' + str(tmax) + ' : ' + str(p2_loss))


        img_gen_255 = misc.to_pil(img_gen_raw[0])
        img_gen_2552 = np.asarray(img_gen_255)
        qry = torch.from_numpy(img_gen_2552)
        # qry = qry[None, :]
        # qry = torch.transpose(qry, 0, 3, 1, 2)

        target = torch.from_numpy(imgs2)
        target = target[None, :]
        target = target.transpose(2, 0, 1)

        p4_l = percept(qry, target).sum()
        p4_loss = p4_l.detach().cpu().numpy()

        # img_gen_2552 = crop(img_gen_255, args.ratio)
        # img_gen_np = img_gen_tsr.cpu().detach().numpy()
        # img_gen_np = img_gen_np.transpose(1, 2, 0)

        # min_qry1 = np.amin(img_gen_raw[0])
        # max_qry1 = np.amax(img_gen_raw[0])
        # print('qry_gen_raw: ' + str(min_qry1) + '  :  ' + str(max_qry1))

        # norm_255 = misc.to_pil(img_gen_raw[0])
        # min_qry2 = np.amin(norm_255)
        # max_qry2 = np.amax(norm_255)
        # print('qry_norm255: ' + str(min_qry2) + '  :  ' + str(max_qry2))

        # norm_1 = misc.adjust_range(norm_255, [0, 255], [-1, 1])
        # min_qry3 = np.amin(norm_1)
        # max_qry3 = np.amax(norm_1)
        # print('qry_norm1: ' + str(min_qry3) + '  :  ' + str(max_qry3))

        # norm_1 = norm_1.transpose(2, 0, 1)
        # img_gen = torch.from_numpy(norm_1)
        # img_gen = img_gen[None, :]

        # img_gen_3 = transform(norm_255)
        # img_gen = img_gen_3[None, :]
        # min_qry = np.amin(img_gen.cpu().detach().numpy())
        # max_qry = np.amax(img_gen.cpu().detach().numpy())
        # print('qry: ' + str(min_qry) + '  :  ' + str(max_qry))

        # optimizer.zero_grad()
        # p2_loss = percept(img_gen, imgs).sum()

        """
                Pred and target are Variables.
                If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
                If normalize is False, assumes the images are already between [-1,+1]

                Inputs pred and target are Nx3xHxW
                Output pytorch Variable N long
                """
        p_loss = p4_l
        optimizer.zero_grad()
        p_loss.backward()
        optimizer.step()

        # latent_path.append(latent_n.detach().clone())

        num_loss = p_loss.detach().cpu().numpy()
        if num_loss < min_loss:
            min_loss = num_loss
            latent_path.append(latent_n.detach().clone())
        #     # Save the image
        #     output_dir = args.path_to_gen
        #     if os.path.exists(output_dir) is False:
        #         os.makedirs(output_dir)
        #     pattern = "{}/{{:06d}}_{{:04f}}.png".format(output_dir)
        #     # dst = crop(misc.to_pil(img_gen_raw[0]), args.ratio).save(pattern.format(i, min_loss))
        #     dst = crop(misc.to_pil(img_gen_3.detach().cpu().numpy()), args.ratio).save(pattern.format(i, min_loss))
            pattern1 = "{}/{{:06d}}_1_{{:04f}}.png".format(output_dir)
            crop(misc.to_pil(img_gen_raw[0]), args.ratio).save(pattern1.format(i, p1_loss))
            pattern3 = "{}/{{:06d}}_3_{{:04f}}.png".format(output_dir)
            crop(misc.to_pil(img_gen_norm[0]), args.ratio).save(pattern3.format(i, p3_loss))
            pattern2 = "{}/{{:06d}}_2_{{:04f}}.png".format(output_dir)
            crop(misc.to_pil(img_gen_tsr[0].cpu().detach().numpy()), args.ratio).save(pattern2.format(i, p2_loss))

            pattern4 = "{}/{{:06d}}_4_{{:04f}}.png".format(output_dir)
            crop(misc.to_pil(img_gen_raw[0].cpu().detach().numpy()), args.ratio).save(pattern4.format(i, p4_loss))


        pbar.set_description(
            (
                f" loss: {p4_loss.item():.4f};"
                # f" noise: {noise_strength.item(): .4f};"
                # f" max: {lmax:.4f};"
                # f" min: {lmin:.4f};"
                f" min_loss: {min_loss.item():.4f}; lr: {lr:.6f}"
            )
        )

    return latent_path[-1]




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    ro = '/home/na/1_Face_morphing/2_data/99_exp_ganformer/5_improve/'
    out_dir = ro + 'exp29'
    os.makedirs(out_dir, exist_ok=True)
    src_img = ro + '04827d02.png'

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)  # 0.05
    parser.add_argument("--noise_ramp", type=float, default=0.75)  # 0.75

    parser.add_argument("--model", type=str, default='models/ffhq-snapshot-1024_v2.pkl')
    # parser.add_argument("--model", type=str, default='models/ffhq-snapshot.pkl')
    parser.add_argument("--path_to_gen", type=str, default=out_dir)
    # parser.add_argument("--path_to_latent", type=str, default=dst_path_latent)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--n_mean_latent", type=int, default=10000)
    parser.add_argument("--step", type=int, default=1500)  # cal W

    # parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)

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
        # latent_ini = torch.randn(*G.input_shape[1:], device=device)
        latent_std = ((noise_sample - latent_mean).pow(2).sum() / args.n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="alex", use_gpu=True  #device.startswith("cuda")
    )
    # 'vgg', 'alex', 'squeeze'

    w1 = projection(args, src_img, percept, G, latent_mean, latent_std)

    # # # Generate an image
    # imgs = G(w1, args.truncation_psi)[0].cpu().numpy()
    # dst_img = 'images/2_frgc_data/frgc_exp_1024/04827d02_latent_vgg.png'
    # img = crop(misc.to_pil(imgs[0]), args.ratio).save(dst_img)

