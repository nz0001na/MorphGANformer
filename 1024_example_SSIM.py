'''
    Get latent code of targte images
    using SSIM loss
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

def l2(p0, p1, range=255.):
    return .5*np.mean((p0 / range - p1 / range)**2)

def psnr(p0, p1, peak=255.):
    return 10*np.log10(peak**2/np.mean((1.*p0-1.*p1)**2))

def dssim(p0, p1, range=255.):
    from skimage.measure import compare_ssim
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.

def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1,2,0))

def np2tensor(np_obj):
     # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def projection(args, path_img1, G, latent_mean, latent_std):
    imgs = image_transform(path_img1)

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    latent_in.requires_grad = True
    optimizer = torch.optim.Adam([latent_in], lr=args.lr, weight_decay=0.0001)
    # optimizer = torch.optim.SGD([latent_in], lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,, weight_decay=0, amsgrad=False


    pbar = tqdm(range(args.step))
    latent_path = []
    min_loss = 100

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr, args.lr_rampdown, args.lr_rampup)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen_raw = G(latent_n, args.truncation_psi)[0].cpu().detach().numpy()
        img_gen = torch.from_numpy(img_gen_raw)
        img_gen_raw2 = img_gen.permute(0, 3, 1, 2)

        pre = tensor2np(img_gen_raw2)
        trg = tensor2np(imgs)

        # optimizer.zero_grad()
        # p_loss = percept(img_gen, imgs).sum()
        # psnr_loss = psnr(img_gen.to(torch.device("cuda")), imgs)
        ssim_loss = dssim(pre.flatten(), trg.flatten())
        ssim_loss = torch.FloatTensor(np.array(ssim_loss))
        ssim_loss = ssim_loss.to(torch.device("cuda"))
        # psnr_loss = np2tensor(psnr_loss).to(torch.device("cuda"))
        ssim_loss.requires_grad = True


        # all_loss = args.lamda * p_loss + (1 - args.lamda) * mse_loss

        optimizer.zero_grad()
        # p_loss.backward()
        ssim_loss.backward()
        # all_loss.backward()
        optimizer.step()

        num_loss = ssim_loss.detach().cpu().numpy()
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
                # f" perceptual: {p_loss.item():.4f};"
                f" ssim: {ssim_loss:.4f};"
                # f" ALL: {all_loss.item():.4f};"
                f" min_loss: {min_loss.item():.4f}; lr: {lr:.4f}"
            )
        )

    return latent_path[-1]


# ---------------- main -------------------
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda")

    out_dir = 'images/2_frgc_data/frgc_exp_1024/ssim'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='models/ffhq-snapshot-1024.pkl')
    parser.add_argument("--path_to_gen", type=str, default=out_dir)
    # parser.add_argument("--path_to_latent", type=str, default=dst_path_latent)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--n_mean_latent", type=int, default=10000)
    parser.add_argument("--step", type=int, default=5000)  # cal W

    parser.add_argument("--lamda", type=float, default=0.5)
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

    #
    # percept = lpips.PerceptualLoss(
    #     model="net-lin", net="alex", use_gpu=True  #device.startswith("cuda")
    # )
    # # 'vgg', 'alex', 'squeeze'
    #
    # MSE = torch.nn.MSELoss().to(device)

    path_img1 = 'images/2_frgc_data/frgc_exp_1024/04827d02.png'
    w1 = projection(args, path_img1, G, latent_mean, latent_std)

    # # Generate an image
    # imgs = G(w1, args.truncation_psi)[0].cpu().numpy()
    # dst_img = 'images/2_frgc_data/frgc_exp_1024/04827d02_latent_vgg_mse.png'
    # img = crop(misc.to_pil(imgs[0]), args.ratio).save(dst_img)

