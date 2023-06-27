'''
    Get latent code of target images
    using wing loss + perceptual + MSE
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
import dlib

import misc
from misc import crop_max_rectangle as crop
import lpips
import loader
import wing_loss
import cv2


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

# L2 norm
def l2(p0, p1, range=255.):
    return .5*np.mean((p0 / range - p1 / range)**2)

# landmarks: rectangle to bounding box
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# landmarks: shape to numpy
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# get latent code
def projection(args, path_img1, percept, WingLoss, MSE, G, latent_mean, latent_std, detector, predictor):
    # get target landmarks
    image = cv2.imread(path_img1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shape = predictor(gray, rects[0])
    shape = shape_to_np(shape)
    y = torch.from_numpy(shape)
    y = y.type(torch.DoubleTensor)

    imgs = image_transform(path_img1)
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    latent_in.requires_grad = True
    optimizer = torch.optim.Adam([latent_in], lr=args.lr)
    # optimizer = torch.optim.SGD([latent_in], lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,, weight_decay=0, amsgrad=False

    pbar = tqdm(range(args.step))
    latent_path = []
    min_loss = 100.0

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr, args.lr_rampdown, args.lr_rampup)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        img_gen_raw = G(latent_n, args.truncation_psi)[0].cpu().detach().numpy()
        img_gen = torch.from_numpy(img_gen_raw)
        img_g = img_gen.permute(0, 2, 3, 1)
        img_g = img_g.reshape([1024, 1024, 3]).cpu().detach().numpy()
        img_gI = cv2.normalize(img_g, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        gray2 = cv2.cvtColor(img_gI, cv2.COLOR_BGR2GRAY)
        rects2 = detector(gray2, 1)
        if len(rects2) == 0:
            continue
        shape2 = predictor(gray2, rects2[0])
        shape2 = shape_to_np(shape2)
        y_hat = torch.from_numpy(shape2)
        y_hat = y_hat.type(torch.DoubleTensor)

        # y_hat.requires_grad = True
        w_loss = WingLoss(y_hat, y)
        w_loss.requires_grad = True
        p_loss = percept(img_gen, imgs).sum()
        mse_loss = MSE(img_gen.to(torch.device("cuda")), imgs)
        mse_loss.requires_grad = True

        all_loss = p_loss + args.lamda * w_loss + args.beta * mse_loss
        optimizer.zero_grad()
        # p_loss.backward()
        # mse_loss.backward()
        all_loss.backward()
        optimizer.step()

        num_loss = all_loss.detach().cpu().numpy()
        if num_loss < min_loss:
            min_loss = num_loss
            latent_path.append(latent_n.detach().clone())
            # Save the image
            output_dir = args.path_to_gen
            if os.path.exists(output_dir) is False:
                os.makedirs(output_dir)
            pattern = "{}/{{:06d}}_{{:04f}}.png".format(output_dir)
            dst = crop(misc.to_pil(img_gen_raw[0]), args.ratio).save(pattern.format(i, min_loss))

        pbar.set_description(
            (
                f" p_loss: {p_loss.item():.4f};"
                f" w_loss: {args.lamda * w_loss.item():.4f};"
                f" mse_loss: {args.beta * mse_loss.item():.4f};"
                # f" ALL: {all_loss.item():.4f};"
                # f" min_loss: {min_loss:.4f};"
                f" lr: {lr:.4f}"
            )
        )

    return latent_path[-1]


# ---------------- main -------------------
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda")

    predictor_file = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_file)

    output_dir = 'images/2_frgc_data/frgc_exp_1024/wing_sqz_MSE/'
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='models/ffhq-snapshot-1024_v2.pkl')
    parser.add_argument("--path_to_gen", type=str, default=output_dir)
    # parser.add_argument("--path_to_latent", type=str, default=dst_path_latent)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--n_mean_latent", type=int, default=10000)
    parser.add_argument("--step", type=int, default=5000)  # cal W

    parser.add_argument("--lamda", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.01)

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


    percept = lpips.PerceptualLoss(
        model="net-lin", net="squeeze", use_gpu=True  #device.startswith("cuda")
    )
    # # 'vgg', 'alex', 'squeeze'
    #
    MSE = torch.nn.MSELoss().to(device)
    WingLoss = wing_loss.WingLoss()

    path_img1 = 'images/2_frgc_data/frgc_exp_1024/04827d02.png'
    w1 = projection(args, path_img1, percept, WingLoss, MSE, G, latent_mean, latent_std, detector, predictor)

