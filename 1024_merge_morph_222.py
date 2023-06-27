'''
    do morphing using two bona fide faces
    step 2:
            do morphing

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


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='models/ffhq-snapshot-1024_v2.pkl')
    parser.add_argument("--truncation_psi", type=float, default=0.7)
    parser.add_argument("--ratio", type=float, default=1.0)
    args = parser.parse_args()


    ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/3_Self_v1_exp/1_ganformer_morphed_data/0_GANformer_morph_raw/'
    src_path = ro + 'R3_pecept_wing_MSE/1_bonafides/'
    dst_path_morph = ro + 'R3_pecept_wing_MSE/2_morphs/'

    # Load pre-trained network
    print("Loading networks...")
    G = loader.load_network(args.model)["Gs"].to(device)

    ids = os.listdir(src_path)
    for id in ids:
        print(id)
        dst_id = dst_path_morph + id + '/'
        if os.path.exists(dst_id) is False:
            os.makedirs(dst_id)

        name_list = os.listdir(src_path + id)
        names = []
        for name in name_list:
            if len(name.split('.')) > 1: continue
            names.append(name)
        imgs1 = os.listdir(src_path + id + '/' + names[0])
        imgs2 = os.listdir(src_path + id + '/' + names[1])

        for im1 in imgs1:
            if im1.split('.')[2] == 'png': continue
            dw1 = sio.loadmat(src_path + id + '/' + names[0] + '/' + im1)
            w1 = dw1['w']

            for im2 in imgs2:
                if im2.split('.')[2] == 'png': continue
                dw2 = sio.loadmat(src_path + id + '/' + names[1] + '/' + im2)
                w2 = dw2['w']

                new_name = im1[0:len(im1)-4] + '+' + im2[0:len(im2)-4]
                dst_mat = dst_id + new_name + '.mat'
                dst_img = dst_id + new_name + '.jpg'
                if os.path.exists(dst_img) is True: continue

                dw = 0.5 * w1 + 0.5 * w2
                W = torch.from_numpy(dw).to(torch.device("cuda"))
                fina_im = G(W, args.truncation_psi)[0].cpu().numpy()
                img = crop(misc.to_pil(fina_im[0]), args.ratio).save(dst_img)

                # dw = W.detach().cpu().numpy()
                # dw = W
                dataw = {}
                dataw['w'] = dw
                sio.savemat(dst_mat, dataw)

