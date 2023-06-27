'''
    Generate images using pretrained network pickle.
'''

import os
import numpy as np
import PIL.Image
from tqdm import trange 
import argparse

import dnnlib
import torch
import loader

import misc
from misc import crop_max_rectangle as crop

# Generate images using pretrained network pickle.
def run(model, gpus, output_dir, images_num, truncation_psi, ratio):
    # Set GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = torch.device("cuda")

    # Load pre-trained network
    print("Loading networks...")
    G = loader.load_network(model)["Gs"].to(device)
    print(G)
    # Make output directory
    print("Generate and save images...")
    os.makedirs(output_dir, exist_ok = True)

    for i in trange(images_num):
        # Sample latent vector
        z = torch.randn([1, *G.input_shape[1:]], device = device)
        # # Generate an image
        imgs = G(z, truncation_psi = truncation_psi)[0].cpu().numpy()
        min = np.amin(imgs)
        max = np.amax(imgs)
        print(min)
        print(max)
        # pp = 4

        imgp = misc.to_pil(imgs[0])
        min = np.amin(imgp)
        max = np.amax(imgp)
        print(min)
        print(max)

        print('   **************  ')
        # Output images name pattern
        # pattern = "{}/sample_{{:06d}}.png".format(output_dir)
        # Save the image
        # img = crop((), ratio).save(pattern.format(i))

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description = "Generate images with the GANformer")
    parser.add_argument("--model",              help = "Filename for a snapshot to resume", type = str)
    parser.add_argument("--gpus",               help = "Comma-separated list of GPUs to be used (default: %(default)s)", default = "0", type = str)
    parser.add_argument("--output-dir",         help = "Root directory for experiments (default: %(default)s)", default = "images", metavar = "DIR")
    parser.add_argument("--images-num",         help = "Number of images to generate (default: %(default)s)", default = 32, type = int)
    parser.add_argument("--truncation-psi",     help = "Truncation Psi to be used in producing sample images (default: %(default)s)", default = 0.7, type = float)
    parser.add_argument("--ratio",              help = "Crop ratio for output images (default: %(default)s)", default = 1.0, type = float)
    # Pretrained models' ratios: CLEVR (0.75), Bedrooms (188/256), Cityscapes (0.5), FFHQ (1.0)
    args, _ = parser.parse_known_args()
    run(**vars(args))

if __name__ == "__main__":
    # main()
    ro = '/home/na/1_Face_morphing/1_code/2_morphing/5_gansformer-main/pytorch_version/'

    #####   model pretrained in 256*256 size   ######
    # model = ro + 'models/bedrooms-snapshot.pkl'
    # model = ro + 'models/ffhq-snapshot.pkl'
    model = ro + 'models/ffhq-snapshot-1024_v2.pkl'
    # model = ro + 'models/clevr-snapshot.pkl'
    # model = ro + 'models/cityscapes-snapshot.pkl'
    gpus = '3'
    output_dir = ro + 'images/face_1024/'
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    images_num = 100
    truncation_psi = 0.7
    batch_size = 8
    # Pretrained models' ratios:
    #   CLEVR (0.75), Bedrooms (188/256), Cityscapes (0.5), FFHQ (1.0)
    ratio = 1.0

    #######  generate images  ##################
    run(model, gpus, output_dir, images_num, truncation_psi, ratio)