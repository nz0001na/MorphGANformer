'''
    1.Get latent codes of two bona fide faces
    2. generate morph faces
    3. use landmarks to warp by delaunay

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
from scipy.spatial import Delaunay



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

# get landmarks by image path
def get_landmarks_img(path_img):
    image = cv2.imread(path_img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shape = predictor(gray, rects[0])
    shape = shape_to_np(shape)
    y = torch.from_numpy(shape)
    y = y.type(torch.DoubleTensor)
    return y

# get landmarks by generated image of G
def get_landmarks_G(img_gen):
    img_g = img_gen.permute(0, 2, 3, 1)
    img_g = img_g.reshape([1024, 1024, 3]).cpu().detach().numpy()
    img_gI = cv2.normalize(img_g, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gray2 = cv2.cvtColor(img_gI, cv2.COLOR_BGR2GRAY)
    rects2 = detector(gray2, 1)
    if len(rects2) == 0:
        return None, 0
    shape2 = predictor(gray2, rects2[0])
    shape2 = shape_to_np(shape2)
    y_hat = torch.from_numpy(shape2)
    y_hat = y_hat.type(torch.DoubleTensor)
    return y_hat, 1


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps triangular regions from img_G to img_avg
def morphTriangle(img_G, img_avg, t_G, t_avg) :
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t_G]))
    r = cv2.boundingRect(np.float32([t_avg]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t_avg[i][0] - r[0]),(t_avg[i][1] - r[1])))
        t1Rect.append(((t_G[i][0] - r1[0]),(t_G[i][1] - r1[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img_G[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    imgRect = warpImage1

    # Copy triangular region of the rectangular patch to the output image
    img_avg[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_avg[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask



# ---------------- main -------------------
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda")

    # Load pre-trained network
    print("Loading networks...")
    model = 'models/ffhq-snapshot-1024_v2.pkl'
    G = loader.load_network(model)["Gs"] #.to(device)

    # Load landmark detector
    predictor_file = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_file)

    shape = (1024, 1024, 3)
    truncation_psi = 0.7
    ratio = 1.0
    extra_points = [[0, 0], [0, 341], [0, 682], [0, 1023],
                     [341, 0], [682, 0], [1023, 0], [1023, 341],
                     [1023, 682], [1023, 1023], [341, 1023], [682, 1023]]

    ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/example3/'
    name1 = '000080_0.927806'
    name2 = '000129_0.853518'
    img1 = ro + name1 + '.png'
    img2 = ro + name2 + '.png'

    # im1 = cv2.imread(img1)
    # im1 = np.float32(im1)
    # im2 = cv2.imread(img2)
    # im2 = np.float32(im2)
    # flag = 3  # 1:im1,  2:im2   3:G,

    # 1. get latent codes
    ltcode1 = sio.loadmat(ro + name1 + '.mat')
    ltcode2 = sio.loadmat(ro + name2 + '.mat')
    w1 = torch.from_numpy(ltcode1['w'])
    w2 = torch.from_numpy(ltcode2['w'])

    # 2. do morphing
    W = 0.5 * w1 + 0.5 * w2
    G_im = G(W, truncation_psi)[0].cpu().numpy()
    dst_img = ro + 'morph_G.png'
    img = crop(misc.to_pil(G_im[0]), ratio).save(dst_img)

    # 3. get landmarks
    landmark1 = get_landmarks_img(img1)
    landmark2 = get_landmarks_img(img2)
    landmark_avg = list(torch.div(landmark1.add(landmark2), 2).detach().cpu().numpy())
    points_avg = [*landmark_avg, *np.asarray(extra_points)]
    tri_avg = Delaunay(points_avg)
    tri_indx_avg = tri_avg.simplices.tolist()

    G_im = torch.from_numpy(G_im)
    landmark_G, flag = get_landmarks_G(G_im)
    points_G = [*landmark_G.detach().cpu().numpy(), *np.asarray(extra_points)]
    # if flag == 3:
    #     points_G = [*landmark_G.detach().cpu().numpy(), *np.asarray(extra_points)]
    # if flag == 1:
    #     points_G = [*landmark1.detach().cpu().numpy(), *np.asarray(extra_points)]
    # if flag == 2:
    #     points_G = [*landmark2.detach().cpu().numpy(), *np.asarray(extra_points)]

    # Allocate space for final output
    imgMorph = np.zeros(shape, dtype=np.float32)

    # read morphed face
    im_G = cv2.imread(dst_img)
    # Convert Mat to float data type
    im_G = np.float32(im_G)

    # warp  for each triangles
    for line in tri_indx_avg:
        x = int(line[0])
        y = int(line[1])
        z = int(line[2])

        t_G = [points_G[x], points_G[y], points_G[z]]
        t_avg = [points_avg[x], points_avg[y], points_avg[z]]

        # Morph one triangle at a time.
        morphTriangle(im_G, imgMorph, t_G, t_avg)
        # if flag == 1:
        #     morphTriangle(im1, imgMorph, t_G, t_avg)
        # if flag == 2:
        #     morphTriangle(im2, imgMorph, t_G, t_avg)
        # if flag == 3:
        #     morphTriangle(im_G, imgMorph, t_G, t_avg)

    # Display Result
    final_name = ro + 'Morph_final.png'
    # cv2.imshow("Morphed Face", np.uint8(imgMorph))
    cv2.imwrite(final_name, np.uint8(imgMorph))

    print('done')

