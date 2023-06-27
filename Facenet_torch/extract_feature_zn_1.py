'''
this code is to extract feature of DenseNet121 network
code is from: https://keras.io/api/applications/
'''

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import cv2
import os
import csv
import numpy as np
import scipy.io as sio

model = InceptionResnetV1(pretrained='vggface2').eval()
nn_name = 'facenet'

ro = '/home/na/1_Face_morphing/2_data/1_self_collect/AA_real_raw_v2/'
flist = [
    # '3_raw_aligned_1024_rename_V2_stylegan_crop_224x224/',
    '3_raw_aligned_1024_rename_V2_stylegan_upgrade_crop_224x224/',
    '0_raw_aligned_1024_rename_V2_crop_224x224'
    # '2_morph_stylegan_1024_sharp', '4_morph_stylegan_512_sharp',
    # '2_morph_OpenCV_1024'
]

img_path = ro + 'cropped_version/'
dst_path = ro + 'FaceNet_version/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

for fold in flist:
    subs = os.listdir(img_path + fold)
    for sub in subs:

        img_list = os.listdir(img_path + fold +'/'+ sub )
        dst_ff = dst_path + fold + '/'+ sub
        if os.path.exists(dst_ff) is False:
            os.makedirs(dst_ff)

        for img in img_list:
            image = img_path + fold + '/'+ sub +'/'  + img
            if os.path.exists(dst_ff + '/' + img.split('.')[0] + '.mat'): continue

            I = cv2.imread(image)
            I = cv2.resize(I, (224, 224))
            img1 = torch.FloatTensor(np.array(I))
            input_imgs_r = torch.reshape(img1, [-1, 224, 224, 3])
            input_imgs_r = torch.clamp(input_imgs_r, 0, 255).to(torch.float32)
            input_imgs_r = (input_imgs_r - 127.5) / 128.0
            input_imgs_r = input_imgs_r.permute(0, 3, 1, 2)
            img_embedding = model(input_imgs_r)
            fea = np.array(img_embedding.detach().numpy()).flatten()

            dict = {}
            dict['feature'] = np.asarray(fea)
            dict['name'] = img
            dst_file = dst_ff + '/' + img.split('.')[0] + '.mat'
            sio.savemat(dst_file, mdict=dict)





