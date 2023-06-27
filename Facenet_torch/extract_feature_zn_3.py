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

ro = '/home/na/1_Face_morphing/2_data/2_FRGC-Morphs/0_for_citer/'
img_path = ro + '2_cropped/'
dst_path = ro + '3_FaceNet/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

flist = os.listdir(img_path)
for fold in flist:
    print(fold)
    dst_ff = dst_path + fold
    if os.path.exists(dst_ff) is False: os.makedirs(dst_ff)
    img_list = os.listdir(img_path + fold + '/')
    for img in img_list:
        image = img_path + fold + '/' + img
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





