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

ro = '/home/na/1_MAD/2_Data/1_good_FRGC2.0/'
dst = '/home/na/1_MAD/2_Data/2_good_FRGC2.0_features/'
if os.path.exists(dst) is False:
    os.makedirs(dst)

dbs = os.listdir(ro)
for db in dbs:
    print(db)

    db_path = ro + db + '/'
    train_file = db_path + 'train.csv'
    test_file = db_path + 'test.csv'

    img_path = db_path

    dst_path = dst + db + '/'
    if os.path.exists(dst_path) is False:
        os.makedirs(dst_path)

    # read train feature
    train_feature = []
    train_label = []
    f = csv.reader(open(train_file, 'r', newline=''))
    for row in f:
        if row[0] == 'image_name':continue
        fi = img_path + 'train/' + row[0]
        I = cv2.imread(fi)
        I = cv2.resize(I, (224, 224))
        img1 = torch.FloatTensor(np.array(I))
        input_imgs_r = torch.reshape(img1, [-1, 224, 224, 3])
        input_imgs_r = torch.clamp(input_imgs_r, 0, 255).to(torch.float32)
        input_imgs_r = (input_imgs_r - 127.5) / 128.0
        input_imgs_r = input_imgs_r.permute(0, 3, 1, 2)
        img_embedding = model(input_imgs_r)
        fea = np.array(img_embedding.detach().numpy()).flatten()
        train_feature.append(fea)
        train_label.append(int(row[1]))

    # read test feature
    test_feature = []
    test_label = []
    f = csv.reader(open(test_file, 'r', newline=''))
    for row in f:
        if row[0] == 'image_name': continue
        fi = img_path + 'test/' + row[0]
        I = cv2.imread(fi)
        I = cv2.resize(I, (224, 224))
        img1 = torch.FloatTensor(np.array(I))
        input_imgs_r = torch.reshape(img1, [-1, 224, 224, 3])
        input_imgs_r = torch.clamp(input_imgs_r, 0, 255).to(torch.float32)
        input_imgs_r = (input_imgs_r - 127.5) / 128.0
        input_imgs_r = input_imgs_r.permute(0, 3, 1, 2)
        img_embedding = model(input_imgs_r)
        fea = np.array(img_embedding.detach().numpy()).flatten()
        test_feature.append(fea)
        test_label.append(int(row[1]))

    # save data
    dict = {}
    dict['train_feature'] = np.asarray(train_feature)
    dict['train_label'] = np.asarray(train_label).flatten()
    dict['test_feature'] = np.asarray(test_feature)
    dict['test_label'] = np.asarray(test_label).flatten()
    dst_file = dst_path + nn_name + '_train_test.mat'
    sio.savemat(dst_file, mdict=dict)

    # print(db)
    print(np.shape(train_feature))
    print(np.shape(train_label))
    print(np.shape(test_feature))
    print(np.shape(test_label))

