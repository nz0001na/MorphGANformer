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
from sklearn.svm import SVC

model = InceptionResnetV1(pretrained='vggface2').eval()
nn_name = 'facenet'


ro = '/home/na/1_MAD/2_Data/'
src_path = ro + '3_few_shot_dbs_exp_sets/2_db_sets/C_set/'
shots = ['0_shot', '1_shot', '5_shot']
folder_list = ['train/real', 'train/fake', 'test/real', 'test/fake']

dst_path = ro + '4_few_shot_dbs_exp_other_feature_C_set/FaceNet/'
if os.path.exists(dst_path) is False: os.makedirs(dst_path)

for shot in shots:
    print(shot)
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []
    for folder in folder_list:
        print(folder)
        set = folder.split('/')[0]
        subset = folder.split('/')[1]
        img_list = os.listdir(src_path + shot + '/' + folder)
        for i in range(len(img_list)):
            # if i >3:continue
            im = img_list[i]
            fi = src_path + shot + '/' + folder + '/' + im
            I = cv2.imread(fi)
            I = cv2.resize(I, (224, 224))
            img1 = torch.FloatTensor(np.array(I))
            input_imgs_r = torch.reshape(img1, [-1, 224, 224, 3])
            input_imgs_r = torch.clamp(input_imgs_r, 0, 255).to(torch.float32)
            input_imgs_r = (input_imgs_r - 127.5) / 128.0
            input_imgs_r = input_imgs_r.permute(0, 3, 1, 2)
            img_embedding = model(input_imgs_r)
            fea = np.array(img_embedding.detach().numpy()).flatten()
            if set == 'train':
                train_feature.append(fea.flatten())
                if subset == 'fake':
                    train_label.append(0)
                else:
                    train_label.append(1)
            if set == 'test':
                test_feature.append(fea.flatten())
                if subset == 'fake':
                    test_label.append(0)
                else:
                    test_label.append(1)

    train_feature = np.asarray(train_feature)
    test_feature = np.asarray(test_feature)
    train_label = np.asarray(train_label).flatten()
    test_label = np.asarray(test_label).flatten()
    print(np.shape(train_feature))
    print(np.shape(train_label))
    print(np.shape(test_feature))
    print(np.shape(test_label))

    clf = SVC(kernel='linear', probability=True)
    clf.fit(train_feature, train_label)
    accuracy = clf.score(test_feature, test_label)
    print('******* Accuracy: {}'.format(accuracy))

    # # predict and generate positives and negatives
    positives = []
    negatives = []
    # binary, return (n_samples,2)
    # get the probability score on positive label
    y_pred = clf.predict_proba(test_feature)
    test_pred = y_pred[:,1]
    for i in range(len(test_pred)):
        if test_label[i] == 1:
            positives.append(test_pred[i])
        else:
            negatives.append(test_pred[i])

    sio.savemat(dst_path + shot + '_pos_neg_scores.mat', {'positives': positives, 'negatives': negatives})


