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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import fbeta_score

model = InceptionResnetV1(pretrained='vggface2').eval()
nn_name = 'facenet'


ro = '/home/na/1_MAD/2_Data/5_morphing_fingerprint_V3_128x128/'
src_path = ro + '5_morphing_fingerprint_128x128/'
split_file = ro + '5_morphing_fingerprint_train_test_split_single/'
dbs = ['5_self1024_224x224', '2_FERET_270x270', '3_FRGC_270x270', '4_CelebA_128x128', '1_FRLL_270x270']

dst_path = ro + '5_morphing_fingerprint_train_test_data2/FaceNet/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

fig_path = ro + '5_morphing_fingerprint_train_test_fig/FaceNet/'
if os.path.exists(fig_path) is False:
    os.makedirs(fig_path)

for db in dbs:
    print(db)

    train_list = []
    train_feature = []
    train_label = []
    test_list = []
    test_feature = []
    test_label = []
    folders = os.listdir(split_file + db + '/')
    for fold in folders:
        # if fold not in ['1_Jake_LMA', '1_Jake_real']: continue
        print(fold)
        train_file = split_file + db + '/' + fold + '/train_list.csv'
        test_file = split_file + db + '/' + fold + '/test_list.csv'
        # train
        train_count = 0
        f = csv.reader(open(train_file, 'r'))
        for row in f:
            if train_count >= 1: break
            if row[0] == 'name': continue
            img = row[0]
            label = int(row[1])
            fi = src_path + img
            I = cv2.imread(fi)
            I = cv2.resize(I, (224, 224))
            img1 = torch.FloatTensor(np.array(I))
            input_imgs_r = torch.reshape(img1, [-1, 224, 224, 3])
            input_imgs_r = torch.clamp(input_imgs_r, 0, 255).to(torch.float32)
            input_imgs_r = (input_imgs_r - 127.5) / 128.0
            input_imgs_r = input_imgs_r.permute(0, 3, 1, 2)
            img_embedding = model(input_imgs_r)
            feature = np.array(img_embedding.detach().numpy()).flatten()
            train_feature.append(feature.flatten())
            train_label.append(label)
            train_count += 1
        # test
        # test_count = 0
        ff = csv.reader(open(test_file, 'r'))
        for row in ff:
            if row[0] == 'name': continue
            img = row[0]
            label = int(row[1])
            fi = src_path + img
            I = cv2.imread(fi)
            I = cv2.resize(I, (224, 224))
            img1 = torch.FloatTensor(np.array(I))
            input_imgs_r = torch.reshape(img1, [-1, 224, 224, 3])
            input_imgs_r = torch.clamp(input_imgs_r, 0, 255).to(torch.float32)
            input_imgs_r = (input_imgs_r - 127.5) / 128.0
            input_imgs_r = input_imgs_r.permute(0, 3, 1, 2)
            img_embedding = model(input_imgs_r)
            feature = np.array(img_embedding.detach().numpy()).flatten()
            test_feature.append(feature.flatten())
            test_label.append(label)
            # test_count += 1

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
    test_pred = clf.predict(test_feature)

    f1 = f1_score(test_label, test_pred, average='macro')
    print('******* f1: {}'.format(f1))
    f_beta = fbeta_score(test_label, test_pred, average='macro', beta=0.5)
    print('******* f_beta: {}'.format(f_beta))
    y_pred = clf.predict_proba(test_feature)
    roc = roc_auc_score(test_label, y_pred, multi_class='ovr')
    print('******* roc: {}'.format(roc))

    # non normalized
    # mt = confusion_matrix(test_label, test_pred, labels=clf.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=mt, display_labels=clf.classes_)
    # disp.plot()
    # plt.show()
    # plt.savefig(fig_path + db + '_train_test1.jpg')
    # plt.close()
    # print(mt)

    # # normalized
    # mt2 = confusion_matrix(test_label, test_pred, normalize='all')
    # disp = ConfusionMatrixDisplay(confusion_matrix=mt2, display_labels=clf.classes_)
    # disp.plot()
    # plt.show()
    # plt.savefig(fig_path + db + '_train_test2.jpg')
    # plt.close()
    # print(mt2)

    dict = {}
    dict['train_feature'] = train_feature
    dict['train_label'] = train_label
    # d = {}
    dict['test_feature'] = test_feature
    dict['test_label'] = test_label
    sio.savemat(dst_path + db + '_train_test.mat', dict)

