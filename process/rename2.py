import scipy.io as sio
import os
import shutil

# file = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/FRGC/9_percept_alex_Wingloss/f179/04649d17/000000_1.490221.mat'
file = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/FRGC/10_percept_alex_Wingloss/f179/04649d17/000011_0.692834.mat'
d = sio.loadmat(file)
d = 2

# ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/'
# src_path = ro + '1_raw_pairs_real_faces_1024_self-collected/'
# dst_path = ro + '2_Self_pairs/'
# if os.path.exists(dst_path) is False:
#     os.makedirs(dst_path)
#
# imgs = os.listdir(src_path)
# for img in imgs:
#     id = img.split('_')[0]
#     dst_fold = dst_path + id
#     if os.path.exists(dst_fold) is False:
#         os.makedirs(dst_fold)
#
#     new_img = img.split('_')[1] + img.split('_')[2]
#     shutil.copy(src_path + img, dst_fold + '/' + new_img)
#
