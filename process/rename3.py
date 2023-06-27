import os
import shutil


ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/'
src_path = ro + '4_raw_aligned_1024_rename/'

acc_path = '/home/na/1_Face_morphing/2_data/1_self_collect/V2_real_faces/1_styleGAN/3_raw_aligned_1024_rename_V2_stylegan_upgrade/'
dst_path = ro + '2_self_pairs_v2/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

sexs = ['female', 'male']

for sex in sexs:
    if sex == 'female': tag = 'f'
    if sex == 'male': tag = 'm'
    count = 1
    imgs = os.listdir(acc_path + sex)
    for img in imgs:
        name1 = img.split('.')[0].split('_')[0] + '.png'
        name2 = img.split('.')[0].split('_')[1] + '.png'
        id = tag + str(count)
        if os.path.exists(dst_path + id) is False:
            os.makedirs(dst_path + id)
        shutil.copy(src_path + sex + '/' + name1, dst_path + id + '/' + name1)
        shutil.copy(src_path + sex + '/' + name2, dst_path + id + '/' + name2)
        count += 1


