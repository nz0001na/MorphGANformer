import os
import shutil

ro = '/home/na/1_Face_morphing/2_data_for_demorphing/'
src_path = ro + '0_bona_fide_Bb/'
dst_A_path = ro + '0_bona_fide_A/'
dst_B_path = ro + '0_bona_fide_B/'
if os.path.exists(dst_A_path) is False:
    os.makedirs(dst_A_path)
if os.path.exists(dst_B_path) is False:
    os.makedirs(dst_B_path)

acc_path = ro + '2_trusted_live_B/'
A_list = os.listdir(acc_path)
for imA in A_list:
    name = imA.split('.')[0]
    id = name.split('_')[0]
    sbj = name.split('_')[1]

    if os.path.exists(src_path + id) is False: continue
    img_list = os.listdir(src_path + id)
    for img in img_list:
        if img.startswith(sbj) is True:
            src = src_path + id + '/' + img
            dst = dst_B_path + id + '_' + img
            shutil.move(src, dst)

