import shutil
import os

ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/3_Self_v1_exp/1_ganformer_morphed_data/0_GANformer_morph_raw2/7_best_Morph_selection/'
fli = ['1_final_H1',
       '2_final_H2',
       '3_final_H3',
       '4_final_H4']
dst_path = ro + 'final/'
for fo in fli:
    ids = os.listdir(ro + fo + '/')
    for id in ids:
        if os.path.exists(dst_path + id + '/') is False:
            os.makedirs(dst_path + id + '/')

        imgs = os.listdir(ro + fo + '/' + id + '/')
        for im in imgs:
            src_img = ro + fo + '/' + id + '/' + im
            dst_img = dst_path + id + '/' + im
            shutil.copy(src_img, dst_img)

