import os
import shutil

ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3_frgc/3_GANformer_Morphs/'
src = ro + '4_morphs_selected_best_final/'
acc = ro + '2_morphs/'
dst = ro + '4_morphs_selected_best_final_mat/'

ids = os.listdir(src)
for id in ids:
    if os.path.exists(dst + id + '/') is False:
        os.makedirs(dst + id + '/')
    if id in ['pair7', 'pair4']: continue
    print(id)
    imgs = os.listdir(src + id)
    name = imgs[0][0:len(imgs[0])-4]
    sec = name.split('_')
    if len(sec) == 3:
        new_name = name + '.mat'
    else:
        new_name = sec[2] + '_' + sec[3] + '_' + sec[4] + '.mat'

    src_img = acc + id + '/' + new_name
    dst_img = dst + id + '/' + new_name
    shutil.copy(src_img, dst_img)



