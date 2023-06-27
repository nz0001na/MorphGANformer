import os
import shutil
import cv2

ro = '/home/na/1_Face_morphing/2_data_morph/'
src_path = ro + '4_citer_ganformer_frgc_v2/1_latent_V2/'
dst_path = ro + '4_citer_ganformer_frgc_v2/1_latent_jpg/'


ids = os.listdir(src_path)
for id in ids:
    print(id)
    names = os.listdir(src_path + id)
    for name in names:
        if os.path.exists(dst_path + id + '/' + name) is False:
            os.makedirs(dst_path + id + '/' + name)

        imgs = os.listdir(src_path + id + '/' + name)
        for img in imgs:
            if img.split('.')[2] == 'mat':continue
            # if id == '195' and img == '000001_2.256126+000020_1.751330.png':continue
            src_im = src_path + id + '/' + name + '/' + img
            dst_im = dst_path + id + '/' + name + '/' + img[0:len(img)-3] + 'jpg'

            if os.path.exists(dst_im) is True: continue
            print(id + '/' + img)
            if img.split('.')[2] == 'jpg':
                shutil.move(src_im, dst_im)
                continue

            if img.split('.')[2] == 'png':
                im = cv2.imread(src_im)
                cv2.imwrite(dst_im, im)
                os.remove(src_im)
                continue
