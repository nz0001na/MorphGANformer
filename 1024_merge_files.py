'''
    do morphing using two bona fide faces
    step 1: combine all bonafides from different results

'''


import os
import torch
import shutil



if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda")

    ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/3_Self_v1_exp/1_ganformer_morphed_data/0_GANformer_morph_raw2/'
    src_path = ro + '1_base_morphs_hog/'
    dst_path_morph = ro + '1_base_morphs_hog_merge/'

    version_list = os.listdir(src_path)
    for versn in version_list:
        dst_versn = dst_path_morph + versn + '/'
        if os.path.exists(dst_versn) is False:
            os.makedirs(dst_versn)

        vars = os.listdir(src_path + versn)
        for var in vars:
            print(versn + ' / ' + var)
            ids = os.listdir(src_path + versn + '/' + var)
            for id in ids:
                dst_id = dst_versn + id + '/'

                names = os.listdir(src_path + versn + '/' + var + '/' + id)
                for name in names:
                    if len(name.split('.')) > 1: continue
                    dst_fold = dst_id + name + '/'
                    if os.path.exists(dst_fold) is False:
                        os.makedirs(dst_fold)

                    imgs = os.listdir(src_path + versn + '/' + var + '/' + id + '/' + name)
                    for img in imgs:
                        src_img = src_path + versn + '/' + var + '/' + id + '/' + name + '/' + img
                        dst_img = dst_fold + img
                        shutil.copy(src_img, dst_img)




