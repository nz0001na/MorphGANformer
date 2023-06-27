import os
import moviepy.video.io.ImageSequenceClip as mv


ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/5_latentcode_exp/'
src_path = ro + '1_use_fixed_image/'
fps = 1

folder = os.listdir(src_path)
for fold in folder:
    if os.path.isdir(src_path + fold) is False:
        continue
    dst_path = src_path + fold + '.mp4'

    img_list = os.listdir(src_path + fold)
    img_list.sort()

    image_files = [os.path.join(src_path + fold, img)
                   for img in img_list
                   if img.endswith(".png")]
    clip = mv.ImageSequenceClip(image_files, fps=fps)

    clip.write_videofile(dst_path)