import os
import moviepy.video.io.ImageSequenceClip as mv


ro = '/home/na/1_Face_morphing/2_data/3_ganformer_v3/5_latentcode_exp/'
image_folder = ro + '1_fixed_E'
fps = 1

img_list = os.listdir(image_folder)
img_list.sort()

image_files = [os.path.join(image_folder, img)
               for img in img_list
               if img.endswith(".png")]
clip = mv.ImageSequenceClip(image_files, fps=fps)

clip.write_videofile(ro + '1_fixed_E.mp4')