'''
model code from: https://github.com/timesler/facenet-pytorch
use facenet to extract deep features

install facenet by pip:
    pip install facenet-pytorch

python environment under anaconda3/envs: 'Facenet'

'''


from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import cv2
import numpy as np

# If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
# Create an inception resnet (in eval mode):
model = InceptionResnetV1(pretrained='vggface2').eval()
# For a model pretrained on CASIA-Webface
# model = InceptionResnetV1(pretrained='casia-webface').eval()

# I = Image.open('002_007.jpg')
I = cv2.imread('002_007.jpg')
I = cv2.resize(I, (224,224))
img1 = torch.FloatTensor(np.array(I))
input_imgs_r = torch.reshape(img1,[-1,224,224,3])
input_imgs_r = torch.clamp(input_imgs_r,0,255).to(torch.float32)
input_imgs_r = (input_imgs_r - 127.5) / 128.0
input_imgs_r = input_imgs_r.permute(0,3,1,2)

# # Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path=<optional save path>)
# Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = model(img.unsqueeze(0))
img_embedding = model(input_imgs_r)

feature = np.array(img_embedding.detach().numpy()).flatten()
print(np.shape(feature))

# Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img.unsqueeze(0))

print('done')