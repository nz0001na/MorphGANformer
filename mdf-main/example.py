import torch as pt
import torch.optim as optim
import imageio
from torch.autograd import Variable

from pytorch_version.mdfloss import MDFLoss

# Set parameters
cuda_available = False
epochs = 25
application = 'Denoising'
image_path = './img/i10.png'

if application == 'SISR':
    path_disc = "./weights/Ds_SISR.pth"
elif application == 'Denoising':
    path_disc = "./weights/Ds_Denoising.pth"
elif application == 'JPEG':
    path_disc = "./weights/Ds_JPEG.pth"

# Read reference images
imgr = imageio.imread(image_path)
imgr = pt.from_numpy(imageio.core.asarray(imgr / 255.0))
imgr = imgr.type(dtype=pt.float64)
imgr = imgr.permute(2, 0, 1)
imgr = imgr.unsqueeze(0).type(pt.FloatTensor)

# Create a noisy image
imgd = pt.rand(imgr.size())

if cuda_available:
    imgr = imgr.cuda()
    imgd = imgd.cuda()

# Convert images to variables to support gradients
imgrb = Variable(imgr, requires_grad=False)
imgdb = Variable(imgd, requires_grad=True)

optimizer = optim.Adam([imgdb], lr=0.1)

# Initialise the loss
criterion = MDFLoss(path_disc, cuda_available=cuda_available)

# Iterate over the epochs optimizing for the noisy image
for ii in range(0, epochs):
    optimizer.zero_grad()
    loss = criterion(imgrb, imgdb)
    print("Epoch: ", ii, " loss: ", loss.item())
    loss.backward()
    optimizer.step()