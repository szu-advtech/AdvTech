import os.path

import numpy as np
import torch
import matplotlib.pyplot as plt

import dataset
import material_decomposition_resnetUnet as mdr
import material_decomposition_torchresnetUnet as mdt
import utils
import json
from unet.unet_model import UNet
from PIL import Image
from torchvision.transforms import transforms

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# ----------------------------------------------------------------------------------------------------------------------
# model_dict = torch.load('save_models/torchresnet_checkpoints_15.pkl', map_location='cpu')
model_dict = torch.load('save_models/checkpoints_16.pkl', map_location='cpu')
# model_dict = torch.load('save_models/unet_checkpoints_10.pkl', map_location='cpu')
model = mdr.res34_unet()
# model = mdt.res34_unet()
# model = UNet(4)
model.load_state_dict(model_dict['state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.RandomCrop(size=256),
    transforms.ToTensor()
])

render_img = Image.open(os.path.join('D:/dataset/abo-benchmark-material/test/B07B4GVN1Q/render/0/render_1.jpg'))
mask_img = Image.open(os.path.join('D:/dataset/abo-benchmark-material/test/B07B4GVN1Q/segmentation/segmentation_1.jpg'))
l_base_color = Image.open(os.path.join('D:/dataset/abo-benchmark-material/test/B07B4GVN1Q/base_color/base_color_1.jpg'))
l_metallic_roughness = Image.open(os.path.join('D:/dataset/abo-benchmark-material/test/B07B4GVN1Q/metallic_roughness'
                                               '/metallic_roughness_1.jpg'))
l_normal = Image.open(os.path.join('D:/dataset/abo-benchmark-material/test/B07B4GVN1Q/normal/normal_1.png'))

seed = torch.random.seed()

torch.set_printoptions(threshold=np.inf)
# input change
torch.random.manual_seed(seed)
render_img = transform(render_img)
torch.random.manual_seed(seed)
mask_img = transform(mask_img)

# label change
torch.random.manual_seed(seed)
l_base_color = transform(l_base_color)
torch.random.manual_seed(seed)
l_normal = transform(l_normal)
torch.random.manual_seed(seed)
l_metallic_roughness = transform(l_metallic_roughness).chunk(3)
l_roughness = l_metallic_roughness[1]
l_metallic = l_metallic_roughness[2]

# make input
x = torch.cat([render_img, mask_img])
x = torch.unsqueeze(x, 0)
# enter network
y = model(x)
# get the results
# print(y.size()) # -> torch.size([1, 8, 256, 256])
y = y.squeeze(dim=0)
infer_base_color = y[0:3, :, :]
infer_roughness = y[3:4, :, :]
infer_metallic = y[4:5, :, :]
infer_normal = y[5:8, :, :]

toPIL = transforms.ToPILImage()
l_base_color = toPIL(l_base_color)
l_roughness = toPIL(l_roughness)
l_metallic = toPIL(l_metallic)
l_normal = toPIL(l_normal)
infer_base_color = toPIL(infer_base_color * mask_img)
infer_roughness = toPIL(infer_roughness * mask_img)
infer_metallic = toPIL(infer_metallic * mask_img)
infer_normal = toPIL(infer_normal * mask_img)

fig = plt.figure()
axes = fig.subplots(2, 4)
axes[0, 0].imshow(l_base_color)
axes[0, 1].imshow(l_roughness)
axes[0, 2].imshow(l_metallic)
axes[0, 3].imshow(l_normal)
axes[1, 0].imshow(infer_base_color)
axes[1, 1].imshow(infer_roughness)
axes[1, 2].imshow(infer_metallic)
axes[1, 3].imshow(infer_normal)
plt.show()
# --------------------------------------------------------------------------------------------------------------------


