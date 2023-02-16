import cv2
from torchvision.utils import save_image
from utils import *

def save_img(image, label, out):
    _image = image[0]
    _label = label[0]
    _out = out[0]
    save_image(_out, f'{save_img_path}/out_{i}.tif')
    #tmp3 = readTif(f'{save_img_path}/out_{i}.tif')
    tmp3 = cv2.imread(f'{save_img_path}/out_{i}.tif')
    tmp1 = cv2.cvtColor(tmp3, cv2.COLOR_BGR2GRAY)
    tmp3 = cv2.cvtColor(tmp1, cv2.COLOR_GRAY2RGB)
    tmp3 = tmp3.swapaxes(1, 2)
    tmp3 = tmp3.swapaxes(0, 1)
    tmp3 = torch.tensor(tmp3)
    img = torch.stack([_image, _label, tmp3], dim=0)
    save_image(img, f'{save_img_path}/{i}.tif')
    #return torch.tensor(tmp1), tmp3
