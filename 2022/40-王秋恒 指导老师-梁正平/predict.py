import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from hrnet import HighResolutionNet

from torchvision.transforms import functional as F
import PIL.Image as Image

def predict():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    weights_path = "./path/30.pth"
    for i in range(1, 31):
        img_path = "./PoreGroundTruthSampleimage/" + f'{i}' + ".bmp"


        image = Image.open(img_path)
        image = F.to_tensor(image)

        img_tensor = torch.unsqueeze(image, dim=0)

        model = HighResolutionNet()
        weights = torch.load(weights_path, map_location=device)
        weights = weights["model"]
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

        outputs = model(img_tensor.to(device))
        outputs = torch.squeeze(outputs)
        # detach函数使tensor变成无梯度的
        pores = outputs[0].cpu().detach().numpy()
        minutiae = outputs[1].cpu().detach().numpy()

        pores = pores * 255
        minutiae = minutiae * 255
        # fromarray实现从矩阵到图像的转变
        pore_im = Image.fromarray(pores)
        minutiae_im = Image.fromarray(minutiae)

        pore_im = pore_im.convert("L")
        minutiae_im = minutiae_im.convert("L")

        pore_path = './predicted/' + f"{i}" + "pore.bmp"
        minutiae_path = './predicted/' + f"{i}" + "minutiae.bmp"

        pore_im.save(pore_path)
        minutiae_im.save(minutiae_path)

if __name__ == '__main__':
    predict()
