import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from hrnet import HighResolutionNet

from torchvision.transforms import functional as F
import PIL.Image as Image

def predict(k):
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    os.makedirs("./predict/pth" + f"{k}")
    weights_path = "./path/"+ f"{k}"+".pth"
    for i in range(1, 31):
        img_path = "../PoreGroundTruthSampleimage/" + f'{i}' + ".bmp"


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
        minutiae = outputs.cpu().detach().numpy()

        
        minutiae = minutiae * 255
        
        minutiae_im = Image.fromarray(minutiae)

        minutiae_im = minutiae_im.convert("L")
        
        
        minutiae_path = './predict/pth' + f"{k}"+ "/"+ f"{i}" + "minutiae.bmp"

        minutiae_im.save(minutiae_path)

if __name__ == '__main__':
    for i in range(21,50) :
        predict(i)
