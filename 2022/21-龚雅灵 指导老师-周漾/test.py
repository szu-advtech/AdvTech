import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import torch.nn.functional as F
import os

from PIL import Image
import copy
import argparse

from models.net import MultiLevelNetwork


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='inputs/contents/actress.jpg')
    parser.add_argument('--style', type=str, default='inputs/styles/asheville.jpg')
    parser.add_argument('--output', type=str, default='outputs/result.jpg')
    parser.add_argument('--pretrained_path', type=str, default='pretrained_models')
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--additional_style_flag', type=bool, default=False)
    parser.add_argument('--style1', type=str, default='inputs/styles/sketch.png')
    parser.add_argument('--beta', type=float, default=0)
    args = parser.parse_args()
    
    trans = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    c = Image.open(args.content).convert('RGB')
    s = Image.open(args.style).convert('RGB')
    c = c.resize((400, int(400*c.size[1]/c.size[0])))
    s = s.resize((400, int(400*s.size[1]/s.size[0])))
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)
    
    if args.additional_style_flag:
        s1 = Image.open(args.style1).convert('RGB')
        s1 = s1.resize((400, int(400*s1.size[1]/s1.size[0])))
        s1_tensor = trans(s1).unsqueeze(0).to(device)
        model = MultiLevelNetwork(device, args.pretrained_path, args.alpha, args.beta).to('cuda')
        out = model(c_tensor, s_tensor, args.additional_style_flag, s1_tensor)
    else:
        model = MultiLevelNetwork(device, args.pretrained_path, args.alpha).to('cuda')
        out = model(c_tensor, s_tensor)
    
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    save_image(out, args.output)
