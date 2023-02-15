import torch
import torch.nn as nn

import copy
import os

from models import net
from rigid_alignment import *


vgg_normalised_conv5_1 = nn.Sequential(
    nn.Conv2d(3,3,(1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3,64,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,64,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,128,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,128,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
)

feature_invertor_conv5_1 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,256,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,128,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,128,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,64,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,64,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,3,(3, 3)),
)

class Encoder(nn.Module):
    def __init__(self, depth, pretrained_path):
        super(Encoder, self).__init__()
        self.depth = depth
        if depth == 1:
            self.model = vgg_normalised_conv5_1[:4]
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'vgg_normalised_conv1_1.pth')))
        elif depth == 2:
            self.model = vgg_normalised_conv5_1[:11]
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'vgg_normalised_conv2_1.pth')))
        elif depth == 3:
            self.model = vgg_normalised_conv5_1[:18]
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'vgg_normalised_conv3_1.pth')))
        elif depth == 4:
            self.model = vgg_normalised_conv5_1[:31]
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'vgg_normalised_conv4_1.pth')))

    def forward(self, x):
        out = self.model(x)
        return out

class Decoder(nn.Module):
    def __init__(self, depth, pretrained_path):
        super(Decoder, self).__init__()
        self.depth = depth
        if depth == 1:
            self.model = nn.Sequential(*copy.deepcopy(list(feature_invertor_conv5_1.children())[-2:]))
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'feature_invertor_conv1_1.pth')))
        elif depth == 2:
            self.model = nn.Sequential(*copy.deepcopy(list(feature_invertor_conv5_1.children())[-9:]))
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'feature_invertor_conv2_1.pth')))
        elif depth == 3:
            self.model = nn.Sequential(*copy.deepcopy(list(feature_invertor_conv5_1.children())[-16:]))
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'feature_invertor_conv3_1.pth')))
        elif depth == 4:
            self.model = nn.Sequential(*copy.deepcopy(list(feature_invertor_conv5_1.children())[-29:]))
            self.model.load_state_dict(torch.load(os.path.join(pretrained_path, 'feature_invertor_conv4_1.pth')))

    def forward(self, x):
        out = self.model(x)
        return out

class MultiLevelNetwork(nn.Module):
    def __init__(self, device, pretrained_path, alpha=0.5, beta=0):
        super(MultiLevelNetwork, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        
        self.e1 = Encoder(1, pretrained_path)
        self.e2 = Encoder(2, pretrained_path)
        self.e3 = Encoder(3, pretrained_path)
        self.e4 = Encoder(4, pretrained_path)
        self.encoders = [ self.e4, self.e3, self.e2, self.e1]
        
        self.d1 = Decoder(1, pretrained_path)
        self.d2 = Decoder(2, pretrained_path)
        self.d3 = Decoder(3, pretrained_path)
        self.d4 = Decoder(4, pretrained_path)
        self.decoders = [self.d4, self.d3, self.d2, self.d1]

    def forward(self, content_img, style_img, additional_style_flag=False, style_img1=None):
        if additional_style_flag:
            content_img = stylize_ra(0, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha,
                                     beta=self.beta, style1=style_img1)
            content_img = stylize_mm(1, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha,
                                     beta=self.beta, style1=style_img1)
            content_img = stylize_mm(2, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha,
                                     beta=self.beta, style1=style_img1)
            content_img = stylize_mm(3, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha,
                                     beta=self.beta, style1=style_img1)
        else:
            content_img = stylize_ra(0, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha)
            content_img = stylize_mm(1, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha)
            content_img = stylize_mm(2, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha)
            content_img = stylize_mm(3, content_img, style_img, self.encoders, self.decoders, self.device, self.alpha)
        return content_img