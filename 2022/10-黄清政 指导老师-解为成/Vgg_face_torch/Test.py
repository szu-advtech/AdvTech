import torch
import torchfile
from torchvision import models
from torchsummary import summary




x = torchfile.load('VGG_FACE.t7')
print(type(x))