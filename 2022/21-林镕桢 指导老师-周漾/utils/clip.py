import torch
import clip
from PIL import Image

model, preprocess = clip.load("ViT-B/32", "cuda")