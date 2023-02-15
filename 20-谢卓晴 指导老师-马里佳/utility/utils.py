#!/usr/bin/env python3
# coding=utf-8

import torch
from PIL import Image


def create_padding_mask(batch_size, total_length, lengths, device):
    mask = torch.arange(total_length, device=device).expand(batch_size, total_length)
    mask = mask >= lengths.unsqueeze(1)  # shape: (B, T)
    return mask


def resize_to_square(image, target_size: int, background_color="white"):
    width, height = image.size
    if width / 2 > height:
        result = Image.new(image.mode, (width, width // 2), background_color)
        result.paste(image, (0, (width // 2 - height) // 2))
        image = result
    elif height * 2 > width:
        result = Image.new(image.mode, (height * 2, height), background_color)
        result.paste(image, ((height * 2 - width) // 2, 0))
        image = result

    image = image.resize([target_size * 2, target_size], resample=Image.BICUBIC)
    return image
