#!/usr/bin/env python3

import math
import random

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from .rand_augment import rand_augment_transform
from .random_erasing import RandomErasing

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR

class RandomResizedCropAndInterpolation:

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [_pil_interpolation_to_str[x] for x in self.interpolation]
            )
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


def transforms_imagenet_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="random",
    use_prefetcher=False,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    separate=False,
):
    if isinstance(img_size, tuple):
        img_size = img_size[-2:]
    else:
        img_size = img_size

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            raise NotImplementedError("Augmix not implemented")
        else:
            raise NotImplementedError("Auto aug not implemented")
    elif color_jitter is not None:

        if isinstance(color_jitter, (list, tuple)):

            assert len(color_jitter) in (3, 4)
        else:
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    final_tfl += [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
                cube=False,
            )
        )

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
