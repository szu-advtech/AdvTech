#!/usr/bin/env python3

import learn2learn as l2l
import datasets

import datapre
from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Normalize)


def digits_tasksets(
    train_ways=10,
    train_samples=10,
    test_ways=10,
    test_samples=10,
    root='~/data',
    data_augmentation=None,
    **kwargs,
):
    """Tasksets for digits benchmarks."""
    if data_augmentation is None:
        train_data_transforms = None
        test_data_transforms = None
    else:
        raise('Invalid data_augmentation argument.')
    train_dataset = datasets.digits_color(
        root=root,
        mode='train',
        transform=train_data_transforms,
        download=True,
    )
    valid_dataset = datasets.digits_color(
        root=root,
        mode='valid',
        transform=test_data_transforms,
        download=True,
    )
    test_dataset = datasets.digits_color(
        root=root,
        mode='test',
        transform=test_data_transforms,
        download=True,
    )
    train_dataset = datapre.MetaDataset(train_dataset)
    valid_dataset = datapre.MetaDataset(valid_dataset)
    test_dataset = datapre.MetaDataset(test_dataset)
    train_transforms = [
        datapre.transforms.NTask(train_dataset),
        datapre.transforms.NWays(train_dataset, train_ways),
        datapre.transforms.KShots(train_dataset, train_samples),
        datapre.transforms.LoadData(train_dataset),
        datapre.transforms.RemapLabels(train_dataset),
        datapre.transforms.ConsecutiveLabels(train_dataset),
    ]
    valid_transforms = [
        datapre.transforms.NTask(valid_dataset),
        datapre.transforms.NWays(valid_dataset, train_ways),
        datapre.transforms.KShots(valid_dataset, train_samples),
        datapre.transforms.LoadData(valid_dataset),
        datapre.transforms.ConsecutiveLabels(valid_dataset),
        datapre.transforms.RemapLabels(valid_dataset),
    ]
    test_transforms = [
        datapre.transforms.NTask(test_dataset),
        datapre.transforms.NWays(test_dataset, test_ways),
        datapre.transforms.KShots(test_dataset, test_samples),
        datapre.transforms.LoadData(test_dataset),
        # datapre.transforms.RemapLabels(test_dataset),
        # datapre.transforms.ConsecutiveLabels(test_dataset),
    ]
    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms
