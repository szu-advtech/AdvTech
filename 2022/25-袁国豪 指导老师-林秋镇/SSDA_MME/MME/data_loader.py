import torch
from torchvision import datasets,transforms
import os
from data_list import Imagelists_VISDA


def return_dataset(args):
    base_path = '../txt/%s' % args.dataset
    root = '../%s' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(227),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(227),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                          transform=data_transforms['val'])
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])

    bs = 8

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=0, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=0,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=0,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=0,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=0,
                                    shuffle=True, drop_last=True)
    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test




