from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def get_image():
    pass

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.functional.to_pil_image(image)
    return image


if __name__ == "__main__":
    p = create_config('configs/env.yml', 'configs/scan/scan_stl10.yml', 20, 'checkpoint.tar', '')
    # Transforms
    strong_transforms = get_train_transformations(p)
    weak_transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, weak_transforms, split='train')
    # indices = np.load('ouptut/baseline/stl-10')
    # indices = indices.astype(np.int32)
    # for i in range(100, 300):
    #     for j, idx in enumerate(indices[i]):
    #         img = np.array(train_dataset.get_image(idx)).astype(np.uint8)
    #         img = Image.fromarray(img)
    #         img.save('../results/figures/stl10/standard/{}_{}.png'.format(i,j))

    for i in range(1000, 1002):
            img = np.array(train_dataset.get_image(i)).astype(np.uint8)
            img = Image.fromarray(img)
            # img = weak_transforms(img)
            # toPIL = transforms.ToPILImage()
            # img = toPIL(img)
            img.save('../results/figures/strong/{}.png'.format(i))
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
            plt.show()


    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # for idx in indices:
    #     img = np.array(train_dataset.get_image(idx)).astype(np.uint8)
    #     img = Image.fromarray(img)
    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(img)
    #     plt.show()






    # dog = 5
    # targets = np.array(train_dataset.targets)
    # index_all = np.where(targets==dog)[0]
    # import random
    # inds = random.sample(index_all.tolist(), 20)
    #
    # for i in range(len(inds)):
    #     image = train_dataset.__getitem__(i)['image']
    #     image_weak_tensor = weak_transforms(image)
    #     image_weak = tensor_to_PIL(image_weak_tensor)
    #
    #     image_strong_tensor1 = strong_transforms(image)
    #     image_strong_tensor2 = strong_transforms(image)
    #     image_strong_tensor3 = strong_transforms(image)
    #     image_strong_tensor4 = strong_transforms(image)
    #
    #     image_strong1 = tensor_to_PIL(image_strong_tensor1)
    #     image_strong2 = tensor_to_PIL(image_strong_tensor2)
    #     image_strong3 = tensor_to_PIL(image_strong_tensor3)
    #     image_strong4 = tensor_to_PIL(image_strong_tensor4)
    #     image.save('../results/figures/anchor/dog{}_1.png'.format(i+1))
    #     image_weak.save('../results/figures/anchor/dog{}_w1.png'.format(i+1))
    #     image_strong1.save('../results/figures/anchor/dog{}_s1.png'.format(i+1))
    #     image_strong2.save('../results/figures/anchor/dog{}_s2.png'.format(i+1))
    #     image_strong3.save('../results/figures/anchor/dog{}_s3.png'.format(i+1))
    #     image_strong4.save('../results/figures/anchor/dog{}_s4.png'.format(i+1))

# import torch
# from data.augment import Cutout
# i = 4
# image = train_dataset.__getitem__(i)['image']
# horizon_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.RandomHorizontalFlip()
#         ])
# cutout_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.RandomHorizontalFlip(),
#             Cutout(
#             n_holes=p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
#             length=p['augmentation_kwargs']['cutout_kwargs']['length'],
#             random=p['augmentation_kwargs']['cutout_kwargs']['random'])
#         ])
# horizon_tensor = horizon_transforms(image)
# horizon_image = tensor_to_PIL(horizon_tensor)
#
# cutout_tensor = cutout_transforms(image)
# cutout_image = tensor_to_PIL(cutout_tensor)
# horizon_image.save('../results/figures/ship{}_h.png'.format(i))
# cutout_image.save('../results/figures/ship{}_c.png'.format(i))


def visualize_indices(indices, dataset, hungarian_match):
    import matplotlib.pyplot as plt
    import numpy as np
    # np.save('/stc615/cst/code/results/figures/topk/indices.npy', indices)
    for idx in indices:
        img = np.array(dataset.get_image(idx)).astype(np.uint8)
        img = Image.fromarray(img)
        img.save('/stc615/cst/code/results/figures/topk/idx_{}.png'.format(idx))
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.show()