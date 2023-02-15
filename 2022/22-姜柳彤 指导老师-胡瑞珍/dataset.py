import os.path
import torch

from torch.utils import data
from torchvision.transforms import transforms
from PIL import Image
from matplotlib import pyplot as plt


# arrange datasets as the form I want
def arrange_datasets(source_img, source_mask, label_base_color, label_metallic_roughness, label_normal):
    total_path = []
    for i in range(0, len(source_img)):
        total_path.append([source_img[i], source_mask[i],
                          label_base_color[i], label_metallic_roughness[i], label_normal[i]])
    return total_path


# In order to create DataLoader
class ABO_dataset(data.Dataset):
    def __init__(self, imgs_path, root_dir):
        self.imgs_path = imgs_path
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        src_img = Image.open(os.path.join(self.root_dir, img_path[0]))
        src_mask = Image.open(os.path.join(self.root_dir, img_path[1]))
        label_base_color = Image.open(os.path.join(self.root_dir, img_path[2]))
        label_metallic_roughness = Image.open(os.path.join(self.root_dir, img_path[3]))
        label_normal = Image.open(os.path.join(self.root_dir, img_path[4]))

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        src_img = self.transform(src_img)
        torch.random.manual_seed(seed)
        mask = self.transform(src_mask)
        torch.random.manual_seed(seed)
        label_base_color = self.transform(label_base_color)
        torch.random.manual_seed(seed)
        label_metallic_roughness = self.transform(label_metallic_roughness).chunk(3)
        label_roughness = label_metallic_roughness[1]
        label_metallic = label_metallic_roughness[2]
        torch.random.manual_seed(seed)
        label_normal = self.transform(label_normal)
        return src_img, mask, label_base_color, label_roughness, label_metallic, label_normal

    def __len__(self):
        return len(self.imgs_path)

class t_ABO_dataset(data.Dataset):
    def __init__(self, imgs_path, root_dir):
        self.imgs_path = imgs_path
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            # transforms.RandomCrop(256),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        src_img = Image.open(os.path.join(self.root_dir, img_path[0]))
        src_mask = Image.open(os.path.join(self.root_dir, img_path[1]))
        label_base_color = Image.open(os.path.join(self.root_dir, img_path[2]))
        label_metallic_roughness = Image.open(os.path.join(self.root_dir, img_path[3]))
        label_normal = Image.open(os.path.join(self.root_dir, img_path[4]))

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        src_img = self.transform(src_img)
        torch.random.manual_seed(seed)
        mask = self.transform(src_mask)
        torch.random.manual_seed(seed)
        label_base_color = self.transform(label_base_color)
        torch.random.manual_seed(seed)
        label_metallic_roughness = self.transform(label_metallic_roughness).chunk(3)
        label_roughness = label_metallic_roughness[1]
        label_metallic = label_metallic_roughness[2]
        torch.random.manual_seed(seed)
        label_normal = self.transform(label_normal)
        return src_img, mask, label_base_color, label_roughness, label_metallic, label_normal

    def __len__(self):
        return len(self.imgs_path)
