import os
import random
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from skimage.draw import polygon
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


class CocoHandler:
    def __init__(self, dataset_path, config_path, supercategory, is_train=True) -> None:
        self.coco = COCO(config_path)
        self.status = "train" if is_train else "val"
        self.img_path = os.path.join(dataset_path, f"{self.status}_img")
        self.label_path = os.path.join(dataset_path, f"{self.status}_label")
        self.inst_path = os.path.join(dataset_path, f"{self.status}_inst")
        self.img_ids = self._get_img_ids(supercategory)
        self.img_list, self.label_list, self.inst_list = self._gen_paths()

    def _get_img_ids(self, supercategory):
        img_ids = set()
        coco = self.coco
        if supercategory == "":
            cat_ids = coco.getCatIds()
        else:
            cat_ids = coco.getCatIds(supNms=[supercategory])
        for id in cat_ids:
            img_list = coco.getImgIds(catIds=[id])
            for img in img_list:
                img_ids.add(img)
        img_ids = sorted(list(img_ids))
        return img_ids

    def _gen_paths(self):
        img_list = []
        label_list = []
        inst_list = []
        img_ids = self.img_ids
        for id in img_ids:
            name = self.coco.loadImgs(id)[0]["file_name"].split(".")[0]
            img_list.append(os.path.join(self.img_path, f"{name}.jpg"))
            label_list.append(os.path.join(self.label_path, f"{name}.png"))
            inst_list.append(os.path.join(self.inst_path, f"{name}.png"))
        return img_list, label_list, inst_list

    def gen_inst_maps(self):
        img_ids = self.img_ids
        for id in img_ids:
            name = self.coco.loadImgs(id)[0]["file_name"].split(".")[0]
            cur_label_path = os.path.join(self.label_path, f"{name}.png")
            map = io.imread(cur_label_path, as_gray=True)
            ann_id = self.coco.getAnnIds(imgIds=id)
            anns = self.coco.loadAnns(ann_id)
            counter = 0
            for ann in anns:
                if type(ann["segmentation"]) == list and "segmentation" in ann:
                    for seg in ann["segmentation"]:
                        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                        xs, ys = polygon(poly[:, 1] - 1, poly[:, 0] - 1)
                        map[xs, ys] = counter
                counter += 1
            io.imsave(os.path.join(self.inst_path, f"{name}.png"), map)

    def gen_clip(self, device="cpu"):
        import clip
        img_ids = self.img_ids
        clip_arr = []
        model, preprocess = clip.load("ViT-B/32", device=device)
        counter = 0
        for cur_img_path in self.img_list:
            img = Image.open(cur_img_path)
            img = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feature = model.encode_image(img)
            clip_arr.append(img_feature.cpu().numpy()[0])
            counter += 1
            print(f"{counter}/{len(img_ids)}", end="\r")

        clip_features = np.array(clip_arr)
        np.save("./clip_feature.npy", clip_features)


    def get_cur_img_ids(self):
        return self.img_ids

    def get_all_list(self):
        return self.img_list, self.label_list, self.inst_list


class CocoDataset(data.Dataset):
    def __init__(self, img_list, label_list, inst_list, feature_list=None, status="train") -> None:
        super().__init__()
        self.img_list = img_list
        self.label_list = label_list
        self.inst_list = inst_list
        self.feature_list = feature_list
        self.is_train = True if status=="train" else False
        self.resize_size = 284 if status=="train" else 256
        self.label_nc = 182
        self.transform_list = [
            transforms.Resize((self.resize_size, self.resize_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            ]
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]
        inst_path = self.inst_list[index]

        transform_list = self.transform_list.copy()
        if self.is_train:
            x = random.randint(0, np.maximum(0, 284 - 256))
            y = random.randint(0, np.maximum(0, 284 - 256))
            transform_list.insert(-1, transforms.Lambda(lambda img: img.crop((x, y, x+256, y+256))))
            if random.random() > 0.5:
                transform_list.insert(-1, transforms.Lambda(lambda img: transforms.functional.vflip(img)))
        transform = transforms.Compose(transform_list)

        img = Image.open(img_path).convert("RGB")
        img = self.normalize(transform(img))
        label = Image.open(label_path)
        label = (transform(label)*255).long()
        label[label == 255] = self.label_nc
        inst = Image.open(inst_path)
        inst = (transform(inst)*255).long()

        if self.feature_list is not None:
            feature = torch.tensor(self.feature_list[index])
            return img, label, inst, feature
        else:
            return img, label, inst


def get_edge(rank, inst):
    edge = torch.zeros_like(inst).to(rank)
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (inst[:, :, :, 1:] != inst[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (inst[:, :, :, 1:] != inst[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (inst[:, :, 1:, :] != inst[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (inst[:, :, 1:, :] != inst[:, :, :-1, :])
    return edge.float()

def get_seg(rank, label, inst=None):
    seg = torch.zeros([label.shape[0], 183, *label.shape[2:]]).to(rank)
    seg = seg.scatter_(1, label, 1.0)
    if inst != None:
        edge = get_edge(rank, inst)
        seg = torch.cat([seg, edge], dim=1)
    
    return seg
        

if __name__ == "__main__":
    # supercategory = "sports"
    supercategory = ""
    dataset_path = "E:/Datasets/cocostuff"
    config_path = "E:/Datasets/cocostuff/instances_val2017.json"
    train_handler = CocoHandler(dataset_path, config_path, supercategory, False)
    # train_handler.gen_inst_maps()
    train_handler.gen_clip("cuda:0")
    # clip_features = np.load("./clip_feature_all.npy")
    # clip_features = np.load("./train_feature.npy")
    # print(clip_features.shape)
    # from sklearn.cluster import KMeans
    # k_means = KMeans(10)
    # k_means.fit(clip_features)
    # x = torch.randn(1,512).half().numpy()
    # index = k_means.predict(x)
    # print(k_means.cluster_centers_.shape)


    # train_loader = data.DataLoader(CocoDataset(*train_handler.get_all_list()))
    # for i, tensor in enumerate(train_loader):
    #     # edge = get_edge(tensor[2])
    #     print(i, tensor[0].shape, end='\r')