from sklearn.cluster import KMeans
import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

from dataset.coco_handler import CocoHandler, CocoDataset, get_seg
from models import Model

import numpy as np


def eval(rank, model, is_encode, is_train, k_means=None):
    # supercategory = "sports"
    supercategory = ""
    dataset_path = "E:/Datasets/cocostuff"
    config_path = "E:/Datasets/cocostuff/instances_val2017.json"
    eval_handler = CocoHandler(dataset_path, config_path, supercategory, False)
    if k_means is not None:
        feature_list = np.load("./eval_feature_all.npy")
        data_loader = data.DataLoader(CocoDataset(*eval_handler.get_all_list(), feature_list, "eval"))
    else:
        data_loader = data.DataLoader(CocoDataset(*eval_handler.get_all_list(), "eval"))

    to_img = transforms.ToPILImage()

    model.eval()
    for i, (real_img, label, inst, feature) in enumerate(data_loader):
        
        real_img, label, inst = real_img.to(rank), label.to(rank), inst.to(rank)
        seg = get_seg(rank, label, inst)

        feature_index = k_means.predict(feature)
        feature = k_means.cluster_centers_[feature_index]
        feature = torch.tensor(feature).float().to(rank)

        if is_train:
            if is_encode:
                mu, log_s = model.module.encode_img(real_img)
                z = model.module.get_nosie(mu, log_s)
            else:
                # z = torch.randn([1, 256]).to(rank)
                size = model.module.net_g.start_size
                z = torch.nn.functional.interpolate(seg, (size, size))
            
            gen_img = model.module.gen_img(z, seg, feature)
        else:
            if is_encode:
                mu, log_s = model.encode_img(real_img)
                z = model.get_nosie(mu, log_s)
            else:
                # z = torch.randn([1, 256]).to(rank)
                size = model.net_g.start_size
                z = torch.nn.functional.interpolate(seg, (size, size))
            gen_img = model.gen_img(z, seg, feature)

        real_img = ((real_img+1)/2).clamp_(0, 1)
        gen_img = ((gen_img+1)/2).clamp_(0, 1)
        to_img(torch.squeeze(real_img, 0).cpu()).save(f"./eval_imgs/real_{i}.png")
        to_img(torch.squeeze(gen_img, 0).cpu()).save(f"./eval_imgs/gen_{i}.png")
        

        # if i>=100:
        #     break

    if is_train:
        model.train()


if __name__ == "__main__":
    rank=0
    model = Model(False)
    model.load("./trained_models/50")
    model.to(rank)

    feature_list = np.load("./clip_feature_all.npy")
    k_means = KMeans(128)
    k_means.fit(feature_list)
    eval(rank, model, False, False, k_means)
    # eval(rank, model, False, False)
    