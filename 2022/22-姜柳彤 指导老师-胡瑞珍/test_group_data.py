import os.path

import torch

import dataset
import material_decomposition_resnetUnet as mdr
import utils
import json
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from torchvision.utils import save_image
from unet import unet_model

def test(test_data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # When we run this line in sever, we should remove the map_location
    model_dict = torch.load('save_models/checkpoints_16.pkl')
    # model = mdr.res34_unet()
    model = unet_model.UNet(4)
    model.to(device)
    model.load_state_dict(model_dict['state_dict'])
    model.eval()

    test_save_path = ''
    with open('config.json') as file:
        data = json.load(file)
        test_save_path = data['test_img_save_path']
    
    amount = len(test_data_loader)
    print('The total amount of dirs:', amount)

    for t_steps, (t_imgs_batch, t_mask_batch, t_label_base_color, t_label_roughness,
                  t_label_metallic, t_label_normal) in enumerate(test_data_loader):

        t_imgs_batch = t_imgs_batch.to(device)
        t_mask_batch = t_mask_batch.to(device)

        t_label_base_color = t_label_base_color.to(device) * t_mask_batch
        t_label_roughness = t_label_roughness.to(device) * t_mask_batch
        t_label_metallic = t_label_metallic.to(device) * t_mask_batch
        t_label_normal = t_label_normal.to(device) * t_mask_batch

        t_x = torch.cat([t_imgs_batch, t_mask_batch], dim=1)

        t_y_pred = model(t_x)
        t_y_pred = t_y_pred * t_mask_batch

        # squeeze matrix, throw away the batch_size
        t_imgs_batch = torch.squeeze(t_imgs_batch)
        t_mask_batch = torch.squeeze(t_mask_batch)
        t_label_base_color = torch.squeeze(t_label_base_color)
        t_label_roughness = torch.squeeze(t_label_roughness)
        t_label_metallic = torch.squeeze(t_label_metallic)
        t_label_normal = torch.squeeze(t_label_normal)

        infer_base_color = t_y_pred[:, 0:3, :, :]
        infer_roughness = t_y_pred[:, 3:4, :, :]
        infer_metallic = t_y_pred[:, 4:5, :, :]
        infer_normal = t_y_pred[:, 5:8, :, :]
        infer_base_color = torch.squeeze(infer_base_color)
        infer_roughness = torch.squeeze(infer_roughness)
        infer_metallic = torch.squeeze(infer_metallic)
        infer_normal = torch.squeeze(infer_normal)

        # save_images
        dir = os.path.join(test_save_path, str(t_steps))
        if not os.path.exists(dir):
            os.mkdir(dir)

        save_image(t_imgs_batch, os.path.join(dir, 'source_img.png'))
        save_image(t_mask_batch, os.path.join(dir, 'source_mask.png'))
        save_image(t_label_base_color, os.path.join(dir, 'label_basic_color.png'))
        save_image(t_label_roughness, os.path.join(dir, 'label_roughness.png'))
        save_image(t_label_metallic, os.path.join(dir, 'label_metallic.png'))
        save_image(t_label_normal, os.path.join(dir, 'label_normal.png'))

        save_image(infer_base_color, os.path.join(dir, 'predict_base_color.png'))
        save_image(infer_roughness, os.path.join(dir, 'predict_roughness.png'))
        save_image(infer_metallic, os.path.join(dir, 'predict_metallic.png'))
        save_image(infer_normal, os.path.join(dir, 'predict_normal.png'))

        # if t_steps % 100 == 0:
        print('test:', str(t_steps), '/', str(amount))


if __name__ == "__main__":
    t_source_img, t_source_mask, t_label_base_color, t_label_metallic_roughness, t_label_normal = utils.get_path('test')
    t_package_data = dataset.arrange_datasets(t_source_img, t_source_mask, t_label_base_color, t_label_metallic_roughness, \
                                              t_label_normal)

    test_path = ''

    with open('config.json') as file:
        data = json.load(file)
        test_path = data['test_path']

    t_ABO_datasets = dataset.t_ABO_dataset(t_package_data, test_path)
    t_ABO_dataloader = torch.utils.data.DataLoader(t_ABO_datasets, batch_size = 1, shuffle=True)

    test(t_ABO_dataloader)
