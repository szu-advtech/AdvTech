import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from models.model_stages import BiSeNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    aux = False  # inference time not need aux_classifier
    classes = 19
    # weights_path = "./checkpoints/STDC1-Seg/model_maxmIOU50.pth"
    # weights_path = "./checkpoints/STDC1-Seg/model_maxmIOU75.pth"
    # weights_path = "./checkpoints/STDC2-Seg/model_maxmIOU50.pth"
    # weights_path = "./checkpoints/STDC2-Seg/model_maxmIOU75.pth"
    # weights_path = "./checkpoints/train_STDC1-Seg/pths/model_maxmIOU50.pth"
    # weights_path = "./checkpoints/train_STDC1-Seg/pths/model_maxmIOU75.pth"
    # weights_path = "./checkpoints/train_STDC2-Seg/pths/model_maxmIOU50.pth"
    weights_path = "./checkpoints/train_STDC2-Seg/pths/model_maxmIOU75.pth"
    img_path = "./test2.png"
    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = BiSeNet('STDCNet813', 19, scale=0.5, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
    model = BiSeNet('STDCNet1446',19, scale=0.5, 
    use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')#['model']
    # for k in list(weights_dict.keys()):
    #     if "aux" in k:
    #         del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([#transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output[0].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        mask.save("test4_75.png")
if __name__ == '__main__':
    main()
