import os
import torch
import numpy as np
import cv2
from net import resnet50_amr

if __name__ == "__main__":
    # model = resnet50_amr.Net()
    # pretrain_dict = torch.load("/data2/xiepuxuan/code/AMR/res50_amr.pth")
    # new_dict = {}
    # for k, v in pretrain_dict.items():
    #     if "resnet50" in k:
    #         new_dict[k.replace("resnet50", "resnet50_spotlight")] = v
    # model_dict = model.state_dict()
    # model_dict.update(new_dict)
    # model.load_state_dict(model_dict)
    # device = torch.device("cuda")
    # model.to(device)

    root = "/data2/xiepuxuan/code/AMR/outputs/amr_voc2012_3/cam_outputs/"
    for cam_dict in os.listdir(r"/data2/xiepuxuan/code/AMR/outputs/amr_voc2012_3/cam_outputs/"):
        cam = np.load(os.path.join(root, cam_dict), allow_pickle=True).item()
        cams = cam['high_res'][0]

        img = cv2.imread(os.path.join("/data2/xiepuxuan/dataset/VOC2012/JPEGImages/", cam_dict[:-4] + ".jpg"))
        cams = cv2.resize(cams, (img.shape[1], img.shape[0]))

        cams = np.uint8(255 * cams)
        cams = cv2.applyColorMap(cams, cv2.COLORMAP_JET)
        cam_img = cv2.addWeighted(img, 1, cams, 0.5, 0)
        cv2.imwrite("/data2/xiepuxuan/code/AMR/outputs/amr_voc2012_3/ir_label_outputs/"+cam_dict[:-4] + ".jpg", cam_img)

