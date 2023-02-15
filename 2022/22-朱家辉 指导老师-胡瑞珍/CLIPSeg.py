from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

class ClipSeg:
    def __init__(self, device, save_path=None):
        self.num_classes = 7

        self.processor = CLIPSegProcessor.from_pretrained("./clipseg/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("./clipseg/clipseg-rd64-refined")
        
        # self.prompts = ["background", "red", "orange", "yellow", "green", "blue", "indigo", "violet", "human", "people"] # human, people can be delete?

        self.device = device
        self.model.to(self.device)

        if not save_path:
            self.save_path = "./lwrf_results/"
        else:
            self.save_path = save_path

    def segment(self, color_img, prompts):
        detected_thold = 50
        self.image = color_img
        orig_size = self.image.shape[:2][::-1]

        inputs = self.processor(text=prompts, images=[self.image] * len(prompts), padding="max_length", return_tensors="pt")
        # inputs = self.processor(text=self.prompts, images=[self.image] * len(self.prompts), padding="max_length", return_tensors="pt")

        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = outputs.logits # shape=[10, 352,352] tensor
        # segm = preds.cpu().detach().numpy()# segm.shape=[352,352,10]
        segm = preds.cpu().detach().numpy().transpose(1, 2, 0) # segm.shape=[352,352,10]
        segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
        segm = segm.argmax(axis=2).astype(np.uint8)
        # segm = segm.astype(np.uint8)

        class_ids = []
        labels = []
        masks = np.zeros((segm.shape[0], segm.shape[1], self.num_classes))
        count = 0
        for class_id in range(1, self.num_classes + 1):
            temp = np.zeros((segm.shape[0], segm.shape[1]))
            temp[segm == class_id] = 1
            if np.sum(temp) > detected_thold:
                masks[:, :, count] = temp
                class_ids.append(class_id)
                # labels.append(self.prompts[0])
                labels.append(prompts[class_id])
                count += 1

        self.res = {'class_ids': class_ids, 'labels': labels, 'masks': masks}
        print("using CLIPSeg module...")
        return self.res

# img = cv2.imread("./clipseg/0.png")
# clipseg = ClipSeg(device="cuda")
# clipseg.segment(img)
