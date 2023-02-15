import torch
import torch.nn as nn


import dataset.dataset as Dataset
import dataset

import time
import torch
from torchvision.utils import save_image
from trainer.trainer import Trainer
import utils.util as util

def visualized_data(data_i):
    label = data_i['label']
    image = data_i['image']
    label_ref = data_i['label_ref']
    ref = data_i['ref']
    util.tensor_to_RGB(label,'./label.jpg')
    util.tensor_to_RGB(image,'./image.jpg')
    util.tensor_to_RGB(label_ref,'./label_ref.jpg')
    util.tensor_to_RGB(ref,'./ref.jpg')
    self_ref = data_i['self_ref']
    #print("self_ref = " , self_ref)
    pass
def visualized_warp(warp_img):
    util.tensor_to_RGB(warp_img[0],"./warp_0.jpg")
    util.tensor_to_RGB(warp_img[1],"./warp_1.jpg")
    util.tensor_to_RGB(warp_img[2],"./warp_2.jpg")
    util.tensor_to_RGB(warp_img[3],"./warp_3.jpg")

if __name__ == "__main__":
    dataloader = dataset.create_dataloader()
    len_dataloader = len(dataloader)

    trainer = Trainer(resume_epoch=6,continue_train=True)
    epoch = 6
    iter = 0
    D_per_G = 5
    
    for epoches in range(20):
        start = time.time()
        if epoch >= 5:
            trainer.update_learning_rate(epoch=epoch)
        for i,data_i in enumerate(dataloader,start=epoch):
            trainer.run_DomainAlin_one_step(data_i)
            now = time.time()
            util.print_latest_losses(epoch=epoch,i=i,num=len(dataloader),errors=trainer.c_losses,t = now - start)
            
            visualized_data(data_i)
            util.tensor_to_RGB(trainer.get_latest_generated_corr(),'./out.jpg')
            util.tensor_to_RGB(trainer.corr_out['warp_cycle'],'./warp_cycle.jpg')
            warp_tensor = trainer.corr_out['warp_out']
            visualized_warp(warp_tensor)
            
            iter += 1
            if iter == 100:
                trainer.save()
                iter = 0
        epoch += 1
        trainer.epoch = epoch
        trainer.save()
        