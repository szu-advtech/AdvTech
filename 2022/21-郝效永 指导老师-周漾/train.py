import torch
import torch.nn as nn

import dataset.dataset as Dataset
import dataset
import os
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

def save_out_result(trainer_out,data,epoch,iter):
    save_dir = os.path.join('./result','%s_epoch'%epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = '%s_iter_out.jpg'%iter
    util.tensor_to_RGB(trainer_out['fake_image'],os.path.join(save_dir,file_name))
    file_name = '%s_iter_real.jpg'%iter
    util.tensor_to_RGB(data['image'],os.path.join(save_dir,file_name))
    file_name = '%s_iter_exampler.jpg'%iter
    util.tensor_to_RGB(data['ref'],os.path.join(save_dir,file_name))
    file_name = '%s_iter_seg.jpg'%iter
    util.tensor_to_RGB(data['label'],os.path.join(save_dir,file_name))
    

    


if __name__ == "__main__":
    dataloader = dataset.create_dataloader()
    len_dataloader = len(dataloader)

    trainer = Trainer(resume_epoch=0,continue_train=False,pre_train=True)
    epoch = 0
    iter = 0
    D_per_G = 1
    
    for epoches in range(20):
        start = time.time()
        if epoch >= 5:
            trainer.update_learning_rate(epoch=epoch)
        for i,data_i in enumerate(dataloader,start=epoch):

            trainer.run_generator_one_step(data_i)
            if i%D_per_G == 0:
                trainer.run_discriminator_one_step(data_i)
            now = time.time()
            util.print_latest_losses(epoch=epoch,i=i,num=len(dataloader),errors=trainer.get_latest_losses(),t = now - start)
            
            visualized_data(data_i)
            util.tensor_to_RGB(trainer.out['fake_image'],'./out.jpg')
            util.tensor_to_RGB(trainer.out['warp_cycle'],'./warp_cycle.jpg')
            warp_tensor = trainer.out['warp_out']
            visualized_warp(warp_tensor)

            
            iter += 1
            if iter == 100:
                trainer.save()
                iter = 0
            if i % 1000 == 0:
                save_out_result(trainer_out=trainer.out,data=data_i,epoch=epoch,iter=i)
        epoch += 1
        trainer.epoch = epoch
        trainer.save()