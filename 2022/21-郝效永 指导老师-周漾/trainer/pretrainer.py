import os
import torch
from model.Model import TrainModel
from model.correspondence import Domain_alin

"""   """
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self, enabled):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
"""   """

class Trainer():
    """
    there has models and optimizers
    """
    def __init__(self,resume_epoch=0,is_train = True,continue_train = False):
        """ opt """
        self.gpu_ids = [0]
        self.is_train = is_train
        self.continue_train = continue_train
        self.epoch = 0
        self.niter = 100
        self.niter_decay = 0
        """ end """
        self.model = Domain_alin()
        # Model to GPU
        if len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model , device_ids=self.gpu_ids)
            self.model_on_one_gpu = self.model.module
        else:
            self.model.to(self.gpu_ids[0])
            self.model_on_one_gpu = self.model

        self.generated = None
        self.optimizer_corr = torch.optim.Adam(self.model.parameters(),lr=0.0002,betas=(0.5,0.999))
        self.old_lr = 0.0002
        if self.continue_train:
            try:
                load_path = os.path.join('./checkpoints','_coco2_','optimizer_corr.pth')
                checkpoint = torch.load(load_path)
                self.optimizer_corr.load_state_dict(checkpoint)
            except FileNotFoundError as err:
                print(err)
                print('Not find optimizer state dict: ' + load_path + '. Do not load optimizer!')
        
        self.last_data, self.last_netCorr, self.last_netG, self.last_optimizer_G = None, None , None ,None
        self.losses = {}
        self.scaler = GradScaler(enabled= True)

    def run_corr_one_step(self,data):
        self.optimizer_corr.zero_grad()
        out = self.model(data['ref'],data['image'],data['label_ref'],data['label'])
        corr_loss = {}
        corr_loss['novgg_feat_loss'] = out['loss_novgg_featpair']
        warp = out['warp_out']
        warp_loss = 0.0
        for i in range(len(warp)):
            warp += 
        corr_loss = sum(corr_loss.values()).mean()
        # g_loss backward
        self.scaler.scale(corr_loss).backward()
        self.scaler.unscale_(self.optimizer_corr)
        # self.optimizer_G.step()
        self.scaler.step(self.optimizer_corr)
        self.scaler.update()
        self.losses = corr_loss
        self.out = out

    def get_latest_losses(self): # ------------------------------------------------------------------
        return self.losses
        pass

    def get_latest_generated(self):
        return self.out['warp_out'][3]

    def save(self):
        self.model_on_one_gpu.save(self.epoch)
        torch.save({
            'Corr':self.optimizer_corr.state_dict(), \
            'lr': self.old_lr, \
        },os.path.join('./checkpoints','_coco2_','optimizer_corr.pth'))

    def update_learning_rate(self,epoch):
        if self.epoch > self.niter:
            lrd = self.old_lr / self.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr
        
        if new_lr != self.old_lr:
            new_lr_G = new_lr
            new_lr_D = new_lr
        else:
            new_lr_G = self.old_lr
            new_lr_D = self.old_lr

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = new_lr_D
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = new_lr_G
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    