import os
import torch
from model.Model import TrainModel

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
    def __init__(self,resume_epoch=0,is_train = True,continue_train = False,pre_train = True):
        """ opt """
        self.gpu_ids = [4,5,6,7]
        self.is_train = is_train
        self.continue_train = continue_train
        self.epoch = resume_epoch
        self.niter = 5
        self.niter_decay = 5
        """ end """
        self.model = TrainModel(epoch=self.epoch,continueTarin=self.continue_train,pre_train = pre_train).to('cuda:4')
        # Model to GPU
        if len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model , device_ids=self.gpu_ids)
            self.model_on_one_gpu = self.model.module
        else:
            self.model.to(self.gpu_ids[0])
            self.model_on_one_gpu = self.model

        self.generated = None
        #self.optimizer_G = self.model_on_one_gpu.create_optimizers()
        self.optimizer_G,self.optimizer_D = self.model_on_one_gpu.create_optimizers()
        self.old_lr = 0.0002
        if self.continue_train:
            try:
                load_path = os.path.join('./checkpoints','_coco2_','optimizer.pth')
                checkpoint = torch.load(load_path)
                self.optimizer_G.load_state_dict(checkpoint['G'])
                self.optimizer_D.load_state_dict(checkpoint['D'])
            except FileNotFoundError as err:
                print(err)
                print('Not find optimizer state dict: ' + load_path + '. Do not load optimizer!')
        
        self.last_data, self.last_netCorr, self.last_netG, self.last_optimizer_G = None, None , None ,None
        self.g_losses = {}
        self.d_losses = {}
        self.c_losses = {}
        self.corr_out = None
        self.scaler = GradScaler(enabled= True)

    def run_generator_one_step(self,data):
        self.optimizer_G.zero_grad()
        g_losses , out = self.model(data,mode= 'generator')
        g_loss = sum(g_losses.values()).mean()
        # g_loss backward
        self.scaler.scale(g_loss).backward()
        self.scaler.unscale_(self.optimizer_G)
        # self.optimizer_G.step()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        self.g_losses = g_losses
        self.out = out

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        GforD = {}
        GforD['fake_image'] = self.out['fake_image']
        GforD['adaptive_feature_seg'] = self.out['adaptive_feature_seg']
        GforD['adaptive_feature_img'] = self.out['adaptive_feature_img']
        d_losses = self.model(data,mode= 'discriminator',GforD = GforD)
        d_loss = sum(d_losses.values()).mean()
        # d_loss.backward()
        self.scaler.scale(d_loss).backward()
        self.scaler.unscale_(self.optimizer_D)
        # self.optimizer_D.step()
        self.scaler.step(self.optimizer_D)
        self.scaler.update()
        self.d_losses = d_losses

    def run_DomainAlin_one_step(self,data):
        self.optimizer_G.zero_grad()
        g_losses , out = self.model(data,mode= 'domainalin')
        corr_losses = {}
        corr_losses['no_vgg_feat'] = g_losses['no_vgg_feat']
        corr_losses['G_warp_cycle'] = g_losses['G_warp_cycle']
        corr_losses['G_warp_self'] = g_losses['G_warp_self']
        c_loss = sum(corr_losses.values()).mean()
        c_loss = 1.0 * c_loss

        # theta = out['adaptive_feature_seg']
        # phi = out['adaptive_feature_img']
        # # m = out['index_check']
        # for i in range(len(theta)):
        #     theta[i].retain_grad()
        #     phi[i].retain_grad()
        #     m[i].retain_grad()
        self.scaler.scale(c_loss).backward()
        
        #
        # test_model = self.model_on_one_gpu.netcorr.patch_match.refine_net
        # for _,para in test_model.named_parameters():
        #     print(para.grad)
        # test_model = self.model_on_one_gpu.netcorr.adptive_model_img.G_middle_2
        # for _,para in test_model.named_parameters():
        #     print(para.grad)
        #
        self.scaler.unscale_(self.optimizer_G)

        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        self.c_losses = corr_losses
        self.corr_out = out

    def get_latest_losses(self): # ------------------------------------------------------------------
        return {**self.g_losses,**self.d_losses}
        pass

    def get_latest_generated(self):
        return self.out['fake_image']
    
    def get_latest_generated_corr(self):
        return self.out['warp_out'][3]

    def save(self):
        self.model_on_one_gpu.save(self.epoch)
        torch.save({
            'G':self.optimizer_G.state_dict(), \
            'D':self.optimizer_D.state_dict(), \
            'lr': self.old_lr, \
        },os.path.join('./checkpoints','_coco2_','optimizer.pth'))

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

    def print_loss(self):
        
        for key in self.g_losses.keys():
            print(key,': ',self.g_losses[key].item(),end=' ')
        print(' ')
        for key in self.d_losses.keys():
            print(key,': ',self.d_losses[key].item(),end=' ')
        print(' ')
        #print(end='\r')
        pass