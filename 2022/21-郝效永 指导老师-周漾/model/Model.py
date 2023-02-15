import torch
import torch.nn.functional as F
import model.network.generator as generrator
import model.network.discriminator as discriminator
import model.correspondence as correspondence
import model.network.architecture as architecture
import model.network.CXloss as CXloss
import model.network.GANloss as GANloss
import utils.util as util
import itertools
try:
    from torch.cuda.amp import autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
class TrainModel(torch.nn.Module):
    def __init__(self,continueTarin = False , epoch = 0,pre_train = True):
        super().__init__()
        """ opt """
        self.gpu_ids = [4,5,6,7]
        self.continueTrain = continueTarin
        self.epoch = epoch
        self.whitch_perceptual = '4_2'
        self.warp_cycle_weight = 1.0
        self.warp_self_weight  = 500.0
        self.GAN_weight = 10.0
        self.GANfeat_weight = 10.0
        self.VGG_weight = 10.0
        self.fm_ratio_weight = 1.0
        self.perceptual_weight = 0.001
        self.CX_weight = 1.0
        self.no_ganFeat_loss = True
        self.lr = 0.0002
        """ end """
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        """ creat net """
        self.net_G = generrator.SPADEGenerator()
        self.net_G.init_weights(init_type='kaiming')
        self.net_D = discriminator.MultiscaleDiscriminator()
        self.net_D.init_weights(init_type='kaiming')
        self.netcorr = correspondence.Domain_alin()
        self.netcorr.init_weights(init_type='kaiming')
        if self.continueTrain:
            #self.seg_expand_net = util.load_network(self.seg_expand_net,"netSeg",self.epoch)
            self.net_G = util.load_network(self.net_G,"netG",self.epoch)
            self.net_D = util.load_network(self.net_D,"netD",self.epoch)
            self.netcorr = util.load_network(self.netcorr,"netCorr",self.epoch)
        """ set loss function """
        if pre_train:
            self.netcorr = util.load_network(self.netcorr,"netCorr",'pre')
        #vgg_net
        self.vggnet_fix = architecture.VGG_Feature_extractor()
        self.vggnet_fix.load_state_dict(torch.load('vgg/vgg19_conv.pth'))
        self.vggnet_fix.eval()
        for param in self.vggnet_fix.parameters():
            param.requires_grad = False
        self.vggnet_fix.to(4)

        #contextual loss
        self.contextual_forward_loss = CXloss.ContextualLoss_forward()
        #GAN loss
        self.criterionGAN = GANloss.GANLoss('original',tensor=self.FloatTensor)
        #L1 loss
        self.criterionFeat = torch.nn.L1Loss()
        #L2 loss
        self.MSE_loss = torch.nn.MSELoss()
        #setting which layer is used in the perceptual loss
        if self.whitch_perceptual == '4_2':
            self.perceptual_layer = -2
        elif self.whitch_perceptual == '5_2':
            self.perceptual_layer = -1

    def forward(self,data,mode,GforD=None):
        input_label, input_semantics, real_img , self_ref , ref_img , ref_label , ref_semantics = self.preprocess_input(data,)
        input_seg = input_semantics
        ref_seg = ref_semantics
        generated_out = {}

        if mode == 'generator':
            g_loss,generated_out = self.compute_generator_loss(input_label , input_seg , real_img , ref_label , ref_seg , ref_img , self_ref)
            out = {}
            out['fake_image'] = generated_out['fake_image']
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            out['warp_out'] = None if 'warp_out' not in generated_out else generated_out['warp_out']
            out['adaptive_feature_seg'] = None if 'adaptive_feature_seg' not in generated_out else generated_out['adaptive_feature_seg']
            out['adaptive_feature_img'] = None if 'adaptive_faeture_img' not in generated_out else generated_out['adaptive_feature_img']
            out['warp_cycle'] = None if 'warp_cycle' not in generated_out else generated_out['warp_cycle']
            return g_loss,out
        
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics,real_img,GforD,label = input_label)
            return d_loss

        elif mode == 'inference':
            out = {}
            with torch.no_grad():
                out = self.inference(input_semantics,ref_semantics=ref_semantics,ref_image=ref_img,self_ref=self_ref,real_image=real_img)
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            return out

        elif mode == 'domainalin':
            a_loss,a_out =  self.compute_domainalin_loss(input_label , input_seg , real_img , ref_label , ref_seg , ref_img , self_ref)
            out = {}
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            out['warp_out'] = None if 'warp_out' not in a_out else a_out['warp_out']
            out['adaptive_feature_seg'] = None if 'adaptive_feature_seg' not in a_out else a_out['adaptive_feature_seg']
            out['adaptive_feature_img'] = None if 'adaptive_feature_img' not in a_out else a_out['adaptive_feature_img']
            out['warp_cycle'] = None if 'warp_cycle' not in a_out else a_out['warp_cycle']
            out['index_check'] = None if 'index_check' not in a_out else a_out['index_check']
            return a_loss,out
        else:
            raise ValueError("|mode| is invalid")
        pass

    def create_optimizers(self):
        beta1, beta2 = 0.0, 0.999
        G_lr, D_lr = self.lr / 2, self.lr * 2
        optimizer_G = torch.optim.Adam(itertools.chain(self.netcorr.parameters(),),lr = G_lr,betas=(beta1,beta2),eps=1e-3)
        optimizer_G = torch.optim.Adam(itertools.chain(self.net_G.parameters(),self.netcorr.parameters()),lr = G_lr,betas=(beta1,beta2),eps=1e-3)
        optimizer_D = torch.optim.Adam(itertools.chain(self.net_D.parameters(),),lr=D_lr,betas=(beta1,beta2))
        return optimizer_G,optimizer_D
        #return optimizer_G

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def save(self,epoch = 0):
        #util.save_network(self.seg_expand_net,'netSeg',epoch=epoch)
        util.save_network(self.net_G,'netG',epoch = epoch)
        util.save_network(self.net_D,'netD',epoch = epoch)
        util.save_network(self.netcorr,'netCorr',epoch = epoch)
        pass

    def preprocess_input(self, data):
        if self.use_gpu():
            for k in data.keys():
                try:
                    data[k] = data[k].cuda()
                except:
                    continue
        label = data['label'].float()  # input skeleton image
        label_ref = data['label_ref'].float() # input ref img's skeleton image
        input_semantics = data['seg'].float() # input skeleton image
        ref_semantics = data['seg_ref'].float() # input ref img's skeleton image
        image = data['image'] # input skeleton's person image
        ref = data['ref'] # input ref person image
        self_ref = data['self_ref']
        return label , input_semantics , image , self_ref , ref , label_ref , ref_semantics
        pass
    
    def compute_domainalin_loss(self, input_label , input_semantics , real_image ,ref_label = None ,ref_semantics = None, ref_iamge = None, self_ref = None):
        A_losses = {}
        """ now !!!! generate_fake !!!! """
        generate_out = self.generate_fake(input_semantics , real_image , ref_semantics ,ref_iamge ,self_ref,is_A=True)
        weights = [1.0/32 , 1.0/16 , 1.0/8 , 1.0/4 , 1.0]
        sample_weights = self_ref/(sum(self_ref)+1e-5)
        sample_weights = sample_weights.view(-1,1,1,1)

        """ Domain Alin Loss """
        if 'loss_novgg_featpair' in generate_out and generate_out['loss_novgg_featpair'] is not None:
            A_losses['no_vgg_feat'] = generate_out['loss_novgg_featpair']
        
        """ Warping Cycle Loss """ # firstly warp img reference 8x8
        warp_cycle = generate_out['warp_cycle']
        scale_factor = ref_iamge.size()[-1]//warp_cycle.size()[-1]
        ref = F.avg_pool2d(ref_iamge,scale_factor,stride = scale_factor)
        A_losses['G_warp_cycle'] = F.l1_loss(warp_cycle,ref) * self.warp_cycle_weight

        """ Warping Loss """ # patch match warping loss
        """ 128 x 128 """
        warp1 ,warp2 ,warp3 ,warp4 = generate_out['warp_out']
        A_losses['G_warp_self'] = \
            torch.mean(F.l1_loss(warp4,real_image,reduction='none') * sample_weights) * self.warp_self_weight *1.0 + \
            torch.mean(F.l1_loss(warp3,F.avg_pool2d(real_image,2,stride = 2) , reduction='none') * sample_weights) * self.warp_self_weight * 1.0 + \
            torch.mean(F.l1_loss(warp2,F.avg_pool2d(real_image,4,stride = 4) , reduction='none') * sample_weights) * self.warp_self_weight * 1.0 + \
            torch.mean(F.l1_loss(warp1,F.avg_pool2d(real_image,8,stride = 8) , reduction='none') * sample_weights) * self.warp_self_weight * 1.0

        return A_losses,generate_out


    def compute_generator_loss(self, input_label , input_semantics , real_image ,ref_label = None ,ref_semantics = None, ref_iamge = None, self_ref = None):
        G_losses = {}
        """ now !!!! generate_fake !!!! """
        generate_out = self.generate_fake(input_semantics , real_image , ref_semantics ,ref_iamge ,self_ref)
        generate_out['fake_image'] = generate_out['fake_image'].float()
        weights = [1.0/32 , 1.0/16 , 1.0/8 , 1.0/4 , 1.0]
        sample_weights = self_ref/(sum(self_ref)+1e-5)
        sample_weights = sample_weights.view(-1,1,1,1)

        """ Domain Alin Loss """
        if 'loss_novgg_featpair' in generate_out and generate_out['loss_novgg_featpair'] is not None:
            G_losses['no_vgg_feat'] = generate_out['loss_novgg_featpair']
        
        """ Warping Cycle Loss """ # firstly warp img reference 8x8
        warp_cycle = generate_out['warp_cycle']
        scale_factor = ref_iamge.size()[-1]//warp_cycle.size()[-1]
        ref = F.avg_pool2d(ref_iamge,scale_factor,stride = scale_factor)
        G_losses['G_warp_cycle'] = F.l1_loss(warp_cycle,ref) * self.warp_cycle_weight

        """ Warping Loss """ # patch match warping loss
        """ 128 x 128 """
        warp1 ,warp2 ,warp3 ,warp4 = generate_out['warp_out']
        G_losses['G_warp_self'] = \
            torch.mean(F.l1_loss(warp4,real_image,reduction='none') * sample_weights) * self.warp_self_weight *1.0 + \
            torch.mean(F.l1_loss(warp3,F.avg_pool2d(real_image,2,stride = 2) , reduction='none') * sample_weights) * self.warp_self_weight * 1.0 + \
            torch.mean(F.l1_loss(warp2,F.avg_pool2d(real_image,4,stride = 4) , reduction='none') * sample_weights) * self.warp_self_weight * 1.0 + \
            torch.mean(F.l1_loss(warp1,F.avg_pool2d(real_image,8,stride = 8) , reduction='none') * sample_weights) * self.warp_self_weight * 1.0 

        """ GAN Loss """
        pred_fake,pred_real = self.discriminate(input_semantics , generate_out['fake_image'] , real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake,True,for_discriminator=False) * self.GAN_weight
        if not self.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = 0.0
            for i in range(num_D):
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.GANfeat_weight / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        """ Feature Matching Loss """
        fake_features = self.vggnet_fix(generate_out['fake_image'], ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        loss = 0.0
        for i in range(len(generate_out['real_features'])):
            loss += weights[i] * util.weighted_l1_loss(fake_features[i],generate_out['real_features'][i].detach(),sample_weights)
        G_losses['fm'] = loss * self.VGG_weight * self.fm_ratio_weight

        """ Perceptual Loss """
        feat_loss = util.mse_loss(fake_features[self.perceptual_layer],generate_out['real_features'][self.perceptual_layer].detach())
        G_losses['perc'] = feat_loss * self.perceptual_weight

        """ CX Loss """
        G_losses['contextual'] = self.get_ctx_loss(fake_features,generate_out['ref_features']) * self.VGG_weight * self.CX_weight
        return G_losses,generate_out
        pass
    
    def compute_discriminator_loss(self,input_semantics,real_image,GforD,label=None):
        D_losses = {}
        with torch.no_grad():
            fake_image = GforD['fake_image'].detach()
            fake_image.requires_grad_()
        pred_fake,pred_real = self.discriminate(input_semantics,fake_image,real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake,False,for_discriminator=True) * self.GAN_weight
        D_losses['D_Real'] = self.criterionGAN(pred_real,True,for_discriminator=True) * self.GAN_weight
        return D_losses
        pass

    def generate_fake(self,input_semantics,real_img,ref_semantics=None ,ref_img=None,self_ref = None,is_A = False):
        generate_out = {}
        """ use example image generate VGG feature """
        generate_out['ref_features'] = self.vggnet_fix(ref_img,['r12','r22','r32','r42','r52'],preprocess=True)
        """ use semantics real image generate VGG feature """
        generate_out['real_features'] = self.vggnet_fix(real_img,['r12','r22','r32','r42','r52'],preprocess=True)
        with autocast(enabled=True):
            corr_out = self.netcorr(ref_img,real_img,ref_semantics,input_semantics)
            if is_A == False:
                generate_out['fake_image']  =  self.net_G(input_semantics,warp_out=corr_out['warp_out'])
        generate_out = {**generate_out,**corr_out}
        return generate_out
        pass

    def discriminate(self,input_semantics, fake_img,real_img):
        fake_concat = torch.cat([input_semantics,fake_img],dim=1)
        real_concat = torch.cat([input_semantics,real_img],dim=1)
        fake_and_real = torch.cat([fake_concat,real_concat],dim=0)
        with autocast(enabled=True):
            discriminator_out = self.net_D(fake_and_real)
        pred_fake,pred_real = self.divide_pred(discriminator_out)
        return pred_fake,pred_real
        pass

    def divide_pred(self,pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0)// 2 :] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake,real
        pass

    def inference(self,input_semantics,ref_semantics = None,ref_image=None,self_ref=None,real_image = None):
        generate_out = {}
        with autocast(enabled=True):
            corr_out = self.netcorr(ref_image,real_image,input_semantics,ref_semantics)
            generate_out['fake_image'] = self.net_G(input_semantics,warp_out = corr_out['warp_out'])
        generate_out = {**generate_out , **corr_out}
        return generate_out
        pass

    def get_ctx_loss(self,source,target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1],target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2],target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-3],2) , F.avg_pool2d(target[-3].detach(),2))) * 2
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1
        pass