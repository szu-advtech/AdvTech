import argparse
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
import shutil
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
import string
import random
from sklearn.metrics import confusion_matrix
from reconstruct import ReconstructiveStudent
from dataloader import MVTecDataset

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)
# 文件夹准备
def prep_dirs(root):
    attentions_path = os.path.join(root, 'attention')
    os.makedirs(attentions_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    # 源代码复制
    # copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE']) # copy source code
    return sample_path, source_code_save_path,attentions_path


def auto_select_weights_file(weights_file_version):
    print()
    # version_list = glob.glob(os.path.join(args.project_path, args.category) + '/lightning_logs/version_*')
    root = os.path.join(args.project_path, args.category)
    # 版本加载
    version_list = glob.glob(root+ r'/lightning_logs/version_*')
    version_list.sort(reverse=True, key=lambda x: os.path.getmtime(x))
    if weights_file_version != None:
        # 把传入的版本设置到头部
        version_list = [root + r'/lightning_logs/' + weights_file_version] + version_list
    for i in range(len(version_list)):
        # if os.path.exists(os.path.join(version_list[i],'checkpoints')):
        weights_file_path = glob.glob(os.path.join(version_list[i],'checkpoints')+'/*')
        if len(weights_file_path) == 0:
            if weights_file_version != None and i == 0:
                print(f'Checkpoint of {weights_file_version} not found')
            continue
        else:
            weights_file_path = weights_file_path[0]
            # 保证是ckpt后缀文件
            if weights_file_path.split('.')[-1] != 'ckpt':
                continue
        print('Checkpoint found : ', weights_file_path)
        print()
        return weights_file_path
    print('Checkpoint not found')
    print()
    return None

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]



def show_cam_on_image(img, anomaly_map):
    heatmap = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(heatmap) + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cvt2heatmap(gray):
    # uni8表示的是无符号整形，表示范围是[0.255]的整数，cv2.COLORMAP_JET用于生成热力图
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)


class RSTPM(pl.LightningModule):
    def __init__(self, hparams):
        super(RSTPM, self).__init__()
        if(hparams is not None):
            self.save_hyperparameters(hparams) # 超参数存储
        self.init_features()
        self.load_models_and_init() # 模型加载
        # 特征提炼，最后拿到的是整个基本的包含两个block的残差结构的输出的特征矩阵
        def hook_t(module, input, output):
            self.features_t.append(output)
        def hook_s(module, input, output):
            self.features_s.append(output)
        def hook_t_2(module, input, output):
            self.features_t_2.append(output)
        def hook_s_2(module, input, output):
            self.features_s_2.insert(0, output) #student2是反着来的
        def hook_t_last(module, input, output): #teacher1的最后的输出是student2的输入
            self.feature_t_2_last = output
        def hook_attention(module, input, output):
            self.attentions.append(output)
        # self.model_t = resnet18(pretrained=True).eval() # 加载resnet18神经网络并且设置为推理模式，并作为teacher网络的参数
        # self.model_t_2 = resnet50(pretrained=True).eval() # teacher2的模型
        # for param in self.model_t.parameters():
        #     param.requires_grad = False # 不会反向传播
        # for param in self.model_t_2.parameters():
        #     param.requires_grad = False # 不会反向传播
        # layer1代表的是各个基本的残差结构
        # 这里表示拿到前三层作为特征金子塔的特征提取器
        # -1表示该层中最后一个反向传播结束后
        self.model_t.layer1[-1].register_forward_hook(hook_t) # register_forward_hook注册hook，表示前向传播输出结果后进行调用hook_t
        self.model_t.layer2[-1].register_forward_hook(hook_t)
        self.model_t.layer3[-1].register_forward_hook(hook_t)
        self.model_t.layer4[-1].register_forward_hook(hook_t_last)
        self.model_t_2.layer1[-1].register_forward_hook(hook_t_2) # register_forward_hook注册hook，表示前向传播输出结果后进行调用hook_t
        self.model_t_2.layer2[-1].register_forward_hook(hook_t_2)
        self.model_t_2.layer3[-1].register_forward_hook(hook_t_2)
        # 表示学生模型有同样的结构
        # self.model_s = resnet18(pretrained=False) # default: False
        # self.model_s_2 = ReconstructiveStudent() # 重构学生网络
        # self.model_s.apply(init_weights)
        self.model_s.layer1[-1].register_forward_hook(hook_s)
        self.model_s.layer2[-1].register_forward_hook(hook_s)
        self.model_s.layer3[-1].register_forward_hook(hook_s)
        self.model_s_2.layer1[-1].register_forward_hook(hook_s_2)
        self.model_s_2.layer2[-1].register_forward_hook(hook_s_2)
        self.model_s_2.layer3[-1].register_forward_hook(hook_s_2)
        # 注意力图打印
        self.model_s_2.atten1.register_forward_hook(hook_attention)
        self.model_s_2.atten2.register_forward_hook(hook_attention)
        self.criterion = torch.nn.MSELoss(reduction='sum') # sum表示不会除以元素的个数
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.gt_list_px_lvl_2 = []
        self.pred_list_px_lvl_2 = []
        self.gt_list_img_lvl_2 = []
        self.pred_list_img_lvl_2 = []
        self.img_path_list = []
        '''
        这里是使用了使用ImageNet的均值和标准差
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        '''
        # 图片处理
        self.data_transforms = transforms.Compose([
                        transforms.Resize((self.hparams.load_size, self.hparams.load_size), Image.ANTIALIAS), # 图片处理  # load_szie初始是256 # ANTIALIAS表示高质量
                        transforms.ToTensor(), # 将图片从HWC转换成CHW，并且将数值从[0,255]归一化到[0,1]
                        transforms.CenterCrop(self.hparams.input_size), # input_size是256
                        transforms.Normalize(mean=mean_train, # 数据归一化
                                            std=std_train)])

        self.gt_transforms = transforms.Compose([
                        transforms.Resize((self.hparams.load_size, self.hparams.load_size)), # resize使用双线性插值
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.hparams.input_size)])
        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    def load_models_and_init(self):
        self.model_t = ResNet(BasicBlock, [2, 2, 2, 2])
        self.model_s = ResNet(BasicBlock, [2, 2, 2, 2])
        self.model_t_2 = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_s_2 = ReconstructiveStudent() # 重构学生网络
        self.model_t.load_state_dict(torch.load('/home2/cxj/models/resnet18.pth'))
        self.model_t_2.load_state_dict(torch.load('/home2/cxj/models/resnet50.pth'))
        self.model_t.eval()
        self.model_t_2.eval()
        for param in self.model_t.parameters():
            param.requires_grad = False  # 不会反向传播
        for param in self.model_t_2.parameters():
            param.requires_grad = False  # 不会反向传播
    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.gt_list_px_lvl_2 = []
        self.pred_list_px_lvl_2 = []
        self.gt_list_img_lvl_2 = []
        self.pred_list_img_lvl_2 = []
        self.img_path_list = []    
    # 老师和学生的特征列表
    def init_features(self):
        self.features_t = []
        self.features_s = []
        self.features_t_2 = []
        self.features_s_2 = []
        self.feature_t_2_last = []
        self.attentions = []
    def forward(self, x):
        self.init_features()
        self.model_t(x)
        self.model_t_2(x)
        self.model_s_2(self.feature_t_2_last, self.features_t_2[2], self.features_t_2[1])
        self.model_s(x)
        return self.features_t, self.features_s, self.features_t_2, self.features_s_2, self.attentions
    # 损失计算
    def cal_loss(self, fs_list, ft_list):
        tot_loss = 0
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            _, _, h, w = fs.shape
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            f_loss = (0.5/(w*h))*self.criterion(fs_norm, ft_norm)
            tot_loss += f_loss
        return tot_loss
    # 计算异常图
    def cal_anomaly_map(self, fs_list, ft_list, fs_list_2, ft_list_2, attentions = [], out_size=256):
        if self.hparams.amap_mode == 'mul':
            anomaly_map = np.ones([out_size, out_size])
        else:
            anomaly_map = np.zeros([out_size, out_size])
        a_map_list = [] # 全部合成的3张图的
        a_map_list_each_1 = [] # student1全部单张
        a_map_list_each_2 = [] # student1全部单张
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
            a_map = torch.unsqueeze(a_map, dim=1)
            # 双线性差值进行改变异常图的尺寸
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear') # 双线性插值扩展到256*256
            a_map = a_map[0,0,:,:].to('cpu').detach().numpy() # 把宽高部分截取下来，这里使得变成二维的了
            a_map_list_each_1.append(a_map)
        # 重构网络
        for i in range(len(ft_list_2)):
            fs = fs_list_2[i]
            ft = ft_list_2[i]
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
            a_map = torch.unsqueeze(a_map, dim=1)
            # 双线性差值进行改变异常图的尺寸
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear')  # 双线性插值扩展到256*256
            a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()  # 把宽高部分截取下来，这里使得变成二维的了
            a_map_list_each_2.append(a_map)
        # a_map_list_each是所有单个的，现在需要合并两个
        for p in zip(a_map_list_each_1, a_map_list_each_2):
            a = p[0]
            b = p[1]
            a_map_list.append(a+b)
        # 全部相乘得到最终的异常图
        for i in range(len(a_map_list)):
            if self.hparams.amap_mode == 'mul':
                anomaly_map *= a_map_list[i]
            else:
                anomaly_map += a_map_list[i]
        if(attentions != None and len(attentions) > 0):
            attentions[0] = F.interpolate(attentions[0], size=out_size, mode='bilinear')
            attentions[1] = F.interpolate(attentions[1], size=out_size, mode='bilinear')
            attentions[0] = attentions[0][0, 0, :, :].to('cpu').detach().numpy()
            attentions[1] = attentions[1][0, 0, :, :].to('cpu').detach().numpy()
        return anomaly_map, a_map_list, a_map_list_each_1+a_map_list_each_2, attentions

    def save_anomaly_map(self, anomaly_map, a_maps, input_img, gt_img, file_name, x_type, a_map_list_each, attentions):
        if(self.hparams.need_sample):
            # (img-min)/(max-min) 表示每个参数和最大值之间差距的比值
            anomaly_map_norm = min_max_norm(anomaly_map)
            # 异常图中范围转成0~255，类型是浮点型
            # 得到相应的热力图
            anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)
            # 64x64 map
            am64 = min_max_norm(a_maps[0])
            am64 = cvt2heatmap(am64*255)
            # 32x32 map
            am32 = min_max_norm(a_maps[1])
            am32 = cvt2heatmap(am32*255)
            # 16x16 map
            am16 = min_max_norm(a_maps[2])
            am16 = cvt2heatmap(am16*255)
            s1_each_1_64 = min_max_norm(a_map_list_each[0])
            s1_each_1_64 = cvt2heatmap(s1_each_1_64*255)

            s2_each_1_64 = min_max_norm(a_map_list_each[3])
            s2_each_1_64 = cvt2heatmap(s2_each_1_64*255)

            s1_each_2_32 = min_max_norm(a_map_list_each[1])
            s1_each_2_32 = cvt2heatmap(s1_each_2_32*255)

            s2_each_2_32 = min_max_norm(a_map_list_each[4])
            s2_each_2_32 = cvt2heatmap(s2_each_2_32*255)

            s1_each_3_16 = min_max_norm(a_map_list_each[2])
            s1_each_3_16 = cvt2heatmap(s1_each_3_16*255)

            s2_each_3_16 = min_max_norm(a_map_list_each[5])
            s2_each_3_16 = cvt2heatmap(s2_each_3_16*255)

            # anomaly map on image
            heatmap = cvt2heatmap(anomaly_map_norm*255)
            hm_on_img = heatmap_on_image(heatmap, input_img)

            # save images
            # file_name = id_generator() # random id

            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_s1_each_1_64.jpg'), s1_each_1_64)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_s2_each_1_64.jpg'), s2_each_1_64)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_s1_each_2_32.jpg'), s1_each_2_32)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_s2_each_2_32.jpg'), s2_each_2_32)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_s1_each_3_16.jpg'), s1_each_3_16)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_s2_each_3_16.jpg'), s2_each_3_16)

            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_am64.jpg'), am64)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_am32.jpg'), am32)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_am16.jpg'), am16)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
            cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)
        if self.hparams.need_att:
            att1_16 = min_max_norm(attentions[0])
            att1_16 = cvt2heatmap(att1_16*255)
            att2_32 = min_max_norm(attentions[1])
            att2_32 = cvt2heatmap(att2_32*255)
            cv2.imwrite(os.path.join(self.attentions_path, f'{x_type}_{file_name}_att1_16.jpg'), att1_16)
            cv2.imwrite(os.path.join(self.attentions_path, f'{x_type}_{file_name}_att2_32.jpg'), att2_32)
    # 优化器
    def configure_optimizers(self):
        opt_s = torch.optim.SGD(self.model_s.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        opt_s_2 = torch.optim.SGD(self.model_s_2.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        return [opt_s, opt_s_2], []
    # 训练数据加载
    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(self.hparams.dataset_path,self.hparams.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory) #, pin_memory=True)
        return train_loader
#     def val_dataloader(self):
#         val_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
#         val_loader = DataLoader(val_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
#         return val_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(self.hparams.dataset_path,self.hparams.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def on_train_start(self):
        self.model_t.eval() # to stop running_var move (maybe not critical)
        self.model_t_2.eval()
        self.model_s.train()
        self.model_s_2.train()
        self.sample_path, self.source_code_save_path, self.attentions_path = prep_dirs(self.logger.log_dir)
    
#     def on_validation_start(self):
#         self.init_results_list()    
    # 准备了相关文件的路径
    def on_test_start(self):
        self.model_t.eval()
        self.model_s.eval()
        self.model_t_2.eval()
        self.model_s_2.eval()
        self.init_results_list()
        self.sample_path, self.source_code_save_path, self.attentions_path = prep_dirs(self.logger.log_dir)
    # batch就是上述数据库__getItem__返回的数，这些数组成一个元祖，下边x就是具体的图片的tensor
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _, _, file_name, _ = batch
        features_t, features_s, features_t_2, features_s_2, _ = self(x)
        if(optimizer_idx == 0): #student1对应的损失
            loss1 = self.cal_loss(features_s, features_t)
            self.log('train_loss_s', loss1, on_epoch=True)
            return loss1
        if(optimizer_idx == 1): #训练student2的损失
            loss2 = self.cal_loss(features_s_2, features_t_2)
            self.log('train_loss_s_2', loss2, on_epoch=True)
            return loss2

#     def validation_step(self, batch, batch_idx):
#         x, gt, label, file_name, x_type = batch
#         features_t, features_s = self(x)
#         # Get anomaly map
#         anomaly_map, _ = self.cal_anomaly_map(features_s, features_t, out_size=args.input_size)

#         gt_np = gt.cpu().numpy().astype(int)
#         self.gt_list_px_lvl.extend(gt_np.ravel())
#         self.pred_list_px_lvl.extend(anomaly_map.ravel())
#         self.gt_list_img_lvl.append(label.cpu().numpy()[0])
#         self.pred_list_img_lvl.append(anomaly_map.max())
#         self.img_path_list.extend(file_name)

    def test_step(self, batch, batch_idx):
        x, gt, label, file_name, x_type = batch
        features_t, features_s, features_t_2, features_s_2, attentions = self(x)
        print('------------------test step---- file_name:{}, x_type:{}'.format(file_name, x_type))
        # Get anomaly map
        # 当为mul模式时，anomaly_map为多个异常图相乘的结果
        # input_size就是输入图片最终转换的形状
        anomaly_map, a_map_list, a_map_list_each, attentions = self.cal_anomaly_map(features_s, features_t,features_t_2, features_s_2, attentions, out_size=self.hparams.input_size)
        gt_np = gt.cpu().numpy().astype(int)
        self.gt_list_px_lvl.extend(gt_np.ravel()) # 让多维度变成一维后添加到list中
        self.pred_list_px_lvl.extend(anomaly_map.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0]) # 添加标签，train阶段的时候，标签是0，test阶段的标签是1
        self.pred_list_img_lvl.append(anomaly_map.max()) # 异常图中的最大值
        self.img_path_list.extend(file_name) # 文件名添加
        # save images 图片保存
        x = self.inv_normalize(x) # inv_normalize转换成图片没有归一化之前的样子
        input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB) # 将BGR转成RGB并改变宽高的位置
        self.save_anomaly_map(anomaly_map, a_map_list, input_x, gt_np[0][0]*255, file_name[0], x_type[0], a_map_list_each, attentions)

#     def validation_epoch_end(self, outputs):
#         pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
#         img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
#         values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
#         self.log_dict(values)

    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        dir = os.path.join(self.hparams.project_path, self.hparams.category)
        if not dir:
            os.makedirs(dir, exist_ok=True)
        record = os.path.join(self.hparams.project_path, self.hparams.record_txt)
        with open(record, 'a+', encoding="utf-8") as f:
            f.write(f'{self.hparams.category}\t\t\tImage-Level：{img_auc}\t\t\tPixel-Level：{pixel_auc}\n')
        self.log_dict(values)



def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'D:\Dataset\mvtec_anomaly_detection') #/tile') #'/home/changwoo/hdd/datasets/mvtec_anomaly_detection'
    parser.add_argument('--category', default='grid')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', default=0.4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_size', default=256) # 256
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--project_path', default=r'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test') #'/home/changwoo/hdd/project_results/STPM_lightning/210621') #210605') #
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--amap_mode', choices=['mul','sum'], default='mul')
    parser.add_argument('--val_freq', default=5)
    parser.add_argument('--weights_file_version', type=str, default=None)
    parser.add_argument('--continue_go', type=bool,  default=False)
    parser.add_argument('--devices', type=int, default=[3])
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--record_txt', type=str, default='record_remake.txt')
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--need_sample', type=bool, default=True)
    parser.add_argument('--need_att', type=bool, default=True)
    # parser.add_argument('--weights_file_version', type=str, default='version_1')
    args = parser.parse_args()
    return args

def get_start():
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_path,
                                                                                args.category),
                                            max_epochs=args.num_epochs)  # , check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)

    if args.phase == 'train':
        if(not args.continue_go):
            model = RSTPM(hparams=args)
        else:
            weights_file_path = auto_select_weights_file( args.weights_file_version)  # select latest weight if args.weights_file_version == None
            model = RSTPM(hparams=args).load_from_checkpoint(weights_file_path)
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        # select weight file.
        weights_file_path = auto_select_weights_file(
            args.weights_file_version)  # select latest weight if args.weights_file_version == None
        if weights_file_path != None:
            model = RSTPM(hparams=args).load_from_checkpoint(weights_file_path)
            trainer.test(model)
        else:
            print('Weights file is not found!')
if __name__ == '__main__':

    args = get_args() # 加载参数
    cates = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
             'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    if(args.category == 'all'):
        for cate in cates:
            args.category = cate
            get_start()
    else:
        get_start()

