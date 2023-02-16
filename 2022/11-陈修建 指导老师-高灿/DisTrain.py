import os
import cv2
import glob
import torch
import argparse
import numpy as np
from PIL import Image
from loss import FocalLoss
from typing import Optional
from RSTPMtrain import RSTPM
import pytorch_lightning as pl
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from utils import min_max_norm, cvt2heatmap
from model_unet import DiscriminativeSubNetwork
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from dataloader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset, MVTecDataset

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class DisNet(pl.LightningModule):
    def __init__(self, hparams):
        super(DisNet, self).__init__()
        self.save_hyperparameters(hparams) # 超参数存储
        self.init_features()
        self.load_models_and_init() # 模型加载
        self.init_trainsform()
    def load_models_and_init(self):
        self.model_seg = DiscriminativeSubNetwork(in_channels=3, out_channels=2)
        self.model_seg.apply(weights_init)
        self.rstpm = RSTPM(args).load_from_checkpoint(args.stpm_path)
        self.rstpm.eval()
        for p in self.rstpm.parameters():
            p.requires_grad = False
    def init_trainsform(self):
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
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_seg.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[args.num_epochs*0.8,args.num_epochs*0.9],gamma=0.2, last_epoch=-1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    def train_dataloader(self):
        dataset = MVTecDRAEMTrainDataset(os.path.join(args.dataset_path, args.category), args.anomaly_source_path, resize_shape=[args.input_size, args.input_size], transform = self.data_transforms, gt_transform=self.gt_transforms)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=args.pin_memory)
        return train_loader
    def on_train_start(self) -> None:
        self.rstpm.eval()
        self.sample_path, self.traning_data_path = prep_dirs(self.logger.log_dir)
    def training_step(self, batch, batch_idx):
        image, anomaly_mask, augmented_image, _, file_name, type, _ = batch
        out_mask, a_map_list = self(augmented_image)
        loss = self.cal_loss(out_mask, anomaly_mask)
        self.save_traning_data(image, augmented_image, anomaly_mask, type, file_name, a_map_list) # 保存增强的图片
        self.log('loss', loss)
        return loss
    def on_test_start(self) -> None:
        self.rstpm.eval()
        self.sample_path, _ = prep_dirs(self.logger.log_dir)
    def test_dataloader(self):
        dataset = MVTecDRAEMTestDataset(os.path.join(args.dataset_path, args.category), resize_shape=[args.input_size, args.input_size], transform = self.data_transforms, gt_transform=self.gt_transforms)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        # test_datasets = MVTecDataset(root=os.path.join(self.hparams.dataset_path,self.hparams.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        # test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
        return test_loader
    def test_step(self,  batch, batch_idx) -> Optional[STEP_OUTPUT]:
        image, mask, label, file_name, type, _ = batch
        out_mask, a_map_list = self(image)
        gt_np = mask.cpu().numpy().astype(int)
        out_mask = out_mask[0, 1, :, :]
        self.gt_list_px_lvl.extend(gt_np.ravel()) # 让多维度变成一维后添加到list中
        self.pred_list_px_lvl_onlyDis.extend(out_mask.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0, 0])
        # self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl_onlyDis.append(out_mask.max())
        a_map_list = [a.squeeze().to('cpu').detach().numpy() for a in a_map_list]
        out_mask = out_mask.to('cpu').detach().numpy()
        total_out_mask = self.get_total_anomaly_map(out_mask, a_map_list)
        self.pred_list_px_lvl_total.extend(total_out_mask.ravel())
        self.pred_list_img_lvl_total.append(total_out_mask.max())
        image = self.inv_normalize(image)  # inv_normalize转换成图片没有归一化之前的样子
        input_x = cv2.cvtColor(image.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)  # 将BGR转成RGB并改变宽高的位置
        self.save_results(out_mask, total_out_mask, a_map_list, input_x, gt_np[0][0]*255,  type[0], file_name[0])
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join('/data/cxj/results/TestDis', args.category),max_epochs=args.num_epochs)
        # trainer.test(self.rstpm)
        print('\ncalculating roc_auc')
        only_dis_pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl_onlyDis)
        only_dis_img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl_onlyDis)
        all_mul_pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl_total)
        all_mul_img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl_total)
        dir = os.path.join(self.hparams.project_path, self.hparams.category)
        if not dir:
            os.makedirs(dir, exist_ok=True)
        record = os.path.join(self.hparams.project_path, self.hparams.record_txt)
        with open(record, 'a+', encoding="utf-8") as f:
            f.write(f'{self.hparams.category}\t\t\tonly_dis_pixel_auc：{only_dis_pixel_auc}\t\t\tonly_dis_img_auc：{only_dis_img_auc}\t\t\tall_mul_pixel_auc：{all_mul_pixel_auc}\t\t\tall_mul_img_auc：{all_mul_img_auc}\n')
        v1 = {'only_dis_pixel_auc': only_dis_pixel_auc, 'only_dis_img_auc': only_dis_img_auc,'all_mul_pixel_auc': all_mul_pixel_auc, 'all_mul_img_auc': all_mul_img_auc}
        self.log_dict(v1)
    def get_total_anomaly_map(self, dis_out_mask, a_map_list):
        anomaly_map = np.ones([dis_out_mask.shape[0], dis_out_mask.shape[1]])
        for i in range(len(a_map_list)):
            anomaly_map *= a_map_list[i]
        anomaly_map = anomaly_map*dis_out_mask
        return anomaly_map
    def cal_loss(self, out_mask_sm, anomaly_mask):
        loss_focal = FocalLoss()
        segment_loss = loss_focal(out_mask_sm, anomaly_mask)
        return segment_loss
    def init_features(self):
        self.gt_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl_onlyDis = []
        self.pred_list_px_lvl_onlyDis = []
        self.pred_list_img_lvl_total = []
        self.pred_list_px_lvl_total = []
    def forward(self, x):
        features_t, features_s, features_t_2, features_s_2, _  = self.rstpm(x)
        joined_in, a_map_list = self.cal_jointed(features_s, features_t, features_s_2, features_t_2, out_size=args.input_size)
        out_mask = self.model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        return out_mask_sm, a_map_list
    def cal_jointed(self, fs_list, ft_list, fs_list_2, ft_list_2, out_size=256):
        a_map_list = []
        a_map_list_2 = []
        all_map_list = []
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
            a_map = torch.unsqueeze(a_map, dim=1)
            # 双线性差值进行改变异常图的尺寸
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear') # 双线性插值扩展到256*256
            a_map = a_map.detach()
            a_map_list.append(a_map)
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
            a_map = a_map.detach()
            a_map_list_2.append(a_map)
        for p in zip(a_map_list, a_map_list_2):
            a = p[0]
            b = p[1]
            all_map_list.append(a+b)
        jointed = torch.cat(all_map_list, dim=1)
        return jointed, all_map_list
    def save_traning_data(self, image, augmented_image, mask, x_type, file_name, a_map_list = None):
        if a_map_list is not None:
            a_map_list = [a.cpu().numpy() for a in a_map_list]
        for i in range(image.size(0)):
            img = self.recover_image(image[i])
            ano_img = self.recover_image(augmented_image[i])
            gt_np = mask[i].cpu().numpy().astype(int)
            gt_np = gt_np[0]*255
            cv2.imwrite(os.path.join(self.traning_data_path, f'{x_type[i]}_{file_name[i]}.jpg'), img)
            cv2.imwrite(os.path.join(self.traning_data_path, f'{x_type[i]}_{file_name[i]}_augmented_image.jpg'), ano_img)
            cv2.imwrite(os.path.join(self.traning_data_path, f'{x_type[i]}_{file_name[i]}_gt.jpg'), gt_np)
            if a_map_list is not None:
                am64 = min_max_norm(a_map_list[0][i][0])
                am64 = cvt2heatmap(am64 * 255)
                am32 = min_max_norm(a_map_list[1][i][0])
                am32 = cvt2heatmap(am32 * 255)
                am16 = min_max_norm(a_map_list[2][i][0])
                am16 = cvt2heatmap(am16 * 255)
                cv2.imwrite(os.path.join(self.traning_data_path, f'{x_type[i]}_{file_name[i]}_am64.jpg'), am64)
                cv2.imwrite(os.path.join(self.traning_data_path, f'{x_type[i]}_{file_name[i]}_am32.jpg'), am32)
                cv2.imwrite(os.path.join(self.traning_data_path, f'{x_type[i]}_{file_name[i]}_am16.jpg'), am16)
    def recover_image(self, image):
        image = self.inv_normalize(image)  # inv_normalize转换成图片没有归一化之前的样子
        input_x = cv2.cvtColor(image.permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_BGR2RGB)  # 将BGR转成RGB并改变宽高的位置
        return input_x
    def save_results(self, out_mask, total_out_mask, a_map_list, input_image, gt, x_type, file_name):
        am64 = min_max_norm(a_map_list[0])
        am64 = cvt2heatmap(am64 * 255)
        am32 = min_max_norm(a_map_list[1])
        am32 = cvt2heatmap(am32 * 255)
        am16 = min_max_norm(a_map_list[2])
        am16 = cvt2heatmap(am16 * 255)
        out_mask = min_max_norm(out_mask)
        out_mask = cvt2heatmap(out_mask*255)
        total_out_mask = min_max_norm(total_out_mask)
        total_out_mask = cvt2heatmap(total_out_mask*255)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_image)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_am64.jpg'), am64)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_am32.jpg'), am32)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_am16.jpg'), am16)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_dis_mask.jpg'), out_mask)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_total_mask.jpg'), total_out_mask)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # 这里的Conv和BatchNnorm是torc.nn里的形式
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # bn层里初始化γ，服从（1，0.02）的正态分布
        m.bias.data.fill_(0)  # bn层里初始化β，默认为0
def prep_dirs(root):
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    traning_data_path = os.path.join(root, 'traning_data')
    os.makedirs(traning_data_path, exist_ok=True)
    return sample_path, traning_data_path
# 获取命令行参数
def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'D:\Dataset\mvtec_anomaly_detection') #/tile') #'/home/changwoo/hdd/datasets/mvtec_anomaly_detection'
    parser.add_argument('--anomaly_source_path', default=r'D:\Dataset\dtd\images')
    parser.add_argument('--category', default='grid')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', default=0.0001)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.0001)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--load_size', default=256) # 256
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--project_path', default=r'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test') #'/home/changwoo/hdd/project_results/STPM_lightning/210621') #210605') #
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--amap_mode', choices=['mul','sum'], default='mul')
    parser.add_argument('--val_freq', default=5)
    parser.add_argument('--weights_file_version', type=str, default=None)
    parser.add_argument('--continue_go', type=bool,  default=False)
    parser.add_argument('--devices', type=int, default=[2])
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--record_txt', type=str, default='record_dis.txt')
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--stpm_ckpt_path',help='path of pretained reconstruct stpm model', default=r'D:\Dataset\mvtec_anomaly_detection') #/tile') #'/home/changwoo/hdd/datasets/mvtec_anomaly_detection'
    args = parser.parse_args()
    return args
# 动态权重加载
def auto_select_weights_file(weights_file_version=None, au_path = None):
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
def get_start():

    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_path,
                                                                                args.category), max_epochs=args.num_epochs)  # , check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    if args.phase == 'train':
        if(not args.continue_go):
            model = DisNet(hparams=args)
        else:
            weights_file_path = auto_select_weights_file(args.weights_file_version)  # select latest weight if args.weights_file_version == None
            model = DisNet(hparams=args).load_from_checkpoint(weights_file_path)
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        # select weight file.
        weights_file_path = auto_select_weights_file(
            args.weights_file_version)  # select latest weight if args.weights_file_version == None
        if weights_file_path != None:
            model = DisNet(hparams=args).load_from_checkpoint(weights_file_path)
            trainer.test(model)
        else:
            print('Weights file is not found!')

def auto_select_weights_file(weights_file_version, au_path = None):
    print()
    # version_list = glob.glob(os.path.join(args.project_path, args.category) + '/lightning_logs/version_*')
    path = au_path or args.project_path
    root = os.path.join(path, args.category)
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
if __name__ == '__main__':
    args = get_args()
    cates = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
             'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    if(args.category == 'all'):
        for cate in cates:
            args.category = cate
            args.stpm_path = auto_select_weights_file(None, au_path=args.stpm_ckpt_path)
            print(f'Current catefory is {args.category}. The checkpoin is finded: {args.stpm_path}')
            get_start()
    else:
        args.stpm_path = auto_select_weights_file(None, au_path=args.stpm_ckpt_path)
        print(f'Current catefory is {args.category}. The checkpoin is finded: {args.stpm_path}')
        get_start()

