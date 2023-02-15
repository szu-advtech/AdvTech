import torch.nn as nn

from tqdm import tqdm
from einops import rearrange
from modules.utils import *
from modules.generation import *
from modules.visualization import *
from metrics.timegan_metrics import calculate_pred_disc


def mask_it(x, masks):
    # x(bs, ts_size, z_dim)
    b, l, f = x.shape
    # ~表示取反 false -> true
    # 不reshape的话形状不对（二维变一维， 三维变二维）
    x_visible = x[~masks, :].reshape(b, -1, f)  # (bs, vis_size, z_dim)
    return x_visible


class Encoder(nn.Module):
    # 升维
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.z_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.hidden_dim)

    def forward(self, x):
        x_enc, _ = self.rnn(x)
        x_enc = self.fc(x_enc)
        return x_enc


class Decoder(nn.Module):
    # 降维
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.hidden_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.z_dim)

    def forward(self, x_enc):
        x_dec, _ = self.rnn(x_enc)
        x_dec = self.fc(x_dec)
        return x_dec


# 外推器（Extrapolator）
class Interpolator(nn.Module):
    def __init__(self, args):
        super(Interpolator, self).__init__()
        self.sequence_inter = nn.Linear(in_features=(args.ts_size - args.total_mask_size),
                                        out_features=args.ts_size)
        self.feature_inter = nn.Linear(in_features=args.hidden_dim,
                                       out_features=args.hidden_dim)

    def forward(self, x):
        # x(bs, vis_size, hidden_dim)
        x = rearrange(x, 'b l f -> b f l')  # x(bs, hidden_dim, vis_size)
        x = self.sequence_inter(x)  # x(bs, hidden_dim, ts_size)
        x = rearrange(x, 'b f l -> b l f')  # x(bs, ts_size, hidden_dim)
        x = self.feature_inter(x)  # x(bs, ts_size, hidden_dim)
        return x


class InterpoMAEUnit(nn.Module):
    def __init__(self, args):
        super(InterpoMAEUnit, self).__init__()
        self.args = args
        self.ts_size = args.ts_size
        # 掩盖的大小，掩盖的数量
        self.mask_size = args.mask_size
        self.num_masks = args.num_masks
        # 总掩盖的大小
        self.total_mask_size = args.num_masks * args.mask_size
        args.total_mask_size = self.total_mask_size
        self.z_dim = args.z_dim
        # 初始化encoder、interpolator、decoder
        self.encoder = Encoder(args)
        self.interpolator = Interpolator(args)
        self.decoder = Decoder(args)

    # 遮蔽一部分序列
    def forward_mae(self, x, masks):
        """
        No mask tokens, using Interpolation in the latent space
        没有掩码标记，在潜在空间中使用插值
        """
        # 获得无遮蔽的序列
        x_vis = mask_it(x, masks)  # (bs, vis_size, z_dim)
        x_enc = self.encoder(x_vis)  # (bs, vis_size, hidden_dim)
        x_inter = self.interpolator(x_enc)  # (bs, ts_size, hidden_dim)
        x_dec = self.decoder(x_inter)  # (bs, ts_size, z_dim)
        return x_inter, x_dec, masks

    # 对原有序列不进行遮蔽，直接编解码
    def forward_ae(self, x, masks):
        """
        mae_pseudo_mask is equivalent to the Autoencoder
            There is no interpolator in this mode
        mae_pseudo_mask 相当于自动编码器，此模式下没有插值器
        """
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_enc, x_dec, masks

    def forward(self, x, masks, mode):
        # 无论是随机更替还是交叉更替都需要遮蔽
        """
        Existing mode:
            1. train_ae
            2. train_mae
            3. random_generation
            4. cross_generation
        """
        if mode == 'train_ae':
            x_encoded, x_decoded, masks = self.forward_ae(x, masks)
        else:
            x_encoded, x_decoded, masks = self.forward_mae(x, masks)
        return x_encoded, x_decoded, masks


class InterpoMAE(nn.Module):
    def __init__(self, args, ori_data):
        super(InterpoMAE, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        # 初始化模型单元
        self.model = InterpoMAEUnit(args).to(self.device)
        self.ori_data = ori_data
        # 伪标记
        self.pseudo_masks = generate_pseudo_masks(args, args.batch_size)
        # 采用均方误差作为评判标准
        self.criterion = torch.nn.MSELoss(reduction='mean')
        # 采用adam优化器来更新参数 默认学习率lr=e-3
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # 迭代次数
        self.num_iteration = 0
        print(f'Successfully initialized {self.__class__.__name__}!')

    '''
        训练
    '''

    # 训练未采用遮蔽的自动编码器
    def train_ae(self):
        self.model.train()

        # tqdm为在循环中加一个进度提示
        for t in tqdm(range(self.args.ae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            x_enc, x_dec, masks = self.model(x_ori, self.pseudo_masks, 'train_ae')
            loss = self.criterion(x_dec, x_ori)

            # 没过一个epoch，iteration+1
            self.num_iteration += 1

            # log_interval表示记录间隔，即每隔几个输出一次，并将相关数据保存一下
            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} total loss')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_args(self.args)

            # 在进行每个epoch前，清空梯度
            self.optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 用adam优化器更新参数
            self.optimizer.step()

    # 训练采用掩盖标记的模型
    def train_embed(self):
        for t in tqdm(range(self.args.embed_epochs)):
            # 随机获取批量数据
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            # 此处是真的要遮蔽数据
            random_masks = generate_random_masks(self.args, self.args.batch_size)

            # Get the target x_ori_enc by Autoencoder
            self.model.eval()
            # 获取无掩码的编码值
            x_ori_enc, _, masks = self.model(x_ori, self.pseudo_masks, 'train_ae')
            '''    
                clone相当于一个中间量，重新定义了个一个变量，不与原变量共享内存，计算梯度时会叠加到原变量的梯度
                detach与原变量共享内存，相当于赋值，不牵扯梯度计算
            '''
            x_ori_enc = x_ori_enc.clone().detach()  # (bs, ts_size, hidden_dim)
            b, l, f = x_ori_enc.size()

            self.model.train()
            x_enc, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')

            # Only calculate loss for those being masked
            # 只计算那些未被屏蔽的损失
            x_enc_masked = x_enc[masks, :].reshape(b, -1, f)
            x_ori_enc_masked = x_ori_enc[masks, :].reshape(b, -1, f)
            loss = self.criterion(x_enc_masked, x_ori_enc_masked)
            # By annotate lines above, we take loss on all patches
            # loss = self.criterion(x_enc, x_ori_enc)  # embed_loss

            self.num_iteration += 1

            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} loss.')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_args(self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def train_recon(self):
        for t in tqdm(range(self.args.recon_epochs)):
            # 获取批量数据并转为tensor
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            # 将获取的批量数据随机掩盖一些
            random_masks = generate_random_masks(self.args, self.args.batch_size)  # (bs, ts_size)

            self.model.train()
            _, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')
            # 计算通过MAE后的数据和原始数据的全部损失
            loss = self.criterion(x_dec, x_ori)

            self.num_iteration += 1

            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} loss.')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_args(self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    '''
        评估
    '''

    # 评估传统的的自动编码器
    def evaluate_ae(self):
        """Evaluate the model as a simple Anto Encoder"""
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = full_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        # 将数据恢复至最大最小归一化前
        art_data *= self.args.max_val
        art_data += self.args.min_val
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization

        # Visualization 可视化
        plot_time_series_no_masks(self.args)
        # 绘制PCA和t-SNE图
        pca_and_tsne(self.args)

        # Calculate Predictive and Discriminative Scores
        print('Calculating Pred and Disc Scores\n')
        # 计算预测和区别分数
        calculate_pred_disc(self.args)

    # 评估MAE
    def evaluate_random_mae(self):
        """Evaluate the model as a Masked Auto Encoder"""
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = random_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val

        # Save Renormalized art_data
        # 保存生成后的数据
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization
        print('Synthetic Data Generation Finished.')

        # Visualization
        # 显示被掩盖数据的标记
        plot_time_series_with_masks(self.args)
        pca_and_tsne(self.args)

        if self.args.data_name == 'argo':
            argo_plot_time_series_with_masks(self.args, 'storage/Continuous argo/random_mae/')

    # 对交叉掩码生成的数据合并后取这些数据的平均值作为交叉掩盖生成数据进行评估
    def synthesize_cross_average(self):
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        # 通过交叉掩码生成数据
        art_data = cross_average_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val

        # Save Renormalized art_data
        # 创建对应文件夹
        save_dir = os.path.join(self.args.synthesis_dir, 'cross_average')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        art_data_dir = os.path.join(save_dir, 'art_data.npy')
        np.save(art_data_dir, art_data)  # save art_data after renormalization
        print('Synthetic Data Generation by Cross Average Finished.')
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization

        # Visualization
        temp_args = self.args
        temp_args.pics_dir = save_dir
        # 不显示被掩盖数据的标记
        plot_time_series_no_masks(temp_args)
        pca_and_tsne(temp_args)

        if self.args.data_name == 'argo':
            argo_plot_time_series_no_masks(self.args, 'storage/Continuous argo/cross_average/')

    # 对合成的生成数据进行评估
    def synthesize_cross_concate(self):
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = cross_concat_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val

        # Save Renormalized art_data
        save_dir = os.path.join(self.args.synthesis_dir, 'cross_concate')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        art_data_dir = os.path.join(save_dir, 'art_data.npy')
        np.save(art_data_dir, art_data)  # save art_data after renormalization
        print('Synthetic Data Generation by Cross Concate Finished.')
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization

        # Visualization
        temp_args = self.args
        temp_args.pics_dir = save_dir
        # 不显示被掩盖数据的标记
        plot_time_series_no_masks(temp_args)
        pca_and_tsne(temp_args)

        if self.args.data_name == 'argo':
            argo_plot_time_series_no_masks(self.args, 'storage/Continuous argo/cross_concate/')

    def synthesize_random_average(self):
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = random_average_generation(self.args, self.model, ori_data)

        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val

        # Save Renormalized art_data
        save_dir = os.path.join(self.args.synthesis_dir, 'random_average')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        art_data_dir = os.path.join(save_dir, 'art_data.npy')
        np.save(art_data_dir, art_data)  # save art_data after renormalization
        print('Synthetic Data Generation by Random Average Finished.')
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization

        # Visualization
        temp_args = self.args
        temp_args.pics_dir = save_dir
        # 不显示被掩盖数据的标记
        plot_time_series_no_masks(temp_args)
        pca_and_tsne(temp_args)

        if self.args.data_name == 'argo':
            argo_plot_time_series_no_masks(self.args, 'storage/Continuous argo/random_average/')
