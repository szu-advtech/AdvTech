import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    # 数据集位置
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    # 数据集类型
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nsvf', 'colmap', 'rffr'],
                        help='which dataset to train/test')
    # 训练 测试
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval'],
                        help='use which split to train')
    # 下采样
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')


    # model parameters
    # 场景比例
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')

    # training options
    # 每个批次的射线数量
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    # 单图像采样射线还是全部图像采样射线
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    # 训练次数
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    # 使用gpu核数
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')

    # 是否使用随机bg颜色(仅真实数据集)进行训练，以避免被预测为透明的黑色对象
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real dataset only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # 评估lpips度量(消耗更多VRAM)
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    # 是否只验证
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    # 是否保存测试图片和视频
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # 实验名称
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    # 模型权重信息等保存位置（包括optimizers）
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    # 模型权重信息等保存位置（不包括optimizers）
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    parser.add_argument('--lambda_trans_depth_smoothness', type=float, default=0.1,
                        help='lambda_trans_depth_smoothness')

    parser.add_argument('--lambda_refl_bdc', type=float, default=1e-4,
                        help='lambda_refl_bdc')

    parser.add_argument('--lambda_refl_depth_smoothness', type=float, default=1e-10,
                        help='lambda_refl_depth_smoothness')

    parser.add_argument('--lambda_beta_smoothness', type=float, default=1e-4,
                        help='lambda_beta_smoothness')

    parser.add_argument('--lambda_trans_lowpass', type=float, default=0.01,
                        help='lambda_trans_lowpass')

    parser.add_argument('--lambda_coarse_dine_align', type=float, default=0.1,
                        help='lambda_coarse_dine_align')

    parser.add_argument('--lambda_beta_mask', type=float, default=0.1,
                        help='lambda_beta_mask')

    parser.add_argument('--lambda_mse', type=float, default=1.0,
                        help='lambda_mse')

    parser.add_argument('--lambda_schedulers', type=str,
                        default="refl_bdc@step@10:0.05:12:1e-4:15:0;refl_depth_smoothness@step@15:0.01;beta_mask@step@10:0",
                        help='lambda_schedulers')

    parser.add_argument('--lambda_schedulers_step', type=str,
                        default="trans_depth_smoothness@step@1000:0.1:5000:0.01;trans_lowpass@step@1000:0;",
                        help='lambda_schedulers_step')

    return parser.parse_args()
