import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='/home/jianjian/yangxunyu/yzz2/LSGAN/dataset/img_align_celeba_resize/images')
    parser.add_argument('--dataset', default='dataset.npy', help='path of dataset')


    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--ncritic', type=int, default=5, help='WGAN-number of critic parameter per generator iteration')
    parser.add_argument('--clip', type=float, default=0.01, help='WGAN-weight clipping')
    parser.add_argument('--lambda_gp', type=int, default=10, help='WGAN_GP-gradient penalty parameter')
    parser.add_argument('--num_label', type=int, default=2)
    parser.add_argument('--checkpoints', default='./data/img_align_celeba', help='folder to output images and model checkpoints')
    return parser