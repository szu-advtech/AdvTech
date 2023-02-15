import torch

from imageio import imread, imsave
from skimage.transform import resize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import DispNetS
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img") # 保存视差图
parser.add_argument("--output-depth", action='store_true', help="save depth img") # 保存深度图
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path") # 预训练DispNet模型的路径
parser.add_argument("--img-height", default=128, type=int, help="Image height") # 图像的高度
parser.add_argument("--img-width", default=416, type=int, help="Image width") # 图像的宽度
parser.add_argument("--no-resize", action='store_true', help="no resizing is done") # 不重新调整大小

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file") # 数据集列表文件，应该就是指定文件目录，用这些目录下面的数据进行推断
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory") # 数据集目录
parser.add_argument("--output-dir", default='output', type=str, help="Output directory") # 输出目录

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob") # 图像格式

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    disp_net = DispNetS().to(device)
    weights = torch.load(args.pretrained) # 可以看到是通过torch.load加载模型参数的
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval() # 推断模式，关闭BN、Dropout等推断过程不会用到的东西

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p() # 与makedirs()一样都是创建文件夹，但是不会因为文件夹已经存在而产生异常

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()] # 指定了使用哪些测试文件
    else:
        test_files = sum([list(dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):

        img = imread(file)

        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = resize(img, (args.img_height, args.img_width))
        img = np.transpose(img, (2, 0, 1)) # H,W,C -> C,H,W

        tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0) # 增加一个维度
        tensor_img = ((tensor_img - 0.5)/0.5).to(device) # 什么操作？

        output = disp_net(tensor_img)[0] # 送入disp_net获得输出

        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        file_name = '-'.join(file_path.splitall()[1:])

        if args.output_disp:
            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1,2,0)))
        if args.output_depth:
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
            imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0)))


if __name__ == '__main__':
    main()
