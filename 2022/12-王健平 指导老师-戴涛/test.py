import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from PIL import Image
from models import FSRCNN,edgeSR_MAX,edgeSR_CNN,edgeSR_TM,edgeSR_TR,ESPCN,edgeSR_TR_ECBSR
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms.functional as TF
from PIL.Image import Resampling
import platform
import time
import pickle
import os
from skimage.metrics import structural_similarity as ssim

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [
        '.png', '.tif', '.jpg', '.jpeg', '.bmp', '.pgm', '.PNG'
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ############################### 必填 ###############################
    """网络模型"""
    # FSRCNN、edgeSR_MAX、edgeSR_CNN
    parser.add_argument('--net', type=str, required=True)
    """输出文件夹"""
    parser.add_argument('--outputs-dir', type=str, required=True)
    """训练好的模型"""
    parser.add_argument('--weights-file', type=str, required=True)
    """gpu选择"""
    # 0到n-1
    parser.add_argument('--gpu-id', type=int, required=True)
    # """是否评估图像质量"""
    # parser.add_argument('--eval', type=int, required=True)
    # """是否输出预测图像"""
    # parser.add_argument('--outputs-img', type=int, required=True)
    # ############################### 选填 ###############################
    """推理图片所在文件夹"""
    parser.add_argument('--image-dir-test', type=str)
    """评估图片所在文件夹"""
    parser.add_argument('--image-dir-eval', type=str)
    """倍率"""
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()
    # 网络输入的数据维度或类型变化不大时，cuDNN的auto-tunner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    cudnn.benchmark = True
    """设备选择"""
    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')
    """输出推理图片和评估数据的文件夹初始化"""
    args.outputs_dir = os.path.join(args.outputs_dir, 'output')
    # 输出文件夹初始化
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    """网络选择"""
    # ##############################    FSRCNN   #############################
    if args.net == 'FSRCNN':
        model = FSRCNN(args.scale).to(device)
    # ############################   edgeSR   ###############################
    # ############################ edgeSR_MAX ###############################
    elif args.net == 'edgeSR_MAX':
        model = edgeSR_MAX(args.scale).to(device)
    # ############################ edgeSR_TM ###############################
    elif args.net == 'edgeSR_TM':
        model = edgeSR_TM(args.scale).to(device)
    # ############################ edgeSR_TR ###############################
    elif args.net == 'edgeSR_TR':
        model = edgeSR_TR(args.scale).to(device)
    # ########################    edgeSR_TR_ECBSR    ##########################
    elif args.net == 'edgeSR_TR_ECBSR':
        model = edgeSR_TR_ECBSR(args.scale, flag=2).to(device)
    # ############################ edgeSR_CNN ###############################
    elif args.net == 'edgeSR_CNN':
        model = edgeSR_CNN(args.scale).to(device)
    # ######################################################################
    # ############################ ESPCN ###############################
    elif args.net == 'ESPCN':
        model = ESPCN(args.scale).to(device)
    else:
        raise Exception(f'没有对应模型{args.net}')

    # 模型字典
    model.load_state_dict(
        torch.load(args.weights_file, map_location=lambda storage, loc: storage),
        # 加载全部或者部分参数
        strict=True
    )
    # 将数据映射为半精度浮点类型
    model.to(device)
    # model.to(device)

    model.eval()

    # ##########################   推理图片   ########################
    if args.image_dir_test:
        # 批量处理推理图片
        input_list = [
            str(f) for f in Path(args.image_dir_test).iterdir() if is_image_file(f.name)
        ]
        for input_file in tqdm(input_list):
            # 获取图片名字及格式
            if 'Windows' == platform.system():
                img_name, img_type = input_file.split('\\')[-1].split('.')
            elif 'Linux' == platform.system():
                img_name, img_type = input_file.split('/')[-1].split('.')
            # 读取图片
            y, cb, cr = Image.open(input_file).convert('YCbCr').split()
            # 只取y 加快训练
            input_tensor = TF.to_tensor(
                y
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                output_img = model(
                    input_tensor
                )
            # 从y通道合并，还原hr图片并保存
            output_img_y = (output_img.cpu().data[0].numpy() * 255.).clip(0, 255)
            output_img_y = Image.fromarray(np.uint8(output_img_y[0]), mode='L')
            out_img_cb = cb.resize(output_img_y.size, Resampling.BICUBIC)
            out_img_cr = cr.resize(output_img_y.size, Resampling.BICUBIC)
            Image.merge('YCbCr', [output_img_y, out_img_cb, out_img_cr]).convert('RGB').save(
                args.outputs_dir + '/{}_{}_x{}.{}'.format(img_name, args.net, args.scale, img_type)
            )
            image = pil_image.open(input_file).convert('RGB')
            hr_bicubic = image.resize(output_img_y.size, Resampling.BICUBIC)
            hr_bicubic.save(args.outputs_dir + '/{}_bicubic_x{}.{}'.format(img_name, args.scale, img_type))

    # ##########################   评估模型   ########################
    if args.image_dir_eval:
        # 批量处理评估图片
        input_list = [
            str(f) for f in Path(args.image_dir_eval).iterdir() if is_image_file(f.name)
        ]
        # 获取测试集名字
        if 'Windows' == platform.system():
            eval_name = args.image_dir_eval.split('\\')[-1]
        elif 'Linux' == platform.system():
            eval_name = args.image_dir_eval.split('/')[-1]
        psnr_list = []
        img_list = []
        time_list = []
        ssim_list = []
        # 将参数和cuda初始化提前完成
        with torch.no_grad():
            preds = model(torch.zeros((1, 1, 1, 1)).to(device)).clamp(0.0, 1.0)
        for input_file in tqdm(input_list):
            # 获取图片名字及格式
            if 'Windows' == platform.system():
                img_name, img_type = input_file.split('\\')[-1].split('.')
            elif 'Linux' == platform.system():
                img_name, img_type = input_file.split('/')[-1].split('.')
            image = pil_image.open(input_file).convert('RGB')
            image_width = (image.width // args.scale) * args.scale
            image_height = (image.height // args.scale) * args.scale
            hr = image.resize((image_width, image_height), Resampling.BICUBIC)
            lr = hr.resize((hr.width // args.scale, hr.height // args.scale), Resampling.BICUBIC)
            bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), Resampling.BICUBIC)
            lr, _ = preprocess(lr, device)
            hr, _ = preprocess(hr, device)
            _, ycbcr = preprocess(bicubic, device)
            # lr = lr
            # hr = hr
            with torch.no_grad():
                start_time = time.perf_counter()
                preds = model(lr).clamp(0.0, 1.0)
                end_time = time.perf_counter()
            psnr = calc_psnr(hr, preds)
            psnr_list.append(psnr)
            img_list.append(img_name)
            time_list.append(end_time - start_time)
            ssim_list.append(ssim(hr.cpu().squeeze(0).squeeze(0).numpy(), preds.cpu().squeeze(0).squeeze(0).numpy()))


        for img_n, psnr, s, t in zip(img_list, psnr_list, ssim_list, time_list):
            print('{} PSNR: {:.2f}, ssim: {:.3f} , time: {:.4f}'.format(img_n, float(psnr), s, t))
        print('{}_psnr: {:.3f}'.format(eval_name, float(sum(psnr_list) / len(psnr_list))))
        print('{}_ssim: {:.2f}'.format(eval_name, float(sum(ssim_list) / len(ssim_list))))
        print('infer_speed: {:.2f}ms'.format(sum(time_list) * 1000))

        # 评估结果写入json文件
        if not os.path.exists('tests.pkl'):
            with open('tests.pkl', 'wb') as f:
                pickle.dump({}, f)
        tests_dict = pickle.load(open('tests.pkl', 'rb'))
        input_dict = {
            'psnr': float(sum(psnr_list) / len(psnr_list)),
            'ssim': float(sum(ssim_list) / len(ssim_list)),
            'infer_speed(ms)': sum(time_list) * 1000
        }
        tests_dict['{}_x{}'.format(args.net, args.scale)][eval_name] = input_dict
        pickle.dump(tests_dict, open('tests.pkl', 'wb'))








