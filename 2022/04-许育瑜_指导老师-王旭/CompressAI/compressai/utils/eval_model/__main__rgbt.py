# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time
import struct

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from compressai.datasets.image_rgbt_test import ImageFolderTest
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torch.utils.data import DataLoader
import compressai

from compressai.zoo import image_models as pretrained_models
from compressai.zoo import load_state_dict
from compressai.zoo.image import model_architectures as architectures
import torchsnooper
from compressai.models.waseda import Cheng2020Attention_R,Cheng2020Attention_D
from compressai.models import Guided_compresser, Master_compresser  

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    #img = Image.open(filepath).convert("RGB")
    img = Image.open(filepath)
    return transforms.ToTensor()(img)


@torch.no_grad()  # TODO：暂未改动
def inference(model,model_guided, x,guided):
    # unsuqeeze成可以输入网络的张量
    print('inference_entropy_estimation')

    if len(x.shape)  == 3:
        x = x.unsqueeze(0)
        guided = guided.unsqueeze(0)


    ## 不使用padding，模块暂不支持不同尺寸输入
    # h, w = x.size(2), x.size(3)
    # p = 64  # maximum 6 strides of 2
    # new_h = (h + p - 1) // p * p
    # new_w = (w + p - 1) // p * p
    # padding_left = (new_w - w) // 2
    # padding_right = new_w - w - padding_left
    # padding_top = (new_h - h) // 2
    # padding_bottom = new_h - h - padding_top
    # x_padded = F.pad(
    #     x,
    #     (padding_left, padding_right, padding_top, padding_bottom),
    #     mode="constant",
    #     value=0,
    # )
    
    ### compress ##############################################
    start = time.time()
    out_net_R = model_guided.compress(guided)
    out_dec_R = model_guided.decompress(out_net_R["strings"], out_net_R["shape"])  # todo:这里需要修改waseda.py里面的model文件，compress也是需要借助hidden的【但是这里暂时不重要，因为结果可以通过estimation来进行展示，现在还没有可视化图片的要求】

    out_net = model.compress(x,out_dec_R["x_hat"])   # 具有依赖关系，需要编解码另外的图片
    enc_time = time.time() - start
    ### compress ##############################################
    

    ### decompress ##############################################
    start = time.time()
    out_dec = model.decompress(out_net,out_dec_R)
    dec_time = time.time() - start
    ### decompress ##############################################
    

    num_pixels = x.size(0) * x.size(2) * x.size(3) 
    bpp = (sum(len(s[0]) for s in out_net["strings"]) * 8.0 + 64*2*4*8) / num_pixels  # 512B 是beta和gamma

    return {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),  # 只有在inference的时候才计算ms-ssim,因为计算时间也比较久
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
# @torchsnooper.snoop()
def inference_entropy_estimation(model,model_guided, x,guided):
    print('inference_entropy_estimation')

    ### forward ####################################################
    start = time.time()
    out_net_R = model_guided(guided)
    hidden = out_net_R['hidden']
    out_net = model(x,guided,hidden) 
    elapsed_time = time.time() - start
    ### forward ####################################################
    
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    # print(x.size())
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr": psnr(x, out_net["x_hat"]),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


def load_checkpoint(arch: str, checkpoint_path: str,channel:int) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path))
    return architectures[arch].from_state_dict(state_dict,channel=channel).eval()



def eval_model(model,model_R,test_dataloader, entropy_estimation=True, half=False, arch='mbt2018-mean',quality=3):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    
    num = 0
    for i, (x,guided) in enumerate(test_dataloader):
        # print("eval:",x.shape,guided.shape)
        x = x.to(device)
        guided = guided.to(device)
            
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model,model_R, x, guided)
        else:
            rv = inference_entropy_estimation(model, model_R,x,guided)
        for k, v in rv.items():
            print(k,v)
            metrics[k] += v
        num +=1
            
            
    # 计算均值
    for k, v in metrics.items():
        metrics[k] = v / num
    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        # choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    # parent_parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # cuda和cpu
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-ch",
        "--channel",
        type=int,
        help="image channel",
    )

    parent_parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )



    parent_parser.add_argument( 
        "--patch-size", # 这里的-也是能和_匹配的
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",  # 可以传入多个参数的意思  nargs=N，一个选项后可以跟多个参数（action='append'时，依然是一个选项后跟一个参数，只不过选项可以多次出现），参数的个数必须为N的值，这些参数会生成一个列表，当nargs=1时，会生成一个长度为1的列表。  如果不是append模式，那么中间使用空格间隔开。
        required=True,
        help="checkpoint path",
    )
    
    checkpoint_parser.add_argument(
        "-q",
        "--quality",
        dest='quality',
        type=int,
        help="quality",
    )
    

    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.source:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    test_dataset = ImageFolderTest(args.dataset, size=args.patch_size,channel=args.channel)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        pin_memory=("cuda"),
    )

    compressai.set_entropy_coder(args.entropy_coder)
    master_chl = 3 
    guided_chl = 1 
    if args.channel == 1:
        master_chl = 1
        guided_chl = 3
    netGuided = Guided_compresser(channel=guided_chl)
    net = Master_compresser(width=512,height=640,channel = master_chl).eval() # 读入图片的size需要变化？怎么处理
    
        
    # 注意传入checkpoint的时候，需要先rgb，再depth
    print("Loading checkpoint", args.paths[0])
    state_dict = load_state_dict(torch.load(args.paths[0])) # eval的load_state=load_pretrained,是有包括更新名称的
    netGuided.load_state_dict(state_dict)  # 要注意，有些函数是有返回值的，有些函数是自己本身更新的，再比如：list.sort()就是更新自身

    print("Loading checkpoint", args.paths[1])
    state_dict = load_state_dict(torch.load(args.paths[1])) # eval的load_state=load_pretrained,是有包括更新名称的
    net.load_state_dict(state_dict)  # 要注意，有些函数是有返回值的，有些函数是自己本身更新的，再比如：list.sort()就是更新自身
    
    if args.cuda and torch.cuda.is_available() and not args.entropy_estimation:
        net = net.to('cuda')
        netGuided = netGuided.to('cuda')
    print("args.entropy_estimation",args.entropy_estimation)
    metrics = eval_model(net,netGuided, test_dataloader, args.entropy_estimation, args.half, args.architecture, args.quality)
    results = defaultdict(list)
    for k, v in metrics.items():
        results[k].append(v)
    
    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": args.architecture,
        "description": f"Inference ({description})",
        "results": results,
    }
    
    result_name = os.path.join(os.path.dirname(args.dataset),os.path.basename(args.dataset).replace('/',''))+'_result'
    if not os.path.exists(result_name):
        os.mkdir(result_name)

    if args.entropy_estimation:
        eval_name=result_name+'/eval_'+args.architecture+'.json'
    else:
        eval_name=result_name+'/ans_eval_'+args.architecture+'.json'
    with open(eval_name,'a') as file:  # 所有的点都描绘完了，可以导出或者直接使用plot绘制
        file.write(json.dumps(output)+'\n')  # 分行是为了分别读取
    print(json.dumps(output, indent=2))
    
if __name__ == "__main__":
    main(sys.argv[1:])

