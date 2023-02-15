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

import argparse
import struct
import sys
import time
import os


from enum import Enum
from pathlib import Path
from typing import IO, Dict, NamedTuple, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from compressai.zoo import load_state_dict

from PIL import Image
from torch import Tensor
from torch.utils.model_zoo import tqdm
from torchvision.transforms import ToPILImage, ToTensor
from compressai.zoo.image import model_architectures as architectures
from compressai.models import Guided_compresser, Master_compresser  

import compressai

from compressai.datasets import RawVideoSequence, VideoFormat
from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)
from compressai.zoo import image_models, models

torch.backends.cudnn.deterministic = True

# 添加进字典
models.update({'Master_compresser':Master_compresser})
models.update({'Guided_compresser':Guided_compresser})

model_ids = {k: i for i, k in enumerate(models.keys())}

metric_ids = {"mse": 0, "ms-ssim": 1}

NORMALIZE_RGB = 255
NORMALIZE_DEPTH = 10000
IS_DEPTH = False

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]

# 图像、视频、数字
class CodecType(Enum):
    IMAGE_CODEC = 0
    VIDEO_CODEC = 1
    NUM_CODEC_TYPE = 2

# 头部、hw，位深度、模型、cuda/cpu
class CodecInfo(NamedTuple):
    codec_header: Tuple
    original_size: Tuple
    original_bitdepth: int
    net: Dict
    device: str


def BoolConvert(a):
    b = [False, True]
    return b[int(a)]

# 取列表平均值
def Average(lst):
    return sum(lst) / len(lst)

# 如果是单一键值，那么该字典，键值对转换
def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}

# 文件大小
def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

# 读取RGB格式文件
def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath)
    
def my_normalize(image,channel):
    if channel==3 or channel==4:  # 默认已经需要0-255展示,ToTensor会帮助自己归一化
        # image = image /255
        pass
    else:
        image = image / NORMALIZE_DEPTH # 将nyu的depth转换到0-1直接
    return image

def my_denomalize(image,channel):
    # image = image * 255 # 统一转换到0-255展示,但实际上depth是需要乘以10000的
    return image 


# img2torch和torch2img就是在执行归一化和反归一化 
def img2torch(img: Image.Image,device) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0).to(device)

def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def write_floats(fd, values, fmt=">{:d}f"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


# binstr = struct.pack('>2I',*intlist)
# int1,int2 = struct.unpack('>2I',binstr)
def write_floats(fd, values, fmt=">{:d}f"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def read_floats(fd, n, fmt=">{:d}f"):  # n为要读取的float数量
    sz = struct.calcsize("f")
    return struct.unpack(fmt.format(n), fd.read(n * sz)) 

def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

# 写入bytes，返回长度
def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

# 输入：根据模型、metric、quailty，取生成header
# 输出：模型id，code信息（包括：metric和quality合并在一起表示）
def get_header(model_name, metric, quality, num_of_frames, codec_type: Enum):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    - 4 bytes for number of frames to be coded (only applicable for video)
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)

    if codec_type == CodecType.VIDEO_CODEC:
        return model_ids[model_name], code, num_of_frames

    print("get_header",model_name,model_ids[model_name],code)
    return model_ids[model_name], code

# get完，现在parser
# 返回model，metric，quality
def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4
    print("parse_header:",model_id,code)

    return (
        inverse_dict(model_ids)[model_id],
        inverse_dict(metric_ids)[metric],
        quality,
    )

# 读取二进制文件，返回string列表和shape【也就是解码结果】
def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    # if IS_DEPTH:  # 还是说连文件的类型也要写入？这是工程性问题，如果现在是为了验证是否正确，就没有必要写的那么完善。
        # maxdepth = read_uints(fd,1)
        
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])  # TODO: 这个长度似乎太长了，超过了4B 
        lstrings.append([s])

    return lstrings, shape

# 如何写入二进制文件？首先写入以uint写入shape[0], shape[1], len(out_strings)，然后是二维列表的长度；然后以bytes写入列表的值
# 返回总共的字节数
def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    # if not IS_DEPTH:
        # bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    # else:
        # bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings),NORMALIZE_DEPTH))  # 写入maxdepth
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def to_tensors(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_value: int = 1,
    device: str = "cpu",
) -> Frame:
    return tuple(
        torch.from_numpy(np.true_divide(c, max_value, dtype=np.float32)).to(device)
        for c in frame
    )


def convert_yuv420_rgb(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device, max_val: int
) -> Tensor:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    frame = to_tensors(frame, device=str(device), max_value=max_val)
    frame = yuv_420_to_444(
        tuple(c.unsqueeze(0).unsqueeze(0) for c in frame), mode="bicubic"  # type: ignore
    )
    return ycbcr2rgb(frame)  # type: ignore


def convert_rgb_yuv420(frame: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    return yuv_444_to_420(rgb2ycbcr(frame), mode="avg_pool")


def pad(x, p=2**6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )

# output [0,1] 转255/1023
def convert_output(t: Tensor, bitdepth: int = 8) -> np.array:
    assert bitdepth in (8, 10)
    # [0,1] fp ->  [0, 2**bitstream-1] uint
    dtype = np.uint8 if bitdepth == 8 else np.uint16
    t = (t.clamp(0, 1) * (2**bitdepth - 1)).cpu().squeeze()
    arr = t.numpy().astype(dtype)
    return arr


def write_frame(fout: IO[bytes], frame: Frame, bitdepth: np.uint = 8):
    for plane in frame:
        convert_output(plane, bitdepth).tofile(fout)

# 每张图片对应一次encode_image,然后bin写入output文件中
# 输入图像，编码器，输出路径
# 一个文件由header和body组成
# 写入文件的信息包括：模型的id，图像原始尺寸（h，w），使用多少位=8【header】，解码结果【body】
# 返回bpp
def encode_image(input, codec: CodecInfo, output,channel=3):

    if type(codec.net) is list: # 说明是master
        master_chl = channel
        if master_chl == 3:
            guided_chl = 1
            x = Image.open(input).convert('RGB').resize((1280,1024))
            guided = Image.open(input.replace('RGB','thermal_8_bit').replace('jpg','jpeg'))
        else:
            guided_chl = 3
            # x = Image.open(input.replace('thermal_8_bit','RGB').replace('jpeg','jpg'))
            # guided = Image.open(input).convert('RGB').resize((1280,1024))
            x = Image.open(input)
            guided = Image.open(input.replace('thermal_8_bit','RGB').replace('jpeg','jpg')).resize((1280,1024))
        x = img2torch(x,codec.device)
        guided = img2torch(guided,codec.device)  # totensor & unsqueeze
    else:
        x = Image.open(input)
        x = img2torch(x,codec.device)


    bitdepth = 8 # 这个实际上是可以更改的，理论上maxdepth=10000，只需要14位存储

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    # x = pad(x, p)  # 这里的pad是否需要考虑
    

    with torch.no_grad():
        if type(codec.net) is list:
            model_guided = codec.net[0]
            model=codec.net[1]
            # guided = pad(guided,p) 
            out_net_R = model_guided.compress(guided)
            out_dec_R = model_guided.decompress(out_net_R["strings"], out_net_R["shape"])  # todo:这里需要修改waseda.py里面的model文件，compress也是需要借助hidden的【但是这里暂时不重要，因为结果可以通过estimation来进行展示，现在还没有可视化图片的要求】
            out = model.compress(x,out_dec_R["x_hat"])   # 具有依赖关系，需要编解码另外的图片
        else:
            out = codec.net.compress(x)

    shape = out["shape"]

    with Path(output).open("wb") as f:
        write_uchars(f, codec.codec_header)
        # write original image size
        write_uints(f, (h, w))
        # write original bitdepth
        write_uchars(f, (bitdepth,))
        print("original bitdepth:",bitdepth)
        print("original_size:",(h, w))
        # write beta and gamma 
        if type(codec.net) is list:
            write_floats(f,list(torch.squeeze(out['beta'])))
            write_floats(f,list(torch.squeeze(out['gamma'])))
        # write shape and number of encoded latents
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (h * w)
    return {"bpp": bpp}


def encode_video(input, codec: CodecInfo, output):
    if Path(input).suffix != ".yuv":
        raise NotImplementedError(
            f"Unsupported video file extension: {Path(input).suffix}"
        )

    # encode frames of YUV sequence only
    org_seq = RawVideoSequence.from_file(input)
    bitdepth = org_seq.bitdepth
    max_val = 2**bitdepth - 1
    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    num_frames = codec.codec_header[2]
    if num_frames < 0:
        num_frames = org_seq.total_frms

    avg_frame_enc_time = []

    f = Path(output).open("wb")
    with torch.no_grad():
        # Write Video Header
        write_uchars(f, codec.codec_header[0:2])
        # write original image size
        write_uints(f, (org_seq.height, org_seq.width))
        # write original bitdepth
        write_uchars(f, (bitdepth,))
        # write number of coded frames
        write_uints(f, (num_frames,))

        x_ref = None
        with tqdm(total=num_frames) as pbar:
            for i in range(num_frames):
                frm_enc_start = time.time()

                x_cur = convert_yuv420_rgb(org_seq[i], codec.device, max_val)
                h, w = x_cur.size(2), x_cur.size(3)
                p = 128  # maximum 7 strides of 2
                x_cur = pad(x_cur, p)

                if i == 0:
                    x_out, out_info = codec.net.encode_keyframe(x_cur)
                    write_body(f, out_info["shape"], out_info["strings"])
                else:
                    x_out, out_info = codec.net.encode_inter(x_cur, x_ref)
                    for shape, out in zip(
                        out_info["shape"].items(), out_info["strings"].items()
                    ):
                        write_body(f, shape[1], out[1])

                x_ref = x_out.clamp(0, 1)

                avg_frame_enc_time.append((time.time() - frm_enc_start))

                pbar.update(1)

        org_seq.close()
    f.close()

    size = filesize(output)
    bpp = float(size) * 8 / (h * w * num_frames)

    return {"bpp": bpp, "avg_frm_enc_time": np.mean(avg_frame_enc_time)}


def _encode(input, num_of_frames, model,model_path, metric, quality, coder, device, output,channel=3):
    encode_func = {
        CodecType.IMAGE_CODEC: encode_image,
        CodecType.VIDEO_CODEC: encode_video,
    }

    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    start = time.time()

    # 根据model_path来确定model
    if model == 'Guided_compresser':
        net = Guided_compresser(channel=channel).to(device)
        # checkpoint = torch.load(model_path)['state_dict']
        checkpoint = torch.load(model_path[0])
        net.load_state_dict(load_state_dict(checkpoint))
    elif model == 'Master_compresser':
        master_chl = channel
        if master_chl == 3:
            guided_chl = 1
        else:
            guided_chl = 3
        checkpoint1 = torch.load(model_path[0])
        checkpoint2 = torch.load(model_path[1])

        print("guided_chl-model",guided_chl)
        netG = Guided_compresser(channel=guided_chl).to(device)
        netG.load_state_dict(load_state_dict(checkpoint1))  # 仅仅是guided

        net = Master_compresser(width=512,height=640,channel=master_chl).to(device)                   # 在RGBT论文复现中，wh固定
        net.load_state_dict(load_state_dict(checkpoint2))
        net = [netG,net] 

    # net = load_checkpoint(model,model_path,channel).to(device)
    # codec_type = (
    #     CodecType.IMAGE_CODEC if model in image_models else CodecType.VIDEO_CODEC
    # )
    codec_type = CodecType.IMAGE_CODEC

    codec_header_info = get_header(model, metric, quality, num_of_frames, codec_type)
    load_time = time.time() - start

    if not Path(input).is_file():
        raise FileNotFoundError(f"{input} does not exist")

    codec_info = CodecInfo(codec_header_info, None, None, net, device)
    out = encode_func[codec_type](input, codec_info, output,channel)

    enc_time = time.time() - enc_start

    print(
        f"{out['bpp']:.3f} bpp |"
        f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
    )

# 输入解析完的文件，解码图片，然后进行保存
def decode_image(f, codec: CodecInfo, output,inputfile,channel=3):
    if type(codec.net) is list:
        beta=torch.tensor(read_floats(f,64)).reshape(1,64,1,1)
        gamma=torch.tensor(read_floats(f,64)).reshape(1,64,1,1)
    strings, shape = read_body(f)

    with torch.no_grad():
        if type(codec.net) is list:
            out_net = {
                "strings":strings,
                "shape":shape,
                "beta":beta,
                "gamma":gamma
            }

            # 复杂之处：这里仍然需要再次对guided进行压缩和解压缩
            # 通过inputfile和channel来，反推出guided的位置# 还要注意resize
            master_chl = channel
            inputfile = Path(inputfile)
            val_dir = inputfile.parent.parent.parent
            val_name = inputfile.stem
            if master_chl == 3:
                guided_chl = 1
                guided_path = val_dir / "thermal_8_bit20" / (val_name+'.jpeg')
            else:
                guided_chl = 3
                guided_path = val_dir / "RGB20_resize" / (val_name+'.jpg')
            guided = Image.open(guided_path)
            guided = img2torch(guided,codec.device)
            
            model_guided = codec.net[0]
            model=codec.net[1]
            out_net_R = model_guided.compress(guided)
            out_dec_R = model_guided.decompress(out_net_R["strings"], out_net_R["shape"])  # todo:这里需要修改waseda.py里面的model文件，compress也是需要借助hidden的【但是这里暂时不重要，因为结果可以通过estimation来进行展示，现在还没有可视化图片的要求】
            out = model.decompress(out_net,out_dec_R)   # 具有依赖关系，需要编解码另外的图片
        else:
            out = codec.net.decompress(strings, shape)


    x_hat = crop(out["x_hat"], codec.original_size)
    img = torch2img(x_hat)

    img.save(output)
    return {"img": img}


def decode_video(f, codec: CodecInfo, output):
    # read number of coded frames
    num_frames = read_uints(f, 1)[0]

    avg_frame_dec_time = []

    with torch.no_grad():
        x_ref = None
        with tqdm(total=num_frames) as pbar:
            for i in range(num_frames):
                frm_dec_start = time.time()

                if i == 0:
                    strings, shape = read_body(f)
                    x_out = codec.net.decode_keyframe(strings, shape)
                else:
                    mstrings, mshape = read_body(f)
                    rstrings, rshape = read_body(f)
                    inter_strings = {"motion": mstrings, "residual": rstrings}
                    inter_shapes = {"motion": mshape, "residual": rshape}

                    x_out = codec.net.decode_inter(x_ref, inter_strings, inter_shapes)

                x_ref = x_out.clamp(0, 1)

                avg_frame_dec_time.append((time.time() - frm_dec_start))

                x_hat = crop(x_out, codec.original_size)
                img = torch2img(x_hat)

                if output is not None:
                    if Path(output).suffix == ".yuv":
                        rec = convert_rgb_yuv420(x_hat)
                        wopt = "wb" if i == 0 else "ab"
                        with Path(output).open(wopt) as fout:
                            write_frame(fout, rec, codec.original_bitdepth)
                    else:
                        img.save(output)

                pbar.update(1)

    return {"img": img, "avg_frm_dec_time": np.mean(avg_frame_dec_time)}



def load_checkpoint(arch: str, checkpoint_path: str,channel:int) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path))
    return architectures[arch].from_state_dict(state_dict,channel).eval()

# 解码参数：输入headers文件（模型、metric、quality）
def _decode(inputpath, model_path,coder, show, device, output=None,channel=3,model='Guided_compresser'):
    decode_func = {
        CodecType.IMAGE_CODEC: decode_image,
        CodecType.VIDEO_CODEC: decode_video,
    }

    compressai.set_entropy_coder(coder)

    dec_start = time.time()
    with Path(inputpath).open("rb") as f:
        _, metric, quality = parse_header(read_uchars(f, 2))

        original_size = read_uints(f, 2)
        original_bitdepth = read_uchars(f, 1)[0]
        print("original bitdepth:",original_bitdepth)
        print("original_size:",original_size)
        

        start = time.time()
        # net = load_checkpoint(model,model_path,channel).to(device).eval()
        # 视频解码还是图像解码
        # codec_type = (
        #     CodecType.IMAGE_CODEC if model in image_models else CodecType.VIDEO_CODEC
        # )
        codec_type=CodecType.IMAGE_CODEC

        print(model)
        if model == 'Guided_compresser':
            net = Guided_compresser(channel=channel).to(device).eval()
            # checkpoint = torch.load(model_path)['state_dict']
            checkpoint = torch.load(model_path[0])
            net.load_state_dict(load_state_dict(checkpoint))
        elif model == 'Master_compresser':
            master_chl = channel
            if master_chl == 3:
                guided_chl = 1
            else:
                guided_chl = 3
            checkpoint1 = torch.load(model_path[0])
            checkpoint2 = torch.load(model_path[1])

            net = Master_compresser(width=512,height=640,channel=master_chl).to(device).eval()                   # 在RGBT论文复现中，wh固定
            netG = Guided_compresser(channel=guided_chl).to(device).eval()
            netG.load_state_dict(load_state_dict(checkpoint1))  # 仅仅是guided
            net.load_state_dict(load_state_dict(checkpoint2))
            net = [netG,net] 

        load_time = time.time() - start
        print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")

        stream_info = CodecInfo(None, original_size, original_bitdepth, net, device)
        out = decode_func[codec_type](f, stream_info, output,inputpath,channel)

    dec_time = time.time() - dec_start
    print(f"Decoded in {dec_time:.2f}s (model loading: {load_time:.2f}s)")

    if show:
        # For video, only the last frame is shown
        show_image(out["img"])


# plt展示解码图片
def show_image(img: Image.Image):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.title.set_text("Decoded image")
    ax.imshow(img)
    fig.tight_layout()
    plt.show()


# 编码参数：python dataset 
def encode(argv):
    parser = argparse.ArgumentParser(description="Encode image/video to bit-stream")
    parser.add_argument(
        "input",
        type=str,
        help="Input path, the first frame will be encoded with a NN image codec if the input is a raw yuv sequence",
    )
    parser.add_argument(
        "-f",
        "--num_of_frames",
        default=-1,
        type=int,
        help="Number of frames to be coded. -1 will encode all frames of input (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        # choices=models.keys(),
        default=list(models.keys())[0],
        help="NN model to use (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--metric",
        choices=metric_ids.keys(),
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        # choices=list(range(1, 9)),
        type=int,
        default=3,
        help="Quality setting (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", help="Output path (bin file)")  
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--path",required=True,nargs='*',help="checkpoint path")
    parser.add_argument(
        "-ch","--channel", type=int,default=3,help="image channel"
    )
    args = parser.parse_args(argv)
    # 如果没有指定output，那么在该图片路径，修改后缀，保存为bin格式
    # if not args.output:
    #     args.output = Path(Path(args.input).resolve().name).with_suffix(".bin")


    pdir = Path(args.input).parent  
    pname = pdir.stem
    ppdir = pdir.parent
    mid_dir = ppdir / (pname+'_bin')/ f'{args.model}_{args.quality}'
    # 中间结果，存储在父级目录+'_bin'文件夹中
    if not os.path.exists(ppdir / (pname+'_bin')):
        os.mkdir(ppdir / (pname+'_bin'))
    if not os.path.exists(mid_dir):  # 多线程如何避免阻塞？multiprocessing.lock，虽然卡住第一次就不会了
        os.mkdir(mid_dir)

    out_filepath = mid_dir / (Path(args.input).stem+'.bin')
    print(args.input,out_filepath)
    


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"  # 在编码的时候可以使用cuda，但是在解码的时候不能
    _encode(
        args.input,
        args.num_of_frames,
        args.model,
        args.path,
        args.metric,
        args.quality,
        args.coder,
        device,
        out_filepath,
        args.channel
    )

# 解码参数：熵编码器，输出路径，是否使用cuda
# python codec.py decode -o result/img_compressed.png --cuda 
def decode(argv):
    parser = argparse.ArgumentParser(description="Decode bit-stream to image/video")
    parser.add_argument("input", type=str)
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--path",required=True,nargs='*',help="checkpoint path")
    parser.add_argument(
        "-ch","--channel", type=int,default=3,help="image channel"
    )
    parser.add_argument(
        "--model",type=str,default='Guided_compresser',help="model arch"
    )  # 这里是为了文件夹命名方便
    parser.add_argument(
        "-q","--quality", type=int,default=3,help="compressed quality"
    )
    # model，它能够从bin中解码出来
    args = parser.parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"


    # 最后结果，存储在父级目录+'_compressed'文件夹中
    pdir = Path(args.input).parent.parent  # val/  
    pname = pdir.stem
    ppdir = pdir.parent

    final_dir = ppdir / ( pname[:-4] + '_compressed')/ f'{args.model}_{args.quality}'
    # 解压缩的文件
    if not os.path.exists(ppdir / ( pname + '_compressed')):
        os.mkdir(ppdir / ( pname + '_compressed'))
    if not os.path.exists(final_dir):
        os.mkdir(final_dir)
    png_filepath = final_dir / (Path(args.input).stem+'.png')
    print(str(png_filepath))

    print(args.input,png_filepath)
    _decode(args.input,args.path,args.coder, args.show, device, png_filepath,args.channel,args.model)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("command", choices=["encode", "decode"])
    args = parser.parse_args(argv)
    return args


# 传参，进行编码/解码
def main(argv):
    args = parse_args(argv[0:1])
    argv = argv[1:]
    torch.set_num_threads(1)  # just to be sure
    if args.command == "encode":
        encode(argv)
    elif args.command == "decode":
        decode(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
