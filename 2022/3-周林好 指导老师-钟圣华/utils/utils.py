import sys
from operator import itemgetter

import cv2
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict, deque
import datetime
import pickle
import time

import torch
import torch.distributed as dist

import errno
import os


# -----------------------------#
#   计算原始输入图像
#   每一次缩放的比例
# -----------------------------#
def calculateScales(img):
    pr_scale = 1.0
    h, w, _ = img.shape

    # --------------------------------------------#
    #   将最大的图像大小进行一个固定
    #   如果图像的短边大于500，则将短边固定为500
    #   如果图像的长边小于500，则将长边固定为500
    # --------------------------------------------#
    if min(w, h) > 500:
        pr_scale = 500.0 / min(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)
    elif max(w, h) < 500:
        pr_scale = 500.0 / max(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    # ------------------------------------------------#
    #   建立图像金字塔的scales，防止图像的宽高小于12
    # ------------------------------------------------#
    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h, w)
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales


# -----------------------------#
#   将长方形调整为正方形
# -----------------------------#
def rect2square(rectangles):
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangles


# -------------------------------------#
#   非极大抑制
# -------------------------------------#
def NMS(rectangles, threshold):
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())
    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    # -------------------------------------#
    #   计算特征点之间的步长
    # -------------------------------------#
    stride = 0
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)

    # -------------------------------------#
    #   获得满足得分门限的特征点的坐标
    # -------------------------------------#
    (y, x) = np.where(cls_prob >= threshold)

    # -----------------------------------------#
    #   获得满足得分门限的特征点得分
    #   最终获得的score的shape为：[num_box, 1]
    # -------------------------------------------#
    score = np.expand_dims(cls_prob[y, x], -1)

    # -------------------------------------------------------#
    #   将对应的特征点的坐标转换成位于原图上的先验框的坐标
    #   利用回归网络的预测结果对先验框的左上角与右下角进行调整
    #   获得对应的粗略预测框
    #   最终获得的boundingbox的shape为：[num_box, 4]
    # -------------------------------------------------------#
    boundingbox = np.concatenate([np.expand_dims(x, -1), np.expand_dims(y, -1)], axis=-1)
    top_left = np.fix(stride * boundingbox + 0)
    bottom_right = np.fix(stride * boundingbox + 11)
    boundingbox = np.concatenate((top_left, bottom_right), axis=-1)
    boundingbox = (boundingbox + roi[y, x] * 12.0) * scale

    # -------------------------------------------------------#
    #   将预测框和得分进行堆叠，并转换成正方形
    #   最终获得的rectangles的shape为：[num_box, 5]
    # -------------------------------------------------------#
    rectangles = np.concatenate((boundingbox, score), axis=-1)
    rectangles = rect2square(rectangles)

    rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)
    rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
    return rectangles


# -------------------------------------#
#   对Rnet处理后的结果进行处理
# -------------------------------------#
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    # -------------------------------------#
    #   利用得分进行筛选
    # -------------------------------------#
    pick = cls_prob[:, 1] >= threshold

    score = cls_prob[pick, 1:2]
    rectangles = rectangles[pick, :4]
    roi = roi[pick, :]

    # -------------------------------------------------------#
    #   利用Rnet网络的预测结果对粗略预测框进行调整
    #   最终获得的rectangles的shape为：[num_box, 4]
    # -------------------------------------------------------#
    w = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)
    h = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)
    rectangles[:, [0, 2]] = rectangles[:, [0, 2]] + roi[:, [0, 2]] * w
    rectangles[:, [1, 3]] = rectangles[:, [1, 3]] + roi[:, [1, 3]] * w

    # -------------------------------------------------------#
    #   将预测框和得分进行堆叠，并转换成正方形
    #   最终获得的rectangles的shape为：[num_box, 5]
    # -------------------------------------------------------#
    rectangles = np.concatenate((rectangles, score), axis=-1)
    rectangles = rect2square(rectangles)

    rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)
    rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
    return np.array(NMS(rectangles, 0.7))


# -------------------------------------#
#   对onet处理后的结果进行处理
# -------------------------------------#
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    # -------------------------------------#
    #   利用得分进行筛选
    # -------------------------------------#
    pick = cls_prob[:, 1] >= threshold

    score = cls_prob[pick, 1:2]
    rectangles = rectangles[pick, :4]
    pts = pts[pick, :]
    roi = roi[pick, :]

    w = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)
    h = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)
    # -------------------------------------------------------#
    #   利用Onet网络的预测结果对预测框进行调整
    #   通过解码获得人脸关键点与预测框的坐标
    #   最终获得的face_marks的shape为：[num_box, 10]
    #   最终获得的rectangles的shape为：[num_box, 4]
    # -------------------------------------------------------#
    face_marks = np.zeros_like(pts)
    face_marks[:, [0, 2, 4, 6, 8]] = w * pts[:, [0, 1, 2, 3, 4]] + rectangles[:, 0:1]
    face_marks[:, [1, 3, 5, 7, 9]] = h * pts[:, [5, 6, 7, 8, 9]] + rectangles[:, 1:2]
    rectangles[:, [0, 2]] = rectangles[:, [0, 2]] + roi[:, [0, 2]] * w
    rectangles[:, [1, 3]] = rectangles[:, [1, 3]] + roi[:, [1, 3]] * w
    # -------------------------------------------------------#
    #   将预测框和得分进行堆叠
    #   最终获得的rectangles的shape为：[num_box, 15]
    # -------------------------------------------------------#
    rectangles = np.concatenate((rectangles, score, face_marks), axis=-1)

    rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)
    rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
    return np.array(NMS(rectangles, 0.3))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
