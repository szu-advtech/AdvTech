import math

import numpy as np
import torch


# 选择channel
def channel_selection(inputs, module, device, sparsity=0.5):

    inputs = inputs.to(device)

    num_channel = inputs.size(1)
    # print(inputs.size())
    # print(num_channel)
    # print(module)
    num_pruned = int(math.ceil(num_channel * sparsity))

    # 蛮力法
    # 实际裁剪中将 输入channel赋0 代替 对应的filter赋0
    indices_pruned = []
    while len(indices_pruned) < num_pruned:
        min_diff = 1e10
        min_idx = 0
        for idx in range(num_channel):
            if idx in indices_pruned:
                continue
            # 当前的T数组
            temp_indices = indices_pruned + [idx]
            # 创建输入
            temp_inputs = torch.zeros_like(inputs)
            # ... 代表后面的维度全选
            temp_inputs[:, temp_indices, ...] = inputs[:, temp_indices, ...]
            # 公式6
            temp_output = module(temp_inputs).norm(2)
            if temp_output < min_diff:
                min_diff = temp_output
                min_idx = idx
        indices_pruned.append(min_idx)

    # 得到要保留的channel编号 并返回
    indices_stayed = list(set([i for i in range(num_channel)]) - set(indices_pruned))
    return indices_stayed, indices_pruned


# 裁剪channel
def channel_change(module, BN_module, Next_module, indices_stayed):
    num_channels_stayed = int(len(indices_stayed))

    # 对当前模块进行裁剪，修改输出channel数量
    if module is not None:
        if isinstance(module, torch.nn.Conv2d):
            module.out_channels = num_channels_stayed
        elif isinstance(module, torch.nn.Linear):
            module.out_features = num_channels_stayed

        # 修改模块的weight
        temp1_weight = module.weight[indices_stayed, ...].clone()
        del module.weight
        module.weight = torch.nn.Parameter(temp1_weight)

        # 对bias进行同样的操作
        if module.bias is not None:
            temp1_bias = module.bias[indices_stayed, ...].clone()
            del module.bias
            module.bias = torch.nn.Parameter(temp1_bias)

    # 修改BN模块
    if BN_module is not None:
        if isinstance(BN_module, torch.nn.modules.BatchNorm2d):
            BN_module.num_features = num_channels_stayed

            running_mean = BN_module.running_mean[indices_stayed, ...].clone()
            del BN_module.running_mean
            BN_module.running_mean = running_mean

            running_var = BN_module.running_var[indices_stayed, ...].clone()
            del BN_module.running_var
            BN_module.running_var = running_var

            temp2_weight = BN_module.weight[indices_stayed, ...].clone()
            del BN_module.weight
            BN_module.weight = torch.nn.Parameter(temp2_weight)

            temp2_bias = BN_module.bias[indices_stayed, ...].clone()
            del BN_module.bias
            BN_module.bias = torch.nn.Parameter(temp2_bias)

    # 修改下一层的输入channel
    if Next_module is not None:
        if isinstance(Next_module, torch.nn.Conv2d):
            Next_module.in_channels = num_channels_stayed
        elif isinstance(Next_module, torch.nn.Linear):
            Next_module.in_features = num_channels_stayed
        # 删除下一层的通道
        temp3_weight = Next_module.weight[:, indices_stayed, ...].clone()
        del Next_module.weight
        Next_module.weight = torch.nn.Parameter(temp3_weight)


# 最小化重构误差
def reconstruction_errors(module, pruned_outputs, outputs):

    device = torch.device('cuda')

    # inputs = inputs.cuda()
    # module = module.cuda()
    # 按照现在的channel数修剪原本的输出
    if module.bias is not None:
        bias_size = [1] * outputs.dim()
        bias_size[1] = -1
        outputs -= module.bias.view(bias_size)

    # Conv2d是Unfold + matmul + fold
    # 从一个批次的输入张量中提取出滑动的局部区域块
    if isinstance(module, torch.nn.Conv2d):
        unfold = torch.nn.Unfold(kernel_size=module.kernel_size, dilation=module.dilation,
                                 padding=module.padding, stride=module.stride)
    # print(unfold)
    # print(module)
    unfold.eval()
    # 将一个批次的输入展开为三维矩阵(batch_size, in_channel×kernel_size, num of kernals in one data)
    x = unfold(pruned_outputs)
    # 调换顺序(1, in_channel×kernel_size, num of kernals in one data)->(batch_size, num of kernals in one data,
    # in_channel×kernel_size)
    x = x.transpose(1, 2)
    # 这个batch一共多少kernal
    batch_kernal = x.size(0) * x.size(1)
    # 大小为(batch×num of kernals in one data, channel×kernel_size)
    x = x.reshape(batch_kernal, -1)
    # 将结果转化为(1,out_channel, W*H)
    y = outputs.view(outputs.size(0), outputs.size(1), -1)
    # 调换顺序(1,out_channel, W*H)->(1,w*H, out_channel)
    y = y.transpose(1, 2)
    # 大小为(w*H, out_channel)
    y = y.reshape(-1, y.size(2))

    x, y = x.cpu(), y.cpu()

    # 利用最小二乘法求解weight
    # 返回的是：解决方案、残差平方和、矩阵的秩、奇异值
    param, _, _, _ = np.linalg.lstsq(x.detach().cpu().numpy(), y.detach().cpu().numpy(), rcond=-1)

    param = torch.from_numpy(param).cuda()
    # contiguous方法改变了多维数组在内存中的存储顺序，以便配合view方法使用
    # 进行选择
    param = param[0:x.size(1), :].clone().t().contiguous().view(y.size(1), -1)
    if isinstance(module, torch.nn.Conv2d):
        temp_param = param.view(module.out_channels, module.in_channels, *module.kernel_size)
    del module.weight
    module.weight = torch.nn.Parameter(temp_param)
