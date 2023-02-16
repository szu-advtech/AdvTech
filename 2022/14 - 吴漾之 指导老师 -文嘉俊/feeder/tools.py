import numpy as np
import random
# 数据增强模块

def shear(data_numpy, r=0.5):
    """

    其中 a12, a13, a21, a23, a31, a32 是从 [−r, r] 中随机采样的剪切因子。 r 是剪切幅度。
    序列 x 与通道维度上的变换矩阵 R 相乘。然后，3D 坐标中的人体姿势以随机角度倾斜。

    """
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    """
    R = np.array([[1,          s1_list[1], s1_list[2]],
                  [s1_list[0], 1,          s2_list[2]],
                  [s2_list[0], s2_list[1], 1        ]]) 
    """
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)  # numpy.transpose中的数组是坐标轴标号，即交换坐标轴
    data_numpy = data_numpy.transpose(3, 0, 1, 2)  # 最后得到角度变换的人体姿势参数
    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    """

    裁剪是对时间维度的增强，它对称地将一些帧填充到序列中，然后随机裁剪到原始长度。填充长度定义为 T /γ，γ 表示为填充比。
    temperal_padding_ratio即为填充比，填充操作使用原始边界的反射。

    """
    C, T, V, M = data_numpy.shape  # 关节特征、关键帧数量、关节数量、一帧中的人数，shape返回的是对应维度元素的个数
    padding_len = T // temperal_padding_ratio  # //”是一个算术运算符，表示整数除法，它可以返回商的整数部分（向下取整）。
    frame_start = np.random.randint(0, padding_len * 2 + 1)  # 返回一个随机整型数在[0,padding_len * 2 + 1]之间
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)  # 沿着视频帧连接，对称填充
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy
