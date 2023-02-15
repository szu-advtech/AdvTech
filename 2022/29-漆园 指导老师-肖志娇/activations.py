#激活函数的上下界

from numba import njit
import numpy as np

"""
激活函数：tanh, sigmoid, atan
"""

# tanh
@njit
def tanh(x):
    return np.tanh(x)   # tanh(x) = (1 - e^-2x) / (1 + e^-2x)

# (tanh(x))'
@njit
def tanh_d(x):
    return 1.0 / np.cosh(x) ** 2    # cosh(x) = ch(x) = (e^x + e^-x) / 2

# 二分法，作斜率为k的切线（上界），切点point d的横坐标
@njit
def tanh_ku(l, u, k):
    upper = u
    lower = l
    for i in range(20):
        d = (upper + lower) / 2
        if tanh_d(d) <= k:
            upper = d
        else:
            lower = d
    return d

# 二分法，作斜率为k的切线（下界），切点point d的横坐标
@njit
def tanh_kl(l, u, k):
    upper = u
    lower = l
    for i in range(20):
        d = (upper + lower) / 2
        if tanh_d(d) <= k:
            lower = d
        else:
            upper = d
    return d

# 二分法，过l作切线，切点point d的横坐标
@njit
def tanh_du(l, u):
    upper = u   # point d的范围
    lower = 0   # point d的范围
    for i in range(20):
        d = (upper + lower) / 2
        # 过 (d,y(d)) 切线的斜率 >= (l,y(l)) 和 (d,y(d)) 连线的斜率
        if tanh_d(d) <= (tanh(d) - tanh(l)) / (d - l):
            upper = d
        else:
            lower = d
    return d

# 二分法，过u作切线，切点point d的横坐标
@njit
def tanh_dl(l, u):
    upper = 0
    lower = l
    for i in range(20):
        d = (upper + lower) / 2
        if tanh_d(d) <= (tanh(u) - tanh(d)) / (u - d):
            lower = d
        else:
            upper = d
    return d

# sigmoid
@njit
def sigmoid(x):
    return 1.0 / (1.0+np.exp(-x))

# (sigmoid(x))'
@njit
def sigmoid_d(x):
    return np.exp(-x) / (1.0+np.exp(-x))**2

# 二分法，作斜率为k的切线（上界），切点point d的横坐标
@njit
def sigmoid_ku(l, u, k):
    upper = u
    lower = l
    for i in range(20):
        d = (upper + lower) / 2
        if sigmoid_d(d) <= k:
            upper = d
        else:
            lower = d
    return d

# 二分法，作斜率为k的切线（下界），切点point d的横坐标
@njit
def sigmoid_kl(l, u, k):
    upper = u
    lower = l
    for i in range(20):
        d = (upper + lower) / 2
        if sigmoid_d(d) <= k:
            lower = d
        else:
            upper = d
    return d

# 二分法，过l作切线，切点point d的横坐标
def sigmoid_du(l, u):
    upper = u
    lower = 0
    for i in range(20):
        d = (upper + lower) / 2
        if (sigmoid(d) - sigmoid(l)) / (d - l) >= sigmoid_d(d):
            upper = d
        else:
            lower = d
    return d

# 二分法，过u作切线，切点point d的横坐标
@njit
def sigmoid_dl(l, u):
    upper = 0
    lower = l
    for i in range(20):
        d = (upper + lower) / 2
        if (sigmoid(u) - sigmoid(d)) / (u - d) >= sigmoid_d(d):
            lower = d
        else:
            upper = d
    return d

# atan
@njit
def atan(x):
    return np.arctan(x)

# (arctan(x))'
@njit
def atan_d(x):
    return 1.0 / (1.0+x**2)

# 二分法，作斜率为k的切线（上界），切点point d的横坐标
@njit
def atan_ku(l, u, k):
    upper = u
    lower = l
    for i in range(20):
        d = (upper + lower) / 2
        if atan_d(d) <= k:
            upper = d
        else:
            lower = d
    return d

# 二分法，作斜率为k的切线（下界），切点point d的横坐标
@njit
def atan_kl(l, u, k):
    upper = u
    lower = l
    for i in range(20):
        d = (upper + lower) / 2
        if atan_d(d) <= k:
            lower = d
        else:
            upper = d
    return d

# 二分法，过l作切线，切点point d的横坐标
@njit
def atan_du(l, u):
    upper = u
    lower = 0
    for i in range(20):
        d = (upper + lower) / 2
        if (atan(d) - atan(l)) / (d - l) >= atan_d(d):
            upper = d
        else:
            lower = d
    return d

# 二分法，过u作切线，切点point d的横坐标
@njit
def atan_dl(l, u):
    upper = 0
    lower = l
    for i in range(20):
        d = (upper + lower) / 2
        if (atan(u) - atan(d)) / (u - d) >=  atan_d(d):
            lower = d
        else:
            upper = d
    return d

"""
激活函数的上下界
h_u[i,j,k] = alpha_u[i,j,k] * (y[i,j,k] + beta_u[i,j,k])
h_l[i,j,k] = alpha_l[i,j,k] * (y[i,j,k] + beta_l[i,j,k])
"""

# ReLU激活函数的上下界
@njit
def relu_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)

    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                # 0 <= l <= u, 神经元一定被激活
                if LB[i,j,k] > 0:
                    alpha_u[i,j,k] = 1
                    alpha_l[i,j,k] = 1
                # l <= u <= 0, 神经元一定不被激活
                elif UB[i,j,k] <= 0:
                    pass
                # l < 0 < u, 神经元可能被激活，也可能不被激活
                else:
                    alpha_u[i,j,k] = UB[i,j,k] / (UB[i,j,k]-LB[i,j,k])
                    alpha_l[i,j,k] = UB[i,j,k] / (UB[i,j,k]-LB[i,j,k])
                    beta_u[i,j,k] = -alpha_u[i,j,k] * LB[i,j,k]

    return alpha_u, alpha_l, beta_u, beta_l

# ReLU激活函数的自适应上下界
@njit
def ada_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)

    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                #  0 <= l <= u, 神经元一定被激活
                if LB[i,j,k] >= 0:
                    alpha_u[i,j,k] = 1
                    alpha_l[i,j,k] = 1
                #  l <= u <= 0, 神经元一定不被激活
                elif UB[i,j,k] <= 0:
                    pass
                #  l < 0 < u, 神经元可能被激活，也可能不被激活
                else:
                    alpha_u[i,j,k] = UB[i,j,k] / (UB[i,j,k]-LB[i,j,k])
                    beta_u[i,j,k] = -alpha_u[i,j,k] * LB[i,j,k]
                    # 斜率>1
                    if UB[i,j,k] >= -LB[i,j,k]:
                        alpha_l[i,j,k] = 1
                    # 斜率<1
                    else:
                        pass

    return alpha_u, alpha_l, beta_u, beta_l

# tanh激活函数的上下界
@njit
def tanh_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)

    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                # l = u
                if UB[i,j,k] == LB[i,j,k]:
                    # 上界
                    alpha_u[i,j,k] = tanh_d(UB[i,j,k])
                    beta_u[i,j,k] = tanh(UB[i,j,k]) - tanh_d(UB[i,j,k])*UB[i,j,k]
                    # 下界
                    alpha_l[i,j,k] = tanh_d(LB[i,j,k])
                    beta_l[i,j,k] = tanh(LB[i,j,k]) - tanh_d(LB[i,j,k])*LB[i,j,k]
                # l != u
                else:
                    k0 = (tanh(UB[i, j, k]) - tanh(LB[i,j,k])) / (UB[i,j,k] - LB[i,j,k])  # end-point连线斜率
                    # case1: tanh_d(l) < k 且 tanh_d(u) > k, l < u <= 0 或 l < 0 < u
                    if tanh_d(LB[i,j,k]) < k0 and tanh_d(UB[i,j,k]) > k0:
                        # 上界
                        alpha_u[i,j,k] = k0
                        beta_u[i,j,k] = tanh(LB[i,j,k]) - k0 * LB[i,j,k]
                        # 下界
                        alpha_l[i,j,k] = k0
                        d = tanh_kl(LB[i,j,k], UB[i,j,k], k0)
                        beta_l[i,j,k] = tanh(d) - k0 * d
                    # case2: tanh_d(l) > k 且 tanh_d(u) < k, 0 <= l < u 或 l < 0 < u
                    elif tanh_d(LB[i,j,k]) > k0 and tanh_d(UB[i,j,k]) < k0:
                        # 下界
                        alpha_l[i,j,k] = k0
                        beta_l[i,j,k] = tanh(LB[i, j, k]) - k0 * LB[i,j,k]
                        # 上界
                        alpha_u[i,j,k] = k0
                        d = tanh_ku(LB[i,j,k], UB[i,j,k], k0)
                        beta_u[i, j, k] = tanh(d) - k0 * d
                    # case3: tanh_d(l) < k 且 tanh_d(u) < k, l < 0 < u
                    else:
                        # 上界
                        d_u = tanh_du(LB[i,j,k], UB[i,j,k])    # 上界的point d的横坐标
                        d_uk1 = tanh_d(d_u)    # 过 (d,y(d))切线的斜率
                        d_uk2 = (tanh(d_u)-tanh(LB[i,j,k])) / (d_u-LB[i,j,k])  # (l,y(l))) 和 (d,y(d)) 连线的斜率
                        if d_uk1 < d_uk2:
                            alpha_u[i,j,k] = d_uk1
                            beta_u[i,j,k] = tanh(d_u) - d_uk1*d_u
                        else:
                            alpha_u[i,j,k] = d_uk2
                            beta_u[i,j,k] = tanh(LB[i,j,k]) - d_uk2 * LB[i,j,k]
                        # 下界
                        d_l = tanh_dl(LB[i,j,k], UB[i,j,k])    # 下界的point d的横坐标
                        d_lk1 = tanh_d(d_l)    # 过 (d,y(d))切线的斜率
                        d_lk2 = (tanh(d_l)-tanh(UB[i,j,k])) / (d_l-UB[i,j,k])  # (u,y(u))) 和 (d,y(d)) 连线的斜率
                        if d_lk1 < d_lk2:
                            alpha_l[i,j,k] = d_lk1
                            beta_l[i,j,k] = tanh(d_l) - d_lk1*d_l
                        else:
                            alpha_l[i,j,k] = d_lk2
                            beta_l[i,j,k] = tanh(UB[i,j,k]) - d_lk2 * UB[i,j,k]

    return alpha_u, alpha_l, beta_u, beta_l

# sigmoid激活函数的上下界
@njit
def sigmoid_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)

    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                # l = u
                if UB[i, j, k] == LB[i, j, k]:
                    # 上界
                    alpha_u[i, j, k] = sigmoid_d(UB[i, j, k])
                    beta_u[i, j, k] = sigmoid(UB[i, j, k]) - sigmoid_d(UB[i, j, k]) * UB[i, j, k]
                    # 下界
                    alpha_l[i, j, k] = sigmoid_d(LB[i, j, k])
                    beta_l[i, j, k] = sigmoid(LB[i, j, k]) - sigmoid_d(LB[i, j, k]) * LB[i, j, k]
                # l != u
                else:
                    k0 = (sigmoid(UB[i, j, k]) - sigmoid(LB[i, j, k])) / (UB[i, j, k] - LB[i, j, k])  # end-point连线斜率
                    # case1: sigmoid_d(l) < k 且 sigmoid_d(u) > k, l < u <= 0 或 l < 0 < u
                    if sigmoid_d(LB[i, j, k]) < k0 and sigmoid_d(UB[i, j, k]) > k0:
                        # 上界
                        alpha_u[i, j, k] = k0
                        beta_u[i, j, k] = sigmoid(LB[i, j, k]) - k0 * LB[i, j, k]
                        # 下界
                        alpha_l[i, j, k] = k0
                        d = sigmoid_kl(LB[i, j, k], UB[i, j, k], k0)
                        beta_l[i, j, k] = sigmoid(d) - k0 * d
                    # case2: sigmoid_d(l) > k 且 sigmoid_d(u) < k, 0 <= l < u 或 l < 0 < u
                    elif sigmoid_d(LB[i, j, k]) > k0 and sigmoid_d(UB[i, j, k]) < k0:
                        # 下界
                        alpha_l[i, j, k] = k0
                        beta_l[i, j, k] = sigmoid(LB[i, j, k]) - k0 * LB[i, j, k]
                        # 上界
                        alpha_u[i, j, k] = k0
                        d = sigmoid_ku(LB[i, j, k], UB[i, j, k], k0)
                        beta_u[i, j, k] = sigmoid(d) - k0 * d
                    # case3: sigmoid_d(l) < k 且 sigmoid_d(u) < k, l < 0 < u
                    else:
                        # 上界
                        d_u = sigmoid_du(LB[i, j, k], UB[i, j, k])  # 上界的point d的横坐标
                        d_uk = sigmoid_d(d_u)  # 过 (d,y(d)) 切线的斜率
                        alpha_u[i, j, k] = d_uk
                        beta_u[i, j, k] = sigmoid(LB[i, j, k]) - d_uk * LB[i, j, k]
                        # 下界
                        d_l = sigmoid_dl(LB[i, j, k], UB[i, j, k])  # 下界的point d的横坐标
                        d_lk = sigmoid_d(d_l)  # 过 (d,y(d)) 切线的斜率
                        alpha_l[i, j, k] = d_lk
                        beta_l[i, j, k] = sigmoid(d_l) - d_lk * d_l

    return alpha_u, alpha_l, beta_u, beta_l

# atan激活函数的上下界
@njit
def atan_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)

    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                # l = u
                if UB[i, j, k] == LB[i, j, k]:
                    # 上界
                    alpha_u[i, j, k] = atan_d(UB[i, j, k])
                    beta_u[i, j, k] = atan(UB[i, j, k]) - atan_d(UB[i, j, k]) * UB[i, j, k]
                    # 下界
                    alpha_l[i, j, k] = atan_d(LB[i, j, k])
                    beta_l[i, j, k] = atan(LB[i, j, k]) - atan_d(LB[i, j, k]) * LB[i, j, k]
                # l != u
                else:
                    k0 = (atan(UB[i, j, k]) - atan(LB[i, j, k])) / (UB[i, j, k] - LB[i, j, k])  # end-point连线斜率
                    # case1: atan_d(l) < k 且 atan_d(u) > k, l < u <= 0 或 l < 0 < u
                    if atan_d(LB[i, j, k]) < k0 and atan_d(UB[i, j, k]) > k0:
                        # 上界
                        alpha_u[i, j, k] = k0
                        beta_u[i, j, k] = atan(LB[i, j, k]) - k0 * LB[i, j, k]
                        # 下界
                        alpha_l[i, j, k] = k0
                        d = atan_kl(LB[i, j, k], UB[i, j, k], k0)
                        beta_l[i, j, k] = atan(d) - k0 * d
                    # case2: atan_d(l) > k 且 atan_d(u) < k, 0 <= l < u 或 l < 0 < u
                    elif atan_d(LB[i, j, k]) > k0 and atan_d(UB[i, j, k]) < k0:
                        # 下界
                        alpha_l[i, j, k] = k0
                        beta_l[i, j, k] = atan(LB[i, j, k]) - k0 * LB[i, j, k]
                        # 上界
                        alpha_u[i, j, k] = k0
                        d = atan_ku(LB[i, j, k], UB[i, j, k], k0)
                        beta_u[i, j, k] = atan(d) - k0 * d
                    # case3: atan_d(l) < k 且 atan_d(u) < k, l < 0 < u
                    else:
                        # 上界
                        d_u = atan_du(LB[i, j, k], UB[i, j, k])  # 上界的point d的横坐标
                        d_uk1 = atan_d(d_u)  # 过 (d,y(d)) 切线的斜率
                        d_uk2 = (atan(d_u) - atan(LB[i, j, k])) / (d_u - LB[i, j, k])  # (l,y(l))) 和 (d,y(d)) 连线的斜率
                        if d_uk1 < d_uk2:
                            alpha_u[i, j, k] = d_uk1
                            beta_u[i, j, k] = atan(d_u) - d_uk1 * d_u
                        else:
                            alpha_u[i, j, k] = d_uk2
                            beta_u[i, j, k] = atan(LB[i, j, k]) - d_uk2 * LB[i, j, k]
                        # 下界
                        d_l = atan_dl(LB[i, j, k], UB[i, j, k])  # 下界的point d的横坐标
                        d_lk1 = atan_d(d_l)  # 过 (d,y(d)) 切线的斜率
                        d_lk2 = (atan(d_l) - atan(UB[i, j, k])) / (d_l - UB[i, j, k])  # (u,y(u))) 和 (d,y(d)) 连线的斜率
                        if d_lk1 < d_lk2:
                            alpha_l[i, j, k] = d_lk1
                            beta_l[i, j, k] = atan(d_l) - d_lk1 * d_l
                        else:
                            alpha_l[i, j, k] = d_lk2
                            beta_l[i, j, k] = atan(UB[i, j, k]) - d_lk2 * UB[i, j, k]

    return alpha_u, alpha_l, beta_u, beta_l