import cv2
import math
import numpy as np
import os

# 色彩
def UICM(img):
    # 对图像通道进行切割，注意使用opencv读取到的图像通道顺序不是RGB
    B,R,G = cv2.split(img)
    # 公式（1）
    RG = R - G
    # 公式（2）
    YB = (R + G) / 2 - B
    # img图片有三个通道，是三维的，而rbg为二维的
    m,n,o = np.shape(img)
    K = m*n
    # 可调参数α_L和α_R
    alpha_L = 0.1
    alpha_R = 0.1
    # α_L·K上取整
    T_alpha_L = math.ceil(alpha_L*K)
    # α_R·K下取整
    T_alpha_R = math.floor(alpha_R*K)
 
    # 计算RG的均值和方差：
    # 先将二维数组转换为一维数组，以便计算
    RG_list = RG.flatten()
    RG_list = sorted(RG_list)
    sum_RG = 0
    # RG均值：公式（3）
    for i in range(T_alpha_L+1, K-T_alpha_R ):
        sum_RG = sum_RG + RG_list[i]
    U_RG = sum_RG/(K - T_alpha_R - T_alpha_L)
    squ_RG = 0
    # RG方差：公式（4）
    for i in range(K):
        squ_RG = squ_RG + np.square(RG_list[i] - U_RG)
    sigma2_RG = squ_RG/K
 
    # 计算YB的均值和方差，与上面一样
    YB_list = YB.flatten()
    YB_list = sorted(YB_list)
    sum_YB = 0
    for i in range(T_alpha_L+1, K-T_alpha_R ):
        sum_YB = sum_YB + YB_list[i]
    U_YB = sum_YB/(K - T_alpha_R - T_alpha_L)
    squ_YB = 0
    for i in range(K):
        squ_YB = squ_YB + np.square(YB_list[i] - U_YB)
    sigma2_YB = squ_YB/K
 
    # 公式（5）
    Uicm = -0.0268*np.sqrt(np.square(U_RG) + np.square(U_YB)) + 0.1586*np.sqrt(sigma2_RG + sigma2_YB)
    return Uicm

# 清晰度——公式（7）
def EME(rbg,L):
    # L是每个区域的大小（长宽有多少个像素），rgb是彩色图像某一个通道的二维矩阵
    # 横向为n列 纵向为m行
    m,n = np.shape(rbg)
    # 获取横向和纵向的区域数
    number_m = math.floor(m/L)
    number_n = math.floor(n/L)
    m1 = 0
    E = 0
    for i in range(number_m):
        n1 = 0
        # 循环每一个区域
        for t in range(number_n):
            # A1就是这个区域块包含的像素点二维矩阵
            # 计算得到每个区域中的亮度最大值和最小值
            A1 = rbg[m1:m1+L, n1:n1+L]
            rbg_min = np.amin(np.amin(A1))
            rbg_max = np.amax(np.amax(A1))
 
            if rbg_min > 0 :
                rbg_ratio = rbg_max/rbg_min
            else :
                rbg_ratio = rbg_max
            E = E + np.log(rbg_ratio + 1e-5)
 
            n1 = n1 + L
        m1 = m1 + L
    E_sum = 2*E/(number_m*number_n)
    return E_sum

# 对比度
def UICONM(rbg,L):
    # L是每个区域的大小（长宽有多少个像素），rgb是彩色图像某一个通道的二维矩阵
    # 横向为n列 纵向为m行
    m,n,o = np.shape(rbg)
    # 获取横向和纵向的区域数
    number_m = math.floor(m/L)
    number_n = math.floor(n/L)
    m1 = 0
    logAMEE = 0
    # 根据公式的方法，通过两层for循环计算
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            # A1就是这个区域块包含的像素点二维矩阵
            # 计算每个区域中的亮度最大值和最小值
            A1 = rbg[m1:m1+L, n1:n1+L]
            rbg_min = int(np.amin(np.amin(A1)))
            rbg_max = int(np.amax(np.amax(A1)))

            # 计算PLIP的加⊕
            plip_add = rbg_max+rbg_min-rbg_max*rbg_min/1026
            if 1026-rbg_min > 0:
                # 计算PLIP的减Θ
                plip_del = 1026*(rbg_max-rbg_min)/(1026-rbg_min)
                # 计算求和里的算式和log里的算式
                if plip_del > 0 and plip_add > 0:
                    local_a = plip_del/plip_add
                    local_b = math.log(plip_del/plip_add)
                    phi = local_a * local_b
                    logAMEE = logAMEE + phi
            n1 = n1 + L
        m1 = m1 + L
    logAMEE = 1026-1026*((1-logAMEE/1026)**(1/(number_n*number_m)))
    return logAMEE
 
def FKY_UIQM(img_path):
    # 根据图像路径读取图像
    img = cv2.imread(img_path)
    r,b,g = cv2.split(img)

    # 色彩
    Uicm = UICM(img)

    # 清晰度
    EME_r = EME(r, 8)
    EME_b = EME(b, 8)
    EME_g = EME(g, 8)
    Uism = 0.299*EME_r + 0.144*EME_b + 0.557*EME_g

    # 对比度
    Uiconm = UICONM(img, 8)
    
    uiqm = 0.3333*Uicm + 0.3333*Uism + 0.3333*Uiconm
    # print(uiqm)
    return uiqm

# def fky_uiconm(img_path):
#     img = cv2.imread(img_path)
#     Uiconm = UICONM(img, 8)
#     print(Uiconm)

# fky_uiconm("test_real/600_img_.png")
# fky_uiconm("HE-1.png")
# fky_uiconm("HE-2.png")
# fky_uiconm("AHE.png")
# fky_uiconm("CLAHE-1.png")
# fky_uiconm("CLAHE-2.png")

# fky_uicm("test_real/600_img_.png")
# fky_uicm("HE-1.png")
# fky_uicm("HE-2.png")
# fky_uicm("AHE.png")
# fky_uicm("CLAHE-1.png")
# fky_uicm("CLAHE-2.png")
# fky_uicm("test_igs/600_img_.png")

if __name__== '__main__':
    file_path = ""
    file_num = len(os.listdir(file_path))
    sum = 0

    f = os.listdir(file_path)
    for filename in f:
        img_file = file_path+'/'+filename
        print(img_file)
        # img = cv2.imread(img_file)
        final = FKY_UIQM(img_file)
        print(final)
        sum = sum + final
    # print(sum/file_num)