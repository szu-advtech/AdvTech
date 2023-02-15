import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ['SimHei']

# GN
def polar_transform_iter(u):
    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0, stages):
        i = 0
        while i < N:
            for j in range(0, n):
                idx = i + j
                x[idx] = x[idx] ^ x[idx + n]  # 异或
            i = i + 2 * n
        n = 2 * n
    return x

image = cv2.imread('lena_gray_256.tif', cv2.IMREAD_GRAYSCALE)
image_list = []  # 像素值列表
for i, ii in enumerate(list(image)):  # 遍历获得索引和值
    for j, jj in enumerate(ii):
        image_list.append(jj)

bi_list = []  # 像素值转2进制列表
for i in image_list:
    bi_list.append(bin(i).replace("0b", ""))  # 转为2进制

bi_list_1 = []
for i, ii in enumerate(bi_list):
    if np.mod(i, 2) == 0:
        k = ii
    else:
        bi_list_1.append(k+ii)   # 每2个8bits序列拼到一起

mat_16 = np.zeros([32768, 16])  # bit序列塞入数组
for i, ii in enumerate(bi_list_1):
    if len(ii) < 16:
        num = 16 - len(ii)
    else:
        num = 0
    for j, jj in enumerate(ii):
        mat_16[i][num+j] = int(jj)
img_int_16 = mat_16.astype(np.uint8)
img_bool_16 = img_int_16.astype(np.bool)

# Channel (BSC(0.11))
#y = img_bool_16 + np.random.binomial(n=1, p=0.11, size=img_bool_16.shape)  # 添加ber(0.11)的噪声

# 16位像素值送入DNN压缩，译出8位信息位，压缩率0.5
def ber(y_true, y_pred):  #ber 误码率
    return K.mean(K.not_equal(y_true, K.round(y_pred)))  # K.round四舍六入五取偶

def errors(y_true, y_pred):
    return K.sum(K.cast(K.not_equal(y_true, K.round(y_pred)), dtype = 'int32'))

A1 = [False, False, False, False, False, False, False, True, False, True, True, True, True, True, True, True]   # 信息位索引
A2 = [True, True, True, True, True, True, True, False, True, False, False, False, False, False, False, False]   # Frozen bit索引

#model = tf.keras.models.load_model('BSC-k8.h5', custom_objects={"ber": ber})
decoder = tf.keras.models.load_model('BSC_decoder-k8.h5', custom_objects={"errors": errors})

prediction = decoder.predict(img_bool_16)  # 信息位置为1的概率
print("prediction:", prediction.shape)
pred = np.zeros((32768, 8), dtype=bool)  # 信息位的译码结果
for i in range(32768):
    for ii in range(8):
        if(prediction[i][ii] >= 0.5):
            pred[i][ii] = True
        else:
            pred[i][ii] = False

# 解压缩，冻结比特位置用0补齐
frozenbit1 = np.zeros((32768, 7), dtype=bool)
frozenbit2 = np.zeros((32768), dtype=bool)
pred1 = pred[0:32768, 0]
pred2 = pred[0:32768, 1:8]

new_img_16 = np.zeros((32768, 16), dtype=bool)
new_img_16[0:32768, 0:7] = frozenbit1
new_img_16[0:32768, 8] = frozenbit2
new_img_16[0:32768, 7] = pred1
new_img_16[0:32768, 9:16] = pred2

new_int_16 = new_img_16.astype(np.uint8)
#print(new_int_16)

# GN
x_img = polar_transform_iter(new_int_16)

# 解压缩后恢复图像
image_1 = []
for i in (img_int_16):
    a = ""
    b = ""
    for j, jj in enumerate(i):
        if j <= 7:
            a = a + str(jj)
        if j > 7:
            b = b + str(jj)
    image_1.append(a)
    image_1.append(b)
image_ori = np.zeros([256*256, 1])
for i, ii in enumerate(image_1):
    x = (int(ii, 2))  # 2进制转10进制
    #image_ori[i] = np.round(x)
    image_ori[i] = x
image_ori = image_ori.reshape([256, 256])

#img = image_ori.astype(np.uint8)
#plt.imshow(image_ori, cmap="gray")
#plt.show()

#plot 1:
plt.subplot(1, 2, 1)
img = plt.imread('lena_gray_256.tif')
plt.imshow(img, cmap="gray")
plt.title("原始图像", fontproperties="SimHei")

#plot 2:
plt.subplot(1, 2, 2)
plt.imshow(image_ori, cmap="gray")
plt.title("解压缩后的图像", fontproperties="SimHei")
plt.show()