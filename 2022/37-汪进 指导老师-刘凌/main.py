import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras import backend as K
import matplotlib.pyplot as plt
import warnings
import os
import h5py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息

# Parameters
k = 8                       # number of information bits
N = 16                      # code length
train_SNR_Eb = 1            # training-Eb/No 比特信噪比

nb_epoch = 2**16            # number of learning epochs 2^16
code = 'polar'              # type of code ('random' or 'polar')
design = [128, 64, 32]      # each list entry defines the number of nodes in a layer
batch_size = 256            # size of batches for calculation the gradient
LLR = True                 # 是否使用LLR层 ('True' or 'False')
optimizer = 'adam'          # 优化算法，调节权重
loss = 'mse'                # 均方误差 ’mean-square error‘ or 'binary-cross-entropy'

train_SNR_Es = train_SNR_Eb + 10*np.log10(k/N)  # 符号信噪比，每个符号的能量与噪声功率谱密度之比
train_sigma = np.sqrt(1/(2*10**(train_SNR_Es/10)))

# Define NN model
'''
def modulateBPSK(x):
    return -2*x + 1
    
#AWGN信道添加噪声
def addNoise(x, sigma): 
    w = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)  # 正态分布
    return x + w
'''
#BSC(0.11)信道
def addNoise(x, sigma):
    w = K.random_binomial(K.shape(x), p=0.11)  # 正态分布
    return x + w

def ber(y_true, y_pred):  #ber 误码率
    return K.mean(K.not_equal(y_true, K.round(y_pred)))  # K.round四舍六入五取偶

# 一阶张量[1,2,3]的shape是(3,);
# 二阶张量[[1,2,3],[4,5,6]]的shape是(2,3);
# 三阶张量[[[1],[2],[3]],[[4],[5],[6]]]的shape是(2,3,1)
# 输出张量的形状
def return_output_shape(input_shape):
    return input_shape

# 构造模型
def compose_model(layers):
    model = Sequential()  # 顺序模型
    for layer in layers:
        model.add(layer)
    return model

def log_likelihood_ratio(x, sigma):  #LLR
    return 2*x/np.float32(sigma**2)

def errors(y_true, y_pred):
    return K.sum(K.cast(K.not_equal(y_true, K.round(y_pred)), dtype = 'int32'))

# 如果只是想对流经该层的数据做个变换，而这个变换本身没有需要学习的参数，那么直接用Lambda Layer是最合适的
'''
# Define modulator 调制
modulator_layers = [Lambda(modulateBPSK, input_shape=(N,), output_shape=return_output_shape, name="modulator")]   # 匿名函数
modulator = compose_model(modulator_layers)  # 添加调制层
modulator.compile(optimizer=optimizer, loss=loss)  # 配置训练方法，告知优化器和损失函数
'''

# Define noise
noise_layers = [Lambda(addNoise, arguments={'sigma': train_sigma}, input_shape=(N,), output_shape=return_output_shape, name="noise")]
noise = compose_model(noise_layers)
noise.compile(optimizer=optimizer, loss=loss)

# Define LLR 使用或不使用LLR层
llr_layers = [Lambda(log_likelihood_ratio, arguments={'sigma': train_sigma}, input_shape=(N,), output_shape=return_output_shape, name="LLR")]
llr = compose_model(llr_layers)
llr.compile(optimizer=optimizer, loss=loss)

# Define decoder 译码层
decoder_layers = [Dense(design[0], activation='relu', input_shape=(N,))]  # dense 全连接层
for i in range(1, len(design)):
    decoder_layers.append(Dense(design[i], activation='relu'))
decoder_layers.append(Dense(k, activation='sigmoid'))  # dense_3
decoder = compose_model(decoder_layers)
decoder.compile(optimizer=optimizer, loss=loss, metrics=[errors])

# Define model
if LLR:
    #model_layers = modulator_layers + noise_layers + llr_layers + decoder_layers
    model_layers = noise_layers + llr_layers + decoder_layers
else:
    #model_layers = modulator_layers + noise_layers + decoder_layers
    model_layers = noise_layers + decoder_layers
model = compose_model(model_layers)
model.compile(optimizer=optimizer, loss=loss, metrics=[ber])  # metrics 评价函数用于评估当前训练模型的性能

# Data Generation0000000000000000000000
# 半加器 S是加和数，C是进位数
def half_adder(a, b):
    s = a ^ b  # 按位异或
    c = a & b  # 按位与
    return s, c

# 全加器，c为进位
def full_adder(a, b, c):
    s = (a ^ b) ^ c
    c = (a & b) | (c & (a ^ b))  # | 按位或
    return s, c

# a + b，数据类型为bool
def add_bool(a, b):
    if len(a) != len(b):
        raise ValueError('arrays with different length')
    k = len(a)
    s = np.zeros(k, dtype=bool)
    c = False
    for i in reversed(range(0, k)):
        s[i], c = full_adder(a[i], b[i], c)
    if c:
        warnings.warn("Addition overflow!")
    return s

#   + 1，数据类型为bool
def inc_bool(a):
    k = len(a)
    increment = np.hstack((np.zeros(k - 1, dtype=bool), np.ones(1, dtype=bool)))  # 数组按水平方向进行拼接 [0,0,...,0,1]
    # print("increment: ", increment)
    a = add_bool(a, increment)
    return a

def bitrevorder(x):
    m = np.amax(x)  # 返回最大值
    print("m", m)
    n = np.ceil(np.log2(m)).astype(int)  # 各元素向上取整
    print("n", n)
    for i in range(0, len(x)):
        x[i] = int('{:0{n}b}'.format(x[i], n=n)[::-1], 2)
        print("x[", i, "]", x[i])
    return x

def int2bin(x, N):
    if isinstance(x, list) or isinstance(x, np.ndarray):  # 判断对象是否是已知类型
        binary = np.zeros((len(x), N), dtype='bool')
        for i in range(0, len(x)):
            binary[i] = np.array([int(j) for j in bin(x[i])[2:].zfill(N)])  # 返回指定长度的字符串，原字符串右对齐，前面填充0
    else:
        binary = np.array([int(j) for j in bin(x)[2:].zfill(N)], dtype=bool)

    return binary

# 将二进制数组的数据类型改为int
def bin2int(b):
    if isinstance(b[0], list):
        integer = np.zeros((len(b),), dtype=int)
        for i in range(0, len(b)):
            out = 0
            for bit in b[i]:
                out = (out << 1) | bit
            integer[i] = out
    elif isinstance(b, np.ndarray):
        if len(b.shape) == 1:
            out = 0
            for bit in b:
                out = (out << 1) | bit
            integer = out
        else:
            integer = np.zeros((b.shape[0],), dtype=int)
            for i in range(0, b.shape[0]):
                out = 0
                for bit in b[i]:
                    out = (out << 1) | bit
                integer[i] = out

    return integer

#  AWGN信道构造极化码  巴氏参数
def polar_design_awgn(N, k, design_snr_dB):
    S = 10 ** (design_snr_dB / 10)
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)
    for j in range(1, int(np.log2(N)) + 1):
        u = 2 ** j
        for t in range(0, int(u / 2)):
            T = z0[t]
            z0[t] = 2 * T - T ** 2  # upper channel
            z0[int(u / 2) + t] = T ** 2  # lower channel
    #print("z0:", z0)

    # sort into increasing order  返回的是元素值从小到大排序后的索引值的数组
    idx = np.argsort(z0)
    #print("z0从小到大排序后的索引值idx:", idx)

    # select k best channels 用于传输信息位
    idx = np.sort(bitrevorder(idx[0:k]))
    #print("比特翻转后索引值从小到大idx:", idx)

    A = np.zeros(N, dtype=bool)
    A[idx] = True
    print("A:", A)
    return A

# 迭代生成Polar码 GN
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

# Create all possible information words
d = np.zeros((2 ** k, k), dtype=bool)
for i in range(1, 2 ** k):
    d[i] = inc_bool(d[i - 1])
    #print("d[i]: ", d[i])

# Create sets of all possible codewords (codebook)
if code == 'polar':

    A = polar_design_awgn(N, k, design_snr_dB=0)  # logical vector indicating the nonfrozen bit locations 信息比特位置
    x = np.zeros((2 ** k, N), dtype=bool)
    u = np.zeros((2 ** k, N), dtype=bool)
    u[:, A] = d

    for i in range(0, 2 ** k):
        x[i] = polar_transform_iter(u[i])  # x = u GN

elif code == 'random':

    np.random.seed(4267)  # for a 16bit Random Code (r=0.5) with Hamming distance >= 2
    x = np.random.randint(0, 2, size=(2 ** k, N), dtype=bool)

#Train Neural Network
model.summary()  # 输出各层参数
history = model.fit(x, d, batch_size=batch_size, epochs=nb_epoch, verbose=0, shuffle=True)  # 训练
#model.save("BSC-k8.h5")  # 保存模型
decoder.save("BSC_decoder-k8.h5")  # 保存模型

# Test NN
test_batch = 1000
num_words = 100000      # multiple of test_batch

SNR_dB_start_Eb = 0
SNR_dB_stop_Eb = 5
SNR_points = 20

SNR_dB_start_Es = SNR_dB_start_Eb + 10 * np.log10(k / N)
SNR_dB_stop_Es = SNR_dB_stop_Eb + 10 * np.log10(k / N)

sigma_start = np.sqrt(1 / (2 * 10 ** (SNR_dB_start_Es / 10)))
sigma_stop = np.sqrt(1 / (2 * 10 ** (SNR_dB_stop_Es / 10)))

sigmas = np.linspace(sigma_start, sigma_stop, SNR_points)  # 在start和end之间生成一个统一的序列，共有num_points个元素

nb_errors = np.zeros(len(sigmas), dtype=int)
nb_bits = np.zeros(len(sigmas), dtype=int)

for i in range(0, len(sigmas)):

    for ii in range(0, np.round(num_words / test_batch).astype(int)):

        # Source
        np.random.seed(0)
        d_test = np.random.randint(0, 2, size=(test_batch, k))   # 生成0~1之间的数，数组大小(1000,k)

        # Encoder
        if code == 'polar':
            x_test = np.zeros((test_batch, N), dtype=bool)
            u_test = np.zeros((test_batch, N), dtype=bool)
            u_test[:, A] = d_test

            for iii in range(0, test_batch):
                x_test[iii] = polar_transform_iter(u_test[iii])

        elif code == 'random':
            x_test = np.zeros((test_batch, N), dtype=bool)
            for iii in range(0, test_batch):
                x_test[iii] = x[bin2int(d_test[iii])]

        # Modulator (BPSK)  BSC信道不用调制
        #s_test = -2 * x_test + 1

        # Channel (AWGN)
        #y_test = s_test + sigmas[i] * np.random.standard_normal(s_test.shape)  # 正态分布

        # Channel (BSC(0.11))
        y_test = x_test + sigmas[i] * np.random.binomial(n=1, p=0.11, size=x_test.shape)  # 添加ber(0.11)的噪声

        if LLR:
            y_test = 2 * y_test / (sigmas[i] ** 2)

        # Decoder
        nb_errors[i] += decoder.evaluate(y_test, d_test, batch_size=test_batch, verbose=0)[1]  # evaluate返回complie.(loss和metrics)
        nb_bits[i] += d_test.size
        #accuracy = model.evaluate(y_test, d_test)  # y_test数据，d_text标签
        #y_pred = model.predict(y_test, batch_size=1)  # 对y_test的预测值
        #print("d_text:", d_test)   # 标签
        #print(("y_pred:", y_pred))   # 预测值

# 保存绘图点
NNX = 10 * np.log10(1 / (2 * sigmas ** 2)) - 10 * np.log10(k / N)  # 0-5dB
NNY = nb_errors / nb_bits
#print("NNX:", NNX)
#print("NNY:", NNY)
#np.savetxt('polar-16.txt', (NNX, NNY))

#Load MAP
#result_map = np.loadtxt('map/{}/results_{}_map_{}_{}.txt'.format(code, code, N, k), delimiter=', ')  # 信息位个数k从1-8
#sigmas_map = result_map[:, 0]  # map中第一个元素
#nb_bits_map = result_map[:, 1]
#nb_errors_map = result_map[:, 2]

# Plot Bit-Error-Rate
legend = []

plt.plot(10 * np.log10(1 / (2 * sigmas ** 2)) - 10 * np.log10(k / N), nb_errors / nb_bits)
legend.append('NN-BSC')

#plt.plot(10*np.log10(1/(2*sigmas_map**2)) - 10*np.log10(k/N), nb_errors_map/nb_bits_map)
#legend.append('MAP')

plt.legend(legend, loc=3)
plt.yscale('log')
plt.xlabel('$E_b/N_0$')
plt.ylabel('BER')
plt.grid(True)  # 显示网格线
plt.show()