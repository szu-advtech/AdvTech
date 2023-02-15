import numpy as np
import matplotlib.pyplot as plt

code = "polar"
N = 16
k = 8
#Load MAP
result_map = np.loadtxt('map/{}/results_{}_map_{}_{}.txt'.format(code, code, N, k), delimiter=', ')  # 信息位个数k从1-8
sigmas_map = result_map[:, 0]  # map中第一个元素
nb_bits_map = result_map[:, 1]
nb_errors_map = result_map[:, 2]

# Plot Bit-Error-Rate
legend = []

#Load data
'''
data1 = np.loadtxt('polar-llr-MSE.txt')
X1 = data1[0, :]  # 第一行元素
Y1 = data1[1, :]

data2 = np.loadtxt('polar-direct-MSE.txt')
X2 = data2[0, :]
Y2 = data2[1, :]

data3 = np.loadtxt('polar-llr-BCE.txt')
X3 = data3[0, :]  # 第一行元素
Y3 = data3[1, :]

data4 = np.loadtxt('polar-direct-BCE.txt')
X4 = data4[0, :]
Y4 = data4[1, :]

plt.plot(X1, Y1)
legend.append('polar-llr-MSE')

plt.plot(X2, Y2)
legend.append('polar-direct-MSE')

plt.plot(X3, Y3)
legend.append('polar-llr-BCE')

plt.plot(X4, Y4)
legend.append('polar-direct-BCE')
'''

'''
data1 = np.loadtxt('polar-12.txt')
X1 = data1[0, :]  
Y1 = data1[1, :]
data2 = np.loadtxt('polar-14.txt')
X2 = data2[0, :]  
Y2 = data2[1, :]
data3 = np.loadtxt('polar-16.txt')
X3 = data3[0, :]  
Y3 = data3[1, :]
data4 = np.loadtxt('polar-18.txt')
X4 = data4[0, :]  
Y4 = data4[1, :]

plt.plot(X1, Y1, marker='^')
legend.append('polar-Mep = 2^12')
plt.plot(X2, Y2, marker='p')
legend.append('polar-Mep = 2^14')
plt.plot(X3, Y3, marker='o')
legend.append('polar-Mep = 2^16')
plt.plot(X4, Y4, marker='s')
legend.append('polar-Mep = 2^18')
'''

data1 = np.loadtxt('random-12.txt')
X1 = data1[0, :]  
Y1 = data1[1, :]
data2 = np.loadtxt('random-14.txt')
X2 = data2[0, :]  
Y2 = data2[1, :]
data3 = np.loadtxt('random-16.txt')
X3 = data3[0, :]  
Y3 = data3[1, :]
data4 = np.loadtxt('random-18.txt')
X4 = data4[0, :]  
Y4 = data4[1, :]
data5 = np.loadtxt('random-20.txt')
X5 = data5[0, :]
Y5 = data5[1, :]

plt.plot(X1, Y1, marker='^')
legend.append('random-Mep = 2^12')
plt.plot(X2, Y2, marker='p')
legend.append('random-Mep = 2^14')
plt.plot(X3, Y3, marker='o')
legend.append('random-Mep = 2^16')
plt.plot(X4, Y4, marker='s')
legend.append('random-Mep = 2^18')
plt.plot(X5, Y5, marker='>')
legend.append('random-Mep = 2^20')

plt.plot(10*np.log10(1/(2*sigmas_map**2)) - 10*np.log10(k/N), nb_errors_map/nb_bits_map, marker='+')
legend.append('MAP')

plt.legend(legend, loc=3)
plt.yscale('log')
plt.xlabel('$E_b/N_0$')
plt.ylabel('BER')
plt.grid(True)  # 显示网格线
plt.show()