import numpy as np
import matplotlib.pyplot as plt

with open('C:\\Users\\Administrator\\Desktop\\pythonProject\\output\\loss.txt', 'r') as f:
    datas = f.readlines()

data_list = []
for i in datas:
    i = i.replace('\n', '')
    data_list.append(float(i))

plt.plot(data_list[0:-1:500])
plt.show()