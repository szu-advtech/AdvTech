import numpy as np
import matplotlib.pyplot as plt
# class_map = np.load('./color_map.npy')    #使用numpy载入npy文件
# plt.imshow(class_map)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# # plt.colorbar()                   #添加colorbar
# # plt.savefig('depthmap.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
# plt.show()                        #在线显示图像
class_map = np.load('./class_map.npy', allow_pickle=True).item()
print(class_map) # {0: 'background', 1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'indigo', 7: 'violet'}