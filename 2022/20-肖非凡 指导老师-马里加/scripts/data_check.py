"""
查看各种数据类型

"""
import numpy as np
import pandas as pd 
grasp = pd.read_csv("/home/xff/VGN/GIGA/data/pile/data_packed_train_random_new/grasps.csv")
print(grasp.shape)


"""
raw data
"""
#scenes
data=np.load('/home/xff/VGN/GIGA/data/pile/data_packed_train_random_raw_4M/scenes/0a0c813213fe426c840f48c6e593431c.npz',allow_pickle=True)
print(data.files)    #  depth_imgs  extrinsics 两个数据集
print(len(data.files))
print(data['depth_imgs'])
print(data['depth_imgs'].shape)  #[1,480,640]

print(data['extrinsics'])
print(data['extrinsics'].shape)   #[1,7]

# occ数据
data1=np.load('/home/xff/VGN/GIGA/data/pile/data_packed_train_random_raw_4M/occ/0a0c813213fe426c840f48c6e593431c/0000.npz',allow_pickle=True)
data2=np.load('/home/xff/VGN/GIGA/data/pile/data_packed_train_random_raw_4M/occ/0a0c813213fe426c840f48c6e593431c/0001.npz',allow_pickle=True)
print(data1.files)    #  points  occ 两个数据集
print(data1['points'])
print(data1['points'].shape)  #[100000,3]
print(data1['occ'])
print(data1['occ'].shape)  #[10000,]  [False.False.....]

print(data2.files)    #  points  occ 两个数据集
print(data2['points'])
print(data2['points'].shape)  #[100000,3]
print(data2['occ'])
print(data2['occ'].shape)  #[100000,]

"""
new data
"""
#scenes  tsdf数据类型
data=np.load('/home/xff/VGN/GIGA/data/pile/data_packed_train_random_new/scenes/99d205b1f8754c858712d9e099c4de49.npz',allow_pickle=True)
print(data.files)  #里面有grid这个数据集
print(len(data.files))

print(data['grid'])
print(data['grid'].shape)  #[1,40,40,40]

#查看point clouds数据类型
data=np.load('/home/xff/VGN/GIGA/data/pile/data_packed_train_random_new/point_clouds/0a0c813213fe426c840f48c6e593431c.npz',allow_pickle=True)
print(data.files)  #里面有pc这个数据集
print(len(data.files))
print(data['pc'])
print(data['pc'].shape)  #[1354,3]

