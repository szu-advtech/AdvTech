import os
import numpy as np
import sys
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.chdir(sys.path[0])

load_path = '../VIGOR-main/data/same_area/Overall/'  
# sat_descriptor = np.load(os.path.join(load_path, 'sat_global_descriptor.npy'))
# grd_descriptor = np.load(os.path.join(load_path, 'grd_global_descriptor.npy'))
sat_descriptor = np.load(load_path + 'sat_global_descriptor.npy')
grd_descriptor = np.load(load_path + 'grd_global_descriptor.npy')
sat_descriptor, grd_descriptor = sat_descriptor[:23279,:], grd_descriptor[:13885,:]

similarity = np.matmul(grd_descriptor, np.transpose(sat_descriptor))
# order_list = np.argmax(similarity, axis=1)
# del(similarity)

# print(sat_descriptor.shape)
# print(grd_descriptor.shape)

def vigor_select(index=0):
    dis=similarity[index]
    dis=-dis
    sat_list=np.argsort(dis)
    # sat_list=[1,2,3,4,5,6,7,8]
    return sat_list[:50]