import os

import numpy as np
import os

GN_DIRECTORY = "E:/dataset/grasp_net/models"

def get_random_name(sample_range=[0, 13], sample_num=10):

    valid_list = np.load("./data/valid_grasp.npy")

    invalid_list = []
    for v in valid_list:
        if v == 0:
            invalid_list.append(v)
    invalid_list.append(5) # banana
    
    ids = [i for i in range(sample_range[0], sample_range[1]) if i not in invalid_list]
    sample_ids = np.random.choice(ids, sample_num)
    ret = []

    for sid in sample_ids:
        ret.append("%03d" % sid)
    return ret

def convert():
    import pybullet as p
    import os
    for index in range(0, 88):
    # for index in [82]:
        obj_name = "%03d" % index
        name_in = os.path.join(GN_DIRECTORY, obj_name, "textured.obj")
        name_out = os.path.join(GN_DIRECTORY, obj_name, "vhacd.obj")
        name_log = os.path.join("./log")
        p.vhacd( name_in, name_out, name_log, resolution=500000, depth=5)

def volume():
    all = []
    import pyvista as pv
    
    for index in range(0, 88):
    # for index in [82]:
        obj_name = "%03d" % index
        name_in = os.path.join(GN_DIRECTORY, obj_name, "textured.obj")
        mesh = pv.read(name_in)
        
        data = pv.MultiBlock([mesh])
        volume = data.volume

        all.append(volume)
        print(index, volume)
    
    np.save("./data/volume", all)

if __name__ == '__main__':
    volume()