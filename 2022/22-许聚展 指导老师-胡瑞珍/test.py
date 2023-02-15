
import os, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from UOC.app import Segmenter
from robot.tools import pv_tools

if __name__ == "__main__":
    seg = Segmenter()
    
    rgb = np.array(Image.open("./images/sc_rgb.png"))[:,:,:3][:,:,::-1].copy()
    dep = np.load("./images/sc_dep.npy")
    cam = np.load("./images/sc_cam.npy")
    c2w = np.load("./images/sc_c2w.npy")

    camera_params = {}
    # camera_params['x_offset'] = cam[0,0]
    # camera_params['y_offset'] = cam[1,1]
    # camera_params['fx'] = cam[0,2]
    # camera_params['fy'] = cam[1,2]
    camera_params['c2w'] = c2w
    camera_params['cam'] = cam

    print(c2w)
    print(cam)

    all_rgb, all_bbox = seg.segment_and_crop(rgb, dep, camera_params)

    all_points = seg.crop_point_cloud(dep, camera_params, all_bbox)
    
    all_pc = []
    for points in all_points:
        all_pc.append(pv_tools.get_pc(points))
    
    mat = np.eye(4)
    mat[:3,3] = c2w[:3,3]
    cam_ax = pv_tools.get_axis(mat)
    
    ax = pv_tools.get_axis()

    all_pc.append(cam_ax)
    all_pc.append(ax)

    pv_tools.show_mesh(all_pc)

    a = 1
