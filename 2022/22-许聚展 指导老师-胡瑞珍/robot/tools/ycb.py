import os
import numpy as np

YCB_DIRECTORY = "E:/dataset/ycb"

ycb_list = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "036_wood_block",
    "046_plastic_bolt",
    "052_extra_large_clamp",
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "065-a_cups",
    "065-b_cups",
    "065-c_cups",
    "065-d_cups",
    "077_rubiks_cube"
]


def get_random_name(sample_range=[0,6], sample_num=10):
    ids = [i for i in range(sample_range[0], sample_range[1])]
    sample_ids = np.random.choice(ids, sample_num)
    ret = []

    for sid in sample_ids:
        ret.append(ycb_list[sid])
    return ret



def convert():
    import pybullet as p
    import os
    for obj_name in ycb_list:
        name_in = os.path.join(YCB_DIRECTORY, obj_name, "google_16k", "textured.obj")
        name_out = os.path.join(YCB_DIRECTORY, obj_name, "google_16k", "vhacd.obj")
        name_log = os.path.join("./log")
        p.vhacd( name_in, name_out, name_log, resolution=500000, depth=1)

if __name__ == "__main__":
    convert()