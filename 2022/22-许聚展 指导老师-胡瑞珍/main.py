from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from robot.scene.task import Env

from UOC.app import Segmenter
from VM.matcher import VisualMatcher
from ui import WidgetsExtension

image_folder = "E:/workspace/visual_match/images"

def save_rgbs(rgbs, folder, prefix):
    import cv2
    import os
    for i, rgb in enumerate(rgbs):
        f = os.path.join(folder, f"{prefix}_{i}.png")
        cv2.imwrite(f, rgb)

matcher = VisualMatcher()
seger = Segmenter()
env = Env(save_folder=image_folder)
ext = WidgetsExtension()
ext.init_window(env, image_folder)

if simulation_app.is_running():
    env.reset()
    env.move_to_init()
    while not env.is_start:
        env.idle(1)

    env.scene.load_objects()
    env.idle(200)

    tg_data = env.save_images('tg')
    ext.show_target_img()
    print("Segment target")
    tg_rgbs, tg_bbox = seger.segment_and_crop( tg_data[0], tg_data[1], tg_data[2] )
    tg_pcs = seger.crop_point_cloud(tg_data[1], tg_data[2], tg_bbox)
    # save_rgbs(tg_rgbs, image_folder, 'tg')

    env.reset()
    env.scene.load_objects_2()
    env.idle(200)

    sc_data = env.save_images('sc')
    ext.show_source_img()

    print("Segment source")
    sc_rgbs, sc_bbox = seger.segment_and_crop( sc_data[0], sc_data[1], sc_data[2] )
    sc_pcs = seger.crop_point_cloud(sc_data[1], sc_data[2], sc_bbox)

    # save_rgbs(sc_rgbs, image_folder, 'sc')

    print("Match objects")
    s_list, t_list = matcher.match_images(sc_rgbs, tg_rgbs, env.scene.names)

    print(s_list)
    print(t_list)

    # generate grasp
    print("Compute pick poses")
    sc_mats = env.get_pick_mat(sc_pcs)
    tg_mats = env.get_pick_mat(tg_pcs)
    
    min_num = len(s_list)
    if len(s_list) > len(t_list):
        min_num = len(t_list)

    for index in range(min_num):
        env.move_to_mat(sc_mats[ index ], 0.3)
        env.gripper_close()
        env.move_up(0.3)
        mat = tg_mats[ t_list[index] ]
        mat[:3, 3][2] += 0.05
        env.move_to_mat(mat, 0.3)
        env.gripper_open()
        env.idle(20)
        env.move_up(0.3)

    env.move_to_left()
    env.idle(50)

    fin_data = env.save_images('fin')
    ext.show_final_img()

    env.world.pause()


while simulation_app.is_running():
    env.world.step(render=True)

simulation_app.close()