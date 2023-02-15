import os
import numpy as np
from PIL import Image

from scipy.spatial.transform import Rotation

from robot.tools.omni_tools import *

from robot.scene.scene import Scene

from omni.isaac.core import World

class Env(object):
    def __init__(self, render=True, save_folder="./images") -> None:

        self.save_folder = save_folder

        world = World(stage_units_in_meters=1.0, physics_prim_path="/World/physicsScene")
        world.scene.add_default_ground_plane()

        self.world = world
        self.max_loop_num = 1000

        self.scene = Scene(world)

        self.use_robot = True
        self.render = render

        self.scene.init_scene()
        self.scene.set_obs_visible(False)
        
        self.world.reset()
        
        self.is_start = False

    def reset(self):
        self.scene.reset()

    def world_step(self, step=1, render=True):
        if self.world.is_playing():
            for _ in range(step):
                self.world.step(render=render)

    def idle(self, step=1, render=True):
        if self.use_robot:
            self.scene.update_state()
        self.world_step(step, render)

    def robot_on(self):
        self.scene.robot.moving_on()
    
    def save_images(self, prefix="cam"):
        print("Take image")
        rgb, dep = self.scene.take_images()
        cam = self.scene.cam.intrinsic.copy()
        c2w = self.scene.cam.extrinsic.copy()

        os.makedirs(self.save_folder, exist_ok=True)

        np.save( os.path.join(self.save_folder, f'{prefix}_cam.npy'), cam )
        np.save( os.path.join(self.save_folder, f'{prefix}_c2w.npy'), c2w )
        np.save( os.path.join(self.save_folder, f'{prefix}_dep.npy'), dep )
        Image.fromarray(rgb, mode='RGBA').save( os.path.join(self.save_folder, f'{prefix}_rgb.png') )

        camera_params = {}
        # camera_params['x_offset'] = cam[0,0]
        # camera_params['y_offset'] = cam[1,1]
        # camera_params['fx'] = cam[0,2]
        # camera_params['fy'] = cam[1,2]
        camera_params['c2w'] = c2w
        camera_params['cam'] = cam

        return rgb[:,:,:3][:,:,::-1].copy(), dep, camera_params

    def move_up(self, offset=0.1, render=True):
        self.robot_on()
        is_stop = False
        loop_num = 0
        while is_stop == False and loop_num < self.max_loop_num:
            self.scene.robot.move_up(offset)
            is_stop = self.scene.robot.target_state is None
            self.scene.update_state()

            self.world_step(render=render)
            loop_num += 1

    def move_to_init(self, render=True):
        self.robot_on()
        is_stop = False
        loop_num = 0
        while is_stop == False and loop_num < self.max_loop_num:
            self.scene.robot.to_init_state()
            is_stop = self.scene.robot.target_state is None
            self.scene.update_state()

            self.world_step(render=render)
            loop_num += 1

    def move_to_mat(self, mat, offset=0, render=True):
        self.robot_on()
        is_stop = False
        loop_num = 0
        while is_stop is False and loop_num < self.max_loop_num:
            self.scene.robot.move_to_mat(mat, offset)
            is_stop = self.scene.robot.target_state is None
            self.scene.update_state()

            self.world_step(render=render)
            loop_num += 1
    
    def pick_and_placce(self, grasp_mat, place_mat, render=True):
        self.move_to_mat(grasp_mat, 0.1, render=render)
        self.gripper_close(render=render)
        self.move_up(0.3, render=render)
        self.move_to_mat(place_mat, 0.4, render=render)
        self.gripper_open(render=render)
        self.move_up(0.3, render=render)

    def gripper_close(self, render=True):
        self.scene.robot.set_gripper_close()
        self.idle(20, render)
        
    def gripper_open(self, render=True):
        self.scene.robot.set_gripper_open()
        self.idle(1, render)
        self.scene.robot.set_gripper_stop()
    
    def move_to_left(self):
        mat = np.eye(4)
        mat[:3,3] = (0.127126, 0.126619, 0.445994)
        mat[:3,:3] = Rotation.from_rotvec(np.pi * np.array([1,0,0])).as_matrix()
        self.move_to_mat(mat)
    
    def get_pick_mat(self, points_list):
        pick_mats = []

        for points in points_list:
            p = points[points[:,2] > 0.01]
            z = p[:,2].max()
            x, y = p[:,:2].mean(axis=0)
            pick_pos = np.array([x,y,z])
            
            mat = np.eye(4)
            mat[:3,:3] = Rotation.from_rotvec(np.pi * np.array([1,0,0])).as_matrix()
            mat[:3,3] = pick_pos
            pick_mats.append(mat)

        return pick_mats

    def test(self):
        mat = np.eye(4)
        mat[:3,3] = [0.4319, -0.008, 0.0906]
        mat[:3,:3] = Rotation.from_rotvec(np.pi * np.array([1,0,0])).as_matrix()
        self.move_to_mat(mat)
        self.gripper_close()
        self.idle(200)
        self.move_up()
        self.move_to_left()
        self.gripper_open()

    def run(self):

        self.reset()
        self.scene.load_objects()

        self.idle(200)
        self.save_images('tg')

        self.move_to_left()

        self.reset()
        self.scene.load_objects_2()
        self.idle(200)
        self.save_images('sc')


        self.world.pause()
