import omni

from robot.scene.camera import Camera
from robot.scene.robot import UR5
import numpy as np
import os

from robot.tools.omni_tools import *


YCB_DIRECTORY = "E:/dataset/ycb"
# GN_DIRECTORY = "E:/dataset/grasp_net/models"
ASSETS_DIR = "E:/workspace/visual_match/robot/assets"

class Scene(object):
    def __init__(self, world, robot_height=0.2, offset=[0,0,0]):
        
        self.task_prim_path = '/World/Scene'
        self.world = world
        self.stage = omni.usd.get_context().get_stage()

        self.offset = np.array(offset).astype("float")
        self.robot_height = robot_height

        self.all_prims = []
        self.all_paths = []
        self.names = []

        # support size
        self.support_translate = np.array([0.6, 0, -self.robot_height/2])
        self.support_size = np.array([0.7, 0.7, 1])

    def init_scene(self):
        
        self.scene_prim = self.stage.DefinePrim( self.task_prim_path, "Xform")
        clean_transform(self.scene_prim)

        self.add_robot()

        self.add_support()

        self.add_camera()

        self.obj_prim_path = f"{self.robot.prim_path}/OBJ"
        self.obs_prim_path = f"{self.robot.prim_path}/OBS"

    def add_robot(self):
        self.robot = UR5( ASSETS_DIR, self.world, self.task_prim_path, position=[0, 0, self.robot_height], offset=self.offset)
      
        self.base_position = self.robot.position

        base_pos = self.base_position - [0,0, self.robot_height/2.0]
        add_box(self.stage, self.task_prim_path + "/ur5_base", [0.2, 0.2, self.robot_height], base_pos, [1,0,0,0], [0.8,0.8,0.8])
        
        self.obs_prim = self.stage.DefinePrim( self.robot.prim_path + "/OBS", "Xform")
        self.robot.to_init_state()

    def set_obs_visible(self, visible=True):
        set_visible(self.obs_prim, visible)

    def add_model(self, obj_name, position=[0,0,0], orientation=[1, 0, 0, 0], scale=[1,1,1], use_convert=False):
        
        if use_convert:
            usd_path = f'{YCB_DIRECTORY}/{obj_name}/google_16k/_converted/text.usd'
        else:
            usd_path = f'{YCB_DIRECTORY}/{obj_name}/google_16k/text.usd'
        
        prim_path = f"{self.robot.prim_path}/OBJ/obj_{obj_name}"
        
        prim = load_obj_usd(usd_path, prim_path, position, orientation, scale, set_rigid=True, kinematic=False)

        prim_path = get_prim_path(prim)
        self.all_prims.append(prim)
        self.all_paths.append(prim_path)
        self.names.append(obj_name[4:].replace('_', ' '))
        
        return prim
    
    def load_objects(self):
        center = self.support_translate + [0, 0, 0.2]
        # self.add_model('026_sponge', position= center+[-0.1, -0.15, 0.05], orientation=[1,0,0,0] )
        self.add_model('008_pudding_box', position= center+[0.1, -0.2, 0.1] )
        self.add_model('011_banana', position= center+[0, -0.1, 0.1] )
        self.add_model('013_apple', position= center+[-0.1, 0, 0.1] )
        self.add_model('014_lemon', position= center+[-0.22, 0.0, 0.1] )
    
    def load_objects_2(self):
        center = self.support_translate + [0, 0, 0.2]
        # self.add_model('026_sponge', position= center+[0.07, -0.15, 0.05], orientation=[1,0,0,0] )
        self.add_model('008_pudding_box', position= center+[0.15, 0.16, 0.05], orientation=[0.62,0,0,0.78], use_convert=True )
        self.add_model('011_banana', position= center+[-0.16, 0.2, 0.05], orientation=[0.89,0,0,-0.438], use_convert=True )
        self.add_model('013_apple', position= center+[-0.15, 0.1, 0.05], use_convert=True )
        self.add_model('014_lemon', position= center+[-0.05, 0.13, 0.05], orientation=[-0.597, 0, 0, 0.797], use_convert=True )

    def add_support(self):
        # object support
        init_support_size = np.array([1, 1, self.robot_height])
        
        load_obj_usd( os.path.join( ASSETS_DIR, "support", "support_flat.usd"), \
            self.robot.prim_path + "/BASE/support", self.support_translate, scale=self.support_size, set_rigid=False, kinematic=False, set_collision="none")
        collision_cube = get_prim(self.robot.prim_path + "/BASE/support/geo/geo")
        collision_cube.GetAttribute("physics:approximation").Set("none")

        self.add_obstacle("SUP",  self.support_translate, scale=init_support_size * self.support_size  )
    
    def add_obstacle(self, name, translate, orientation=[1,0,0,0], scale=[0.1,0.1,0.1]):
        obs_prim_path = self.robot.prim_path + "/OBS/" + name
        obs = omni.isaac.core.objects.cuboid.VisualCuboid( obs_prim_path, translation=translate, orientation=orientation, color=np.array([0, 1.,0]), size=1)
        set_scale( get_prim(obs_prim_path), scale)
        self.robot.add_obstacle(obs)

    def add_camera(self):
        container_translate = self.support_translate + [0, 0, 0.6]
        self.cam_observe = Camera(self.stage, self.robot.prim_path + '/CAM/camera_observe', [1.14, 0.95, 1.69], [ 37, 0, 140], resolution=(1500, 900)  )
        self.cam = Camera(self.stage, self.robot.prim_path + '/CAM/camera', container_translate, [0,0, 90], focal_length=12 )

    def take_images(self, types=['rgb', 'dep']):
        
        ret = []
        if 'rgb' in types:
            ret.append( self.cam.take_rgb() )
        if 'dep' in types:
            ret.append( self.cam.take_dep() )
        return ret

    def update_state(self):
        self.robot.update_state()
    
    def reset(self):

        remove(self.all_paths)
        
        self.all_prims = []
        self.all_paths = []
        self.names = []

        # self.load_objects()
        # scene_id = np.random.choice(self.scene_list)
        # self.load_scene_objs(scene_id)

