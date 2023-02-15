from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})
# 28.53s: False
# 27.53s: True

import omni
from omni.isaac.core import World
import numpy as np
import os
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, UsdPhysics

from pxr import UsdPhysics, PhysxSchema, UsdGeom, Gf, UsdLux, Usd, Sdf

from omni.isaac.core.robots import Robot

from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper_Properties, Surface_Gripper
from omni.isaac.dynamic_control import _dynamic_control
from robot.tools.omni_tools import *

# from tools.ycb import get_random_name
from tools.graspnet import get_random_name
import time
from omni.isaac.core.utils.prims import is_prim_path_valid


start = time.time()

my_world = World(stage_units_in_meters=1.0, physics_prim_path="/World/physicsScene")
my_world.scene.add_default_ground_plane()


def test_gripper():

    stage = omni.usd.get_context().get_stage()

    prim_path = "/World/defaultGroundPlane/tmp"
    tmp = stage.DefinePrim( prim_path, "Xform")
    set_translate(tmp, [0, 0.5, 0])

    prim_path = "/World/defaultGroundPlane/tmp/a"
    tmp = stage.DefinePrim( prim_path, "Xform")
    set_translate(tmp, [0, 0, 0])
    
    add_box(stage, prim_path + "/vc", [0.04, 0.04, 0.04], [0, 0, 0.08], [1,0,0,0], [1,1,0], True, True )

    add_box(stage, prim_path + "/obj", [0.04, 0.04, 0.04], [0.0, 0, 0.02], [1,0,0,0], [1,0,0], True, True )
    
    # Gripper properties
    sgp = Surface_Gripper_Properties()
    sgp.d6JointPath = prim_path + "/vc/APP"
    sgp.parentPath = prim_path + "/vc"
    sgp.offset = _dynamic_control.Transform()
    # sgp.offset.p.x = 0
    sgp.offset.p.z = -0.0201
    sgp.offset.r = [0.7071, 0, 0.7071, 0]  # Rotate to point gripper in Z direction
    sgp.gripThreshold = 0.02
    sgp.forceLimit = 1.0e2
    sgp.torqueLimit = 1.0e3
    sgp.bendAngle = np.pi / 4
    sgp.stiffness = 1.0e4
    sgp.damping = 1.0e3
    dc = _dynamic_control.acquire_dynamic_control_interface()
    print(dc)
    gripper = Surface_Gripper(dc)
    gripper.initialize(sgp)
    return gripper


from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
import numpy as np


class CoolController(BaseController):
    def __init__(self):
        super().__init__(name="my_cool_controller")
        # An open loop controller that uses a unicycle model
        return

    def forward(self, command):
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_positions=command)

def set_drive_parameters(drive, target_type, target_value, stiffness=None, damping=None, max_force=None):
    """Enable velocity drive for a given joint"""

    if target_type == "position":
        if not drive.GetTargetPositionAttr():
            drive.CreateTargetPositionAttr(target_value)
        else:
            drive.GetTargetPositionAttr().Set(target_value)
    elif target_type == "velocity":
        if not drive.GetTargetVelocityAttr():
            drive.CreateTargetVelocityAttr(target_value)
        else:
            drive.GetTargetVelocityAttr().Set(target_value)

    if stiffness is not None:
        if not drive.GetStiffnessAttr():
            drive.CreateStiffnessAttr(stiffness)
        else:
            drive.GetStiffnessAttr().Set(stiffness)

    if damping is not None:
        if not drive.GetDampingAttr():
            drive.CreateDampingAttr(damping)
        else:
            drive.GetDampingAttr().Set(damping)

    if max_force is not None:
        if not drive.GetMaxForceAttr():
            drive.CreateMaxForceAttr(max_force)
        else:
            drive.GetMaxForceAttr().Set(max_force)

class DataGenerate(object):
    def __init__(self, world, ycb_folder="E:/dataset/ycb", save_folder="E:/dataset/tap/train", object_num=10, sample_range=[0,6], start_num=0) -> None:
        self.world = world
        self.save_folder = save_folder
        self.ycb_folder = ycb_folder
        self.object_num = object_num
        self.sample_range = sample_range
        self.all_prims = []
        self.all_paths = []
        self.names = []

        self.state_num = start_num
    
    # def set_controller(self):
        # dc = _dynamic_control.acquire_dynamic_control_interface()
        # articulation = dc.get_articulation(path)
        # # Call this each frame of simulation step if the state of the articulation is changing.
        # self.dc = dc
        # self.articulation = articulation
        # self.articulation = self.robot.get_articulation_controller()
    def config(self):
        stage = omni.usd.get_context().get_stage()

        PhysxSchema.PhysxArticulationAPI.Get(stage, "/World/pusher").CreateSolverPositionIterationCountAttr(64)
        PhysxSchema.PhysxArticulationAPI.Get(stage, "/World/pusher").CreateSolverVelocityIterationCountAttr(64)

        self.gripper_left = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath( f"/World/pusher/center/c_left"), "linear")
        self.gripper_right = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath( f"/World/pusher/center/c_right"), "linear")
        self.gripper_top = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath( f"/World/pusher/center/c_top"), "linear")
        self.gripper_down = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath( f"/World/pusher/center/c_down"), "linear")

    def init_pusher(self):
        set_drive_parameters(self.gripper_left, "position", 0 )
        set_drive_parameters(self.gripper_right, "position", 0 )
        set_drive_parameters(self.gripper_top, "position", 0 )
        set_drive_parameters(self.gripper_down, "position", 0 )

    def close(self):
        pos = self.gripper_left.GetTargetPositionAttr().Get()
        step = 0.001
        pos = pos-step
        min_value = -0.05
        if pos < min_value:
            pos = min_value
        set_drive_parameters(self.gripper_left, "position", pos )
        set_drive_parameters(self.gripper_right, "position", pos )
        set_drive_parameters(self.gripper_top, "position", pos )
        set_drive_parameters(self.gripper_down, "position", pos )
    
    def open(self):
        pos = self.gripper_left.GetTargetPositionAttr().Get()
        step = 0.001
        pos = pos + step
        if pos > 0.3:
            pos = 0.3
        set_drive_parameters(self.gripper_left, "position", pos )
        set_drive_parameters(self.gripper_right, "position", pos )
        set_drive_parameters(self.gripper_top, "position", pos )
        set_drive_parameters(self.gripper_down, "position", pos )
    
    def add_model(self, obj_name, position=[0,0,0], orientation=[1, 0, 0, 0], scale=[1,1,1]):
        YCB_DIRECTORY = "E:/dataset/ycb"

        if 'ycb' in self.ycb_folder:
            usd_path = f'{self.ycb_folder}/{obj_name}/google_16k/text.usd'
        else:
            usd_path = f'{self.ycb_folder}/{obj_name}/omni/simple.usd'
            
        prim_path = f"/World/obj_{obj_name}"
        
        prim = load_obj_usd(usd_path, prim_path, position, orientation, scale, set_rigid=True, kinematic=False)
        return prim

    def generate_ycb(self):

        remove(self.all_paths)
        
        w_range = 0.3
        o_range = 0.25
        container_weight = 0.05
        support_translate = np.array([0,0,0])
        stage = omni.usd.get_context().get_stage()

        height = w_range * 4
        half_height = height/2

        pusher_path = "/World/pusher"
        if not is_prim_path_valid(pusher_path):
            # w1 = create_pusher( stage, "/World/W1", pos=support_translate + [ w_range/2 + container_weight*2, 0, half_height ], size=[ container_weight, w_range * 1, height] , axis='X')
            # w2 = create_pusher( stage, "/World/W2", pos=support_translate - [ w_range/2 + container_weight*2, 0, -(half_height) ], size=[ container_weight, w_range * 1, height] , axis='X')
            # w3 = create_pusher( stage, "/World/W3", pos=support_translate + [ 0, w_range/2 + container_weight*2, half_height ], size=[w_range * 1, container_weight, height] , axis='Y')
            # w4 = create_pusher( stage, "/World/W4", pos=support_translate - [ 0, w_range/2 + container_weight*2, -(half_height) ], size=[w_range * 1, container_weight, height], axis='Y' )
            
            # self.articulation = CoolController()
            pusher = load_obj_usd("./assets/pusher/pusher/pusher.usd", pusher_path, scale=(1, 1, 1), translate=[0, 0, w_range])
            # self.robot = self.world.scene.add(Robot(prim_path=pusher_path, name='pusher'))
            
            self.config()
            # self.walls = [w1,w2,w3,w4]

            # add_box( stage, "/World/W1", position=support_translate + [ w_range/2 + container_weight*2, 0, half_height ], orientation=[1,0,0,0], size=[ container_weight, w_range*2, height], color=[0,0.1,0.7] )
            # add_box( stage, "/World/W2", position=support_translate - [ w_range/2 + container_weight*2, 0, -(half_height) ], orientation=[1,0,0,0], size=[ container_weight, w_range*2, height], color=[0,0.1,0.7] )
            # add_box( stage, "/World/W3", position=support_translate + [ 0, w_range/2 + container_weight*2, half_height ], orientation=[1,0,0,0], size=[w_range*2, container_weight, height], color=[0,0.1,0.7] )
            # add_box( stage, "/World/W4", position=support_translate - [ 0, w_range/2 + container_weight*2, -(half_height) ], orientation=[1,0,0,0], size=[w_range*2, container_weight, height], color=[0,0.1,0.7] )

        names = get_random_name( self.sample_range, self.object_num)

        all_paths = []
        all_prims = []
        for i, name in enumerate(names):
            rand_pos = (np.random.rand(3) - 0.5) * o_range
            rand_pos[2] = 0.2 * i + 0.25

            prim = self.add_model(name, rand_pos + support_translate)
            all_prims.append(prim)
            all_paths.append(prim.GetPrimPath())

        self.all_prims = all_prims
        self.all_paths = all_paths
        self.names = names

    def load_ycb(self, data_index, offset=[0,0,0]):
        data_path = os.path.join(self.save_folder, "%d.npy" % data_index)

        data = np.load(data_path, allow_pickle=True).item()
        names = data['name']
        mats = data['mat']

        remove(self.all_paths)

        all_paths = []
        all_prims = []
        for i, name in enumerate(names):
            mat = mats[i]
            mat[:3,3] += offset
            prim = self.add_model(name)

            set_transform(prim, mat)
            
            all_prims.append(prim)
            all_paths.append(prim.GetPrimPath())
        
        self.all_prims = all_prims
        self.all_paths = all_paths
        self.names = names
    
    def remove_walls(self):
        remove(["/World/W1", "/World/W2", "/World/W3", "/World/W4"])
    
    def record_state(self):

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        
        save_path = os.path.join(self.save_folder, "%s" % (self.state_num))
        state = {}
        state['mat'] = []
        state['name'] = self.names
        for i, prim in enumerate(self.all_prims):
            mat = get_transform(prim)
            state['mat'].append(mat)
        
        np.save( save_path, state)
        self.state_num += 1

train_num = 2
# data = DataGenerate("E:/dataset/ycb", "E:/dataset/tap/train/")
data = DataGenerate( my_world, "E:/dataset/grasp_net/models", "E:/dataset/tap/train", start_num=5)

my_world.reset()

is_load = False

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()

        if not is_load:
            if train_num > 0:
                i += 1
                if i == 1:
                    data.generate_ycb()
                    data.init_pusher()
                    # my_world.pause()

                elif i < 200:
                    continue

                elif i < 500:
                    data.close()

                elif i < 700:
                    data.open()

                elif i < 1000:
                    data.close()
                elif i < 1200:
                    data.open()

                elif i == 1200:
                    print(train_num, ' ====')
                    # data.record_state()
                # elif i == 2000:
                    i = 0
                    train_num -= 1

        else:
            data.load_ycb(0, [0, 0, 0]) 

print(time.time() - start, " s ---------------")

simulation_app.close()