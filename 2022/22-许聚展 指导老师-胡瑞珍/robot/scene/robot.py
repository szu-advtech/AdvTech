
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation import ArticulationMotionPolicy

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper_Properties, Surface_Gripper
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.robots import Robot
from omni.isaac.manipulators import SingleManipulator
from scipy.spatial.transform import Rotation as R
import os
import numpy as np
import omni

from robot.tools.omni_tools import *

class UR5(object):
    def __init__(self, assets, world, task_prim_path, position=[0,0,0], orientation=[1,0,0,0], scale=[1,1,1], offset=[0,0,0]) -> None:

        self.MOVING_STATE = {
            "to_offset": 0,
            "to_target": 1,
            "stop": 2
        }
        
        # TODO add attach method

        self.use_parallel = False

        self.name = get_unique_name( world.scene, "ur5")
        
        rmp_config_dir = os.path.join(assets, "ur5")

        if self.use_parallel:
            self.urdf_path = os.path.join(rmp_config_dir, "ur5_gripper.urdf")
            self.usd_path = os.path.join(rmp_config_dir, "usd", "ur5_gripper.usd")
        else:
            self.urdf_path = os.path.join(rmp_config_dir, "ur5_suction.urdf")
            self.usd_path = os.path.join(rmp_config_dir, "usd", "ur5_suction.usd")

        self.robot_description_path = os.path.join(rmp_config_dir, "config", 'robot_descriptor.yaml')
        self.rmpflow_config_path = os.path.join(rmp_config_dir, "config", 'rmpflow_config.yaml')

        self.end_effector_frame_name = "gripper_center"

        self.world = world
        
        self.task_prim_path = task_prim_path

        self.offset = offset

        self.position = np.array(position).astype('float') + offset
        self.orientation = np.array(orientation).astype('float')
        self.scale = np.array(scale).astype('float')

        self.gripper_state = 0

        self.target_state = None
        self.moving_state = self.MOVING_STATE['to_offset']
        
        self.obstacles = []
        self.init_state = np.array([0, -np.deg2rad(30), -np.deg2rad(100), -np.deg2rad(120), np.deg2rad(90), 0])

        self.load_robot()
        self.set_controller()

    def load_robot(self):
        self.prim_path = self.task_prim_path + "/ur5"
        self.target_prim_path = self.task_prim_path + "/ur5_target"
        
        stage = omni.usd.get_context().get_stage()

        self.prim = load_obj_usd( usd_path=self.usd_path, prim_path=self.prim_path, \
            translate=self.position, orientation=self.orientation, scale=self.scale )

        self.gripper_center_prim = get_prim( self.prim_path + "/" + self.end_effector_frame_name )
        set_translate(self.gripper_center_prim, [0,0,0.02])

        # add target
        self.target_prim = stage.DefinePrim( self.target_prim_path, "Xform")
        set_translate(self.target_prim, [ 0, 0, 0 ])

    def set_controller(self):
        from omni.isaac.universal_robots.controllers import StackingController as UR10StackingController
        if self.use_parallel:
            gripper = ParallelGripper(
                #We chose the following values while inspecting the articulation
                end_effector_prim_path= self.prim_path + "/gripper_base" ,
                joint_prim_names=["gb_gl", "gb_gr"],
                joint_opened_positions=np.array([0, 0]),
                joint_closed_positions=np.array([0.0275, 0.0275]),
                action_deltas=np.array([-0.0275, -0.0275]),
            )
            
            #define the manipulator
            self.robot = self.world.scene.add(
                SingleManipulator(prim_path=self.prim_path, name=self.name,
                    end_effector_prim_name="gripper_base", gripper=gripper))
        else:
            
            # Gripper properties
            sgp = Surface_Gripper_Properties()
            sgp.d6JointPath = self.prim_path + "/gripper_vacuum/SurfaceGripper"
            sgp.parentPath = self.prim_path + "/gripper_vacuum"
            sgp.offset = _dynamic_control.Transform()
            sgp.offset.p.x = 0
            sgp.offset.p.y = 0
            sgp.offset.p.z = 0.005 + 0.02
            sgp.offset.r = [0.7071, 0, 0.7071, 0]  # Rotate to point gripper in Z direction
            sgp.gripThreshold = 0.02
            sgp.forceLimit = 1.0e3
            sgp.torqueLimit = 1.0e4
            # sgp.forceLimit = 1.0e2
            # sgp.torqueLimit = 1.0e3
            sgp.bendAngle = np.pi / 2
            sgp.stiffness = 1.0e4
            sgp.damping = 1.0e3
            dc = _dynamic_control.acquire_dynamic_control_interface()
            gripper = Surface_Gripper(dc)
            gripper.initialize(sgp)
            
            self.robot = self.world.scene.add(Robot(prim_path=self.prim_path, name=self.name))
            
            self.robot.gripper = gripper


        self.rmpflow = RmpFlow(
            robot_description_path = self.robot_description_path,
            urdf_path = self.urdf_path,
            rmpflow_config_path = self.rmpflow_config_path,
            end_effector_frame_name = self.end_effector_frame_name,
            evaluations_per_frame = 5,
            ignore_robot_state_updates=True
        )

        self.rmpflow.set_robot_base_pose( get_translate(self.prim), get_orientation(self.prim) )

        # self.rmpflow.visualize_collision_spheres()
        # self.rmpflow.visualize_end_effector_position()

        physics_dt = 1/60.
        self.articulation_rmpflow = ArticulationMotionPolicy(self.robot, self.rmpflow, physics_dt)
        self.articulation_controller = self.robot.get_articulation_controller()
    
    def set_gripper_open(self):
        self.set_gripper_state(1)

    def set_gripper_close(self):
        self.set_gripper_state(-1)

    def set_gripper_stop(self):
        self.set_gripper_state(0)

    def set_gripper_state(self, state: int):
        self.gripper_state = state

    def gripper_close(self):
        if self.use_parallel:
            gripper_positions = self.robot.gripper.get_joint_positions()
            self.robot.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + 0.0001, gripper_positions[1] + 0.0001]))
                # ArticulationAction(joint_positions=[0.008, 0.008]))
        else:
            self.robot.gripper.close()
    
    def gripper_open(self):
        # gripper_positions = self.robot.gripper.get_joint_positions()
        # self.robot.gripper.apply_action(
        #     ArticulationAction(joint_positions=[gripper_positions[0] - 0.0001, gripper_positions[1] - 0.0001]))
        self.robot.gripper.open()

    def add_obstacle(self, prim):
        self.obstacles.append(prim)
        self.rmpflow.add_obstacle(prim)

    def remove_obstacle(self, prim):
        self.obstacles.remove(prim)
        self.rmpflow.remove_obstacle(prim)
    
    def to_init_state(self):
        if self.target_state is None and self.moving_state != self.MOVING_STATE['stop']:
            self.set_target(target_joints=self.init_state)
            self.moving_state = self.MOVING_STATE['stop']

    def set_target(self, target_position=None, target_orientation=None, target_joints=None):

        if target_joints is not None:
            self.target_state = np.array(target_joints)
        else:
            end_prim = get_prim(self.prim_path + "/gripper_center")
            if target_position is None:
                position = get_translate(end_prim)
                target_position = position + self.position
            else:
                target_position = np.array(target_position).astype('float') + self.position
            set_translate( self.target_prim, target_position )

            if target_orientation is None:
                target_orientation = get_orientation(end_prim)
            else:
                target_orientation = np.array(target_orientation).astype('float')
            set_orientation( self.target_prim, target_orientation )

            self.target_state = [ target_position, target_orientation ]
        
    def move_to_mat(self, mat, offset=0):

        x,y,z,w = Rotation.from_matrix(mat[:3,:3]).as_quat()

        target_position = mat[:3,3]
        target_orientation = np.array([w,x,y,z])
        self.move_to(target_position, target_orientation, offset)

    def move_to(self, target_position, target_orientation, offset=0):
        if self.target_state is None and self.moving_state != self.MOVING_STATE['stop']:
            if self.moving_state == self.MOVING_STATE['to_offset'] and offset != 0:
                w, x, y, z = target_orientation
                rot = R.from_quat([x,y,z,w]) # this use x,y,z,w
                z = np.array([0,0,-1])
                direction = rot.apply(z)
                offset_pos = target_position + direction * offset

                self.set_target(offset_pos, target_orientation)
                self.moving_state = self.MOVING_STATE['to_target']
            else:
                # print("setting")
                self.set_target(target_position, target_orientation)
                self.moving_state = self.MOVING_STATE['stop']

    def move_up(self, z_offset):
        if self.target_state is None and self.moving_state != self.MOVING_STATE['stop']:
            end_prim = get_prim(self.prim_path + "/gripper_center")
            position = get_translate(end_prim)
            position[2] += z_offset
            self.move_to(position, None)

    def moving_on(self):
        self.moving_state = self.MOVING_STATE['to_offset']

    def stop(self):
        self.moving_state = self.MOVING_STATE['stop']

    def is_stop(self):
        return self.moving_state == self.MOVING_STATE['stop']

    def check_valid_target(self, position, quat, joint_name='gripper_center'):
        # quat = w,x,y,z
        ret = self.rmpflow.get_kinematics_solver().compute_inverse_kinematics(joint_name, np.array(position), np.array(quat))
        return ret[1]

    def update_state(self):
        if self.target_state is not None:
            if len(self.target_state) == 2:
                self.rmpflow.set_end_effector_target(
                    target_position=self.target_state[0],
                    target_orientation=self.target_state[1]
                )
            else:
                self.rmpflow.set_cspace_target( self.target_state )

            self.rmpflow.update_world(self.obstacles)
            
            actions = self.articulation_rmpflow.get_next_articulation_action()

            count = len(actions.joint_velocities)
            for v in actions.joint_velocities:
                if v is None or abs(v) < 1e-2:
                    count -= 1

            if count == 0:
                # print('stop')
                self.target_state = None
            else:
                self.articulation_controller.apply_action(actions)

        if self.gripper_state != 0:
            if self.gripper_state == 1:
                self.gripper_open()
            elif self.gripper_state == -1:
                self.gripper_close()
