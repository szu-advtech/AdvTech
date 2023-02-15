import omni
import math
import omni.kit.commands
from pxr import Sdf, Gf

import omni.replicator.core as rep
import numpy as np
from robot.tools.omni_tools import *


class Camera(object):
    def __init__(self, stage, prim_path, translate, orientation, focal_length=18.14, focus_distance=400, resolution=(640, 480)) -> None:

        self.prim_path = prim_path
        self.stage = stage
        self.resolution = resolution
        self.camera = self.add_camera(stage, prim_path, translate, orientation, focal_length, focus_distance)

        self.render_product = rep.create.render_product(prim_path, resolution=resolution)
        self.rgb_anno = rep.AnnotatorRegistry.get_annotator("rgb")
        self.dep_anno = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.rgb_anno.attach([self.render_product])
        self.dep_anno.attach([self.render_product])
        
    def add_camera(self, stage, prim_path, translate, orientation, focal_length, focus_distance):

        cameraGeom = UsdGeom.Camera.Define(stage, prim_path)
        cam = get_prim(prim_path)
        
        cam.GetAttribute('focalLength').Set(focal_length)
        cam.GetAttribute('focusDistance').Set(focus_distance)
        cam.GetAttribute('fStop').Set(0.0)
        cam.GetAttribute('projection').Set('perspective')
        cam.GetAttribute('clippingRange').Set(Gf.Vec2f(0.01, 10000))

        if len(orientation) == 4:
            w,x,y,z = orientation
            orientation = list(Rotation.from_quat([x,y,z,w]).as_euler('XYZ'))
            orientation = [ np.rad2deg(ang) for ang in orientation ]

        rotation = Rotation.from_euler('XYZ', [ np.deg2rad(ang) for ang in orientation ]).as_matrix()

        # Set position.
        UsdGeom.XformCommonAPI(cameraGeom).SetTranslate(list(translate))
        # Set rotation.
        UsdGeom.XformCommonAPI(cameraGeom).SetRotate(list(orientation), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        # Set scale.
        UsdGeom.XformCommonAPI(cameraGeom).SetScale((1, 1, 1))

        # omni.kit.commands.execute('ChangeProperty',
        #     prop_path=Sdf.Path(f'{prim_path}.xformOp:rotateXYZ'),
        #     value=Gf.Vec3d(orientation),
        #     prev=Gf.Vec3d(0,0,0))
        
        

        width, height = self.resolution
        
        horiz_aperture = cam.GetAttribute("horizontalAperture").Get()
        
        # https://forums.developer.nvidia.com/t/creating-a-custom-camera-on-isaac-sim-app/187375/2
        # https://forums.developer.nvidia.com/t/camera-intrinsic-matrix/213799
        horizontal_fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
        vertical_fov = (height / width * horizontal_fov)
        focal_x = (width / 2.0) / np.tan(horizontal_fov / 2.0)
        focal_y = (height / 2.0) / np.tan(vertical_fov / 2.0)
        center_x = width * 0.5
        center_y = height * 0.5

        self.intrinsic = np.array([
                                [focal_x, 0, center_x],
                                 [0, focal_y, center_y],
                                 [0, 0, 1]])

        self.extrinsic = np.eye(4)
        self.extrinsic[:3,:3] = rotation
        self.extrinsic[:3,3] = translate

        return cam

    def take_rgb(self):
        rgb_data = self.rgb_anno.get_data()
        rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
        return rgb_image_data
    
    def take_dep(self):
        
        data = self.dep_anno.get_data()
        # Get size.
        hei, wid = data.shape[:2]
        # Store data (buff[hei][wid]).
        buff = np.frombuffer(data, np.float32).reshape(hei, wid)
        buff[buff == buff.max()] = 0
        return buff

    def cam_to_world(self, points):
        rot = self.extrinsic[:3,:3]
        pos = self.extrinsic[:3, 3]

        points = (rot @ points.transpose()).transpose() + pos
        return points

        
    def get_camera(self):
        # viewport = omni.kit.viewport_legacy.get_viewport_interface()
        # viewportWindow = viewport.get_viewport_window()
        # cameraPath = viewportWindow.get_active_camera()
        
        # Get stage.
        # stage = omni.usd.get_context().get_stage()

        #time_code = omni.timeline.get_timeline_interface().get_current_time() * stage.GetTimeCodesPerSecond()
        time_code = Usd.TimeCode.Default()

        # Get active camera.
        cameraPrim = self.stage.GetPrimAtPath(self.prim_path)
        if cameraPrim.IsValid():
            camera  = UsdGeom.Camera(cameraPrim)        # UsdGeom.Camera
            cameraV = camera.GetCamera(time_code)       # Gf.Camera
            print("Aspect : " + str(cameraV.aspectRatio))
            print("fov(H) : " + str(cameraV.GetFieldOfView(Gf.Camera.FOVHorizontal)))
            print("fov(V) : " + str(cameraV.GetFieldOfView(Gf.Camera.FOVVertical)))
            print("FocalLength : " + str(cameraV.focalLength))
            print("World to camera matrix : " + str(cameraV.transform))

            viewMatrix = cameraV.frustum.ComputeViewMatrix()
            print("View matrix : " + str(viewMatrix))

            viewInv = viewMatrix.GetInverse()

            # Camera position(World).
            cameraPos = viewInv.Transform(Gf.Vec3f(0, 0, 0))
            print("Camera position(World) : " + str(cameraPos))

            # Camera vector(World).
            cameraVector = viewInv.TransformDir(Gf.Vec3f(0, 0, -1))
            print("Camera vector(World) : " + str(cameraVector))

            projectionMatrix = cameraV.frustum.ComputeProjectionMatrix()
            print("Projection matrix : " + str(projectionMatrix))

            #cv = CameraUtil.ScreenWindowParameters(cameraV)
            #print(cv.screenWindow)
            print(self.intrinsic)
            print(self.extrinsic)