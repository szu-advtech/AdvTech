import omni
from pxr import UsdPhysics, UsdGeom, Gf, Sdf, Usd

from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.objects.cuboid import VisualCuboid
from omni.physx.scripts import utils
from scipy.spatial.transform import Rotation

import carb
import carb.events
# from omni.debugdraw import _debugDraw
import math
import numpy as np

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


def clean_transform(prim):
    # prim.RemoveProperty("xformOp:translate")
    # prim.RemoveProperty("xformOp:orient")
    # prim.RemoveProperty("xformOp:scale")
    prim.GetAttribute("xformOpOrder").Set(['xformOp:translate', 'xformOp:rotateXYZ', 'xformOp:orient', 'xformOp:scale'])
    set_translate(prim, [0,0,0])
    set_orientation(prim, [1,0,0,0])
    set_scale(prim, [1,1,1])

def get_prim_path(prim):
    return prim.GetPrimPath()

def get_attr(prim, op_name, data_type):
    
    xform_ops = prim.GetAttribute("xformOpOrder").Get()
    if xform_ops is None:
        xform = UsdGeom.Xformable(prim)
        prim.GetAttribute("xformOpOrder").Set([op_name])

    attr = prim.GetAttribute(op_name).Get()
    if attr is None:
        prim.CreateAttribute(op_name, data_type, False)
        attr = prim.GetAttribute(op_name).Get()
    return attr

def set_orientation(prim, orientation, use_quatd=True):
    orientation = np.array(orientation).astype("float")

    op_name = "xformOp:rotateXYZ"
    attr = get_attr(prim, op_name, Sdf.ValueTypeNames.Float3)

    op_name = "xformOp:orient"
    attr = get_attr(prim, op_name, Sdf.ValueTypeNames.Quatd)

    if attr is not None:
        if type(attr) == Gf.Quatd:
            orient = Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3])
        else:
            orient = Gf.Quatf(orientation[0], orientation[1], orientation[2], orientation[3])
        prim.GetAttribute(op_name).Set( orient )

def set_rotation(prim, orientation):
    orientation = np.array(orientation).astype("float")

    op_name = "xformOp:rotateXYZ"
    attr = get_attr(prim, op_name, Sdf.ValueTypeNames.Float3)

    if attr is not None:
        if type(attr) == Gf.Vec3f:
            orient = Gf.Vec3f(orientation[0], orientation[1], orientation[2])
        else:
            orient = Gf.Vec3d(orientation[0], orientation[1], orientation[2])
        prim.GetAttribute(op_name).Set( orient )


def set_transform(prim, mat):
    # TODO
    translate = mat[:3,3]
    rot = mat[:3,:3]
    x,y,z,w=Rotation.from_matrix(rot).as_quat()

    set_translate(prim, translate)
    set_orientation(prim, [w,x,y,z])
    
    # prim.CreateAttribute("xformOp:transform", Sdf.ValueTypeNames.Matrix4d, False).Set(Gf.Matrix4d(mat))

def set_translate(prim, translate):
    translate = np.array(translate).astype("float")
    
    op_name = "xformOp:translate"
    attr = get_attr(prim, op_name, Sdf.ValueTypeNames.Float3)

    if type(attr) == Gf.Vec3f:
        trans = Gf.Vec3f(translate[0], translate[1], translate[2])
    else:
        trans = Gf.Vec3d(translate[0], translate[1], translate[2])
    prim.GetAttribute(op_name).Set( trans )

def set_scale(prim, scale):
    scale = np.array(scale).astype("float")
    op_name = "xformOp:scale"
    attr = get_attr(prim, op_name, Sdf.ValueTypeNames.Float3)

    if type(attr) == Gf.Vec3f:
        s = Gf.Vec3f(scale[0], scale[1], scale[2])
    else:
        s = Gf.Vec3d(scale[0], scale[1], scale[2])

    prim.GetAttribute(op_name).Set(s)

def get_orientation(prim):
    orient = prim.GetAttribute("xformOp:orient").Get()
    real = [orient.GetReal()]
    img = list(orient.GetImaginary())
    return np.array(real + img)

def get_transform(prim):
    translate = get_translate(prim)
    w,x,y,z = get_orientation(prim)

    ret = np.eye(4)
    mat = Rotation.from_quat([x,y,z,w]).as_matrix()
    ret[:3,3] = translate
    ret[:3,:3] = mat
    return ret

def get_translate(prim):
    translate = prim.GetAttribute("xformOp:translate").Get()
    return np.array(translate)

def get_scale(prim):
    return prim.GetAttribute("xformOp:scale").Get()

def get_prim(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    return prim

def get_unique_path(prim_path):
    prim_path = find_unique_string_name(
        initial_name=prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
    )
    return prim_path

def get_unique_name(scene, name):
    ret = find_unique_string_name(
        initial_name=name, is_unique_fn=lambda x: not scene.object_exists(x)
    )
    return ret

def load_obj_usd(usd_path, prim_path, translate=[0,0,0], orientation=[1,0,0,0], scale=[1,1,1], set_rigid=False, kinematic=False, set_collision='convexDecomposition'):
    # set_collision / approximationShape:
    #   "none", "convexHull", "convexDecomposition", "boundingCube", "boundingSphere", "meshSimplification"
    
    stage = omni.usd.get_context().get_stage()
    
    prim_path = get_unique_path(prim_path)

    prim = stage.GetPrimAtPath(usd_path)

    if not prim.IsValid():
        prim = stage.DefinePrim( prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)

    if set_rigid:
        utils.setRigidBody(prim, set_collision, kinematic)
    
    elif set_collision is not None:
        utils.setStaticCollider(prim, set_collision)

    clean_transform(prim)
    set_translate(prim, translate)
    set_orientation(prim, orientation)
    set_scale(prim, scale)
    return prim

def add_box(stage, primPath, size, position, orientation, color, collision=True, rigid=False):
    # defaultPrimPath = str(stage.GetDefaultPrim().GetPath())

    cubePrimPath = primPath

    position = Gf.Vec3f( position[0], position[1], position[2])
    orientation = Gf.Quatf(orientation[0], orientation[1], orientation[2], orientation[3])
    color = Gf.Vec3f(color[0], color[1], color[2])
    
    cubeGeom = UsdGeom.Cube.Define(stage, cubePrimPath)
    cubePrim = stage.GetPrimAtPath(cubePrimPath)

    cubeGeom.CreateSizeAttr(1)
    
    scale = Gf.Vec3f( list(size) )
    cubeGeom.AddTranslateOp().Set(position)
    cubeGeom.AddOrientOp().Set(orientation)
    cubeGeom.AddScaleOp().Set(scale)
    cubeGeom.CreateDisplayColorAttr().Set([color])

    if collision:
        UsdPhysics.CollisionAPI.Apply(cubePrim)
        UsdPhysics.MassAPI.Apply(cubePrim)
    
    if rigid:
        utils.setRigidBody(cubePrim, 'convexHull', False)
    
    return cubePrim

def set_visible(prim, visible=True):
    if visible:
        prim.GetAttribute("visibility").Set("inherited")
    else:
        print
        prim.GetAttribute("visibility").Set("invisible")

def remove(primPath):
    omni.kit.commands.execute("DeletePrims", paths=primPath)
    
def set_view_reso(w=640, h=480):
    viewport = omni.kit.viewport_legacy.get_viewport_interface()
    # acquire the viewport window
    viewport_handle = viewport.get_instance("Viewport")
    viewport_window = viewport.get_viewport_window(viewport_handle)
    # Set viewport resolution, changes will occur on next frame
    viewport_window.set_texture_resolution(w, h)


def vec_to_mat(fromVector, toVector):
    # https://blog.csdn.net/doubtfire/article/details/122100943

    fromVector = np.array(fromVector)
    fromVector_e = fromVector / np.linalg.norm(fromVector)

    toVector = np.array(toVector)
    toVector_e = toVector / np.linalg.norm(toVector)

    cross = np.cross(fromVector_e, toVector_e)

    cross_e = cross / np.linalg.norm(cross)

    dot = np.dot(fromVector_e, toVector_e)

    angle = math.acos(dot)
    if angle == 0 or angle == math.pi:
        print("两个向量处于一条直线")
        return [1, 0,0,0]
    else:
        quat = [math.cos(angle/2), cross_e[0]*math.sin(angle/2), cross_e[1]*math.sin(angle/2), cross_e[2]*math.sin(angle/2)]
        # return Rotation.from_quat(quat).as_matrix()
        return quat


def vec_to_matrix( from_v, to_v ):

    from_v = np.array(from_v)
    fromVector_e = from_v / np.linalg.norm(from_v)

    to_v = np.array(to_v)
    toVector_e = to_v / np.linalg.norm(to_v)

    cross = np.cross(fromVector_e, toVector_e)

    vec = cross / np.linalg.norm(cross)

    dot = np.dot(fromVector_e, toVector_e)

    theta = np.math.acos(dot)

    rot = np.zeros((3,3))
    x, y, z = vec

    xx = x**2
    yy = y**2
    zz = z**2

    xy = x*y
    xz = x*z
    
    yz = z*y

    cost = np.math.cos(theta)    
    sint = np.math.sin(theta)    

    rot[0,0] = xx*(1-cost) + cost
    rot[0,1] = xy*(1-cost) + z*sint
    rot[0,2] = xz*(1-cost) - y*sint

    rot[1,0] = xy*(1-cost) - z*sint
    rot[1,1] = yy*(1-cost) + cost
    rot[1,2] = yz*(1-cost) + x*sint

    rot[2,0] = xz*(1-cost) + y*sint
    rot[2,1] = yz*(1-cost) - x*sint
    rot[2,2] = zz*(1-cost) + cost
    return rot


def add_arrow(stage, primPath, start_pos=None, end_pos=None, mat=None, arrow_len=None, radius=0.01, color=[1,0,0]):

    line_path = primPath + '_line'
    arrow_path = primPath + '_arrrow'

    if mat is None:
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        direct = end_pos - start_pos
        arrow_len = np.linalg.norm(direct)
    else:
        start_pos = mat[:3,3]
        direct = mat[:3,:3] @ np.array([0,0,1])
        end_pos = start_pos + direct * arrow_len

    orientation = vec_to_mat([0,0,1], direct / arrow_len)
    position = start_pos + direct / 2

    end_pos += direct / arrow_len * radius
    end_position = Gf.Vec3f( end_pos[0], end_pos[1], end_pos[2] )

    position = Gf.Vec3f( position[0], position[1], position[2])
    orientation = Gf.Quatf(orientation[0], orientation[1], orientation[2], orientation[3])
    color = Gf.Vec3f(color[0], color[1], color[2])

    line_geom = UsdGeom.Cylinder.Define(stage, line_path)
    cone_geom = UsdGeom.Cone.Define(stage, arrow_path)

    # line_geom.GetExtentAttr('radius').Set( radius * 1.5 )
    # line_geom.GetExtentAttr('height').Set( radius * 3 )
    omni.kit.commands.execute('ChangeProperty',
        prop_path=Sdf.Path( line_path + '.radius'),
        value=radius, prev=1)
    omni.kit.commands.execute('ChangeProperty',
        prop_path=Sdf.Path( line_path + '.height'),
        value=arrow_len, prev=1)
    
    line_geom.AddTranslateOp().Set(position)
    line_geom.AddOrientOp().Set(orientation)
    line_geom.CreateDisplayColorAttr().Set([color])
    line_geom.AddScaleOp().Set(Gf.Vec3f(1.,1.,1.))
    
    omni.kit.commands.execute('ChangeProperty',
        prop_path=Sdf.Path( arrow_path + '.radius'),
        value=radius * 1.5, prev=2)
    omni.kit.commands.execute('ChangeProperty',
        prop_path=Sdf.Path( arrow_path + '.height'),
        value=radius * 3, prev=2)
    
    cone_geom.AddTranslateOp().Set(end_position)
    cone_geom.AddOrientOp().Set(orientation)
    cone_geom.AddScaleOp().Set(Gf.Vec3f(1.,1.,1.))
    cone_geom.CreateDisplayColorAttr().Set([color])

    line_prim = stage.GetPrimAtPath(line_path)

    return line_prim



def drawArrow (p1, p2, color=0xffffc000):
    # https://github.com/ft-lab/omniverse_sample_scripts/blob/7f4406520da9abcb93c5ffa73bdcff8a2dfad7e5/UI/DebugDraw/UseDebugDraw.py

    _debugDraw = _debugDraw.acquire_debug_draw_interface()
    _debugDraw.draw_line(carb.Float3(p1[0], p1[1], p1[2]), color, carb.Float3(p2[0], p2[1], p2[2]), color)
    P1 = Gf.Vec3f(p1[0], p1[1], p1[2])
    P2 = Gf.Vec3f(p2[0], p2[1], p2[2])
    vDir = P2 - P1
    lenV = vDir.GetLength()
    vDir /= lenV

    v1_2 = Gf.Vec4f(vDir[0], vDir[1], vDir[2], 1.0)
    v2_2 = Gf.Vec4f(0, 1, 0, 1.0)
    v3_2 = Gf.HomogeneousCross(v1_2, v2_2)

    vDirX = Gf.Vec3f(v3_2[0], v3_2[1], v3_2[2]).GetNormalized()
    vD1 = (vDir + vDirX).GetNormalized() * (lenV * 0.1)
    vD2 = (vDir - vDirX).GetNormalized() * (lenV * 0.1)

    pp = P1 + vD1
    _debugDraw.draw_line(carb.Float3(pp[0], pp[1], pp[2]), color, carb.Float3(P1[0], P1[1], P1[2]), color)
    pp = P1 + vD2
    _debugDraw.draw_line(carb.Float3(pp[0], pp[1], pp[2]), color, carb.Float3(P1[0], P1[1], P1[2]), color)

    pp = P2 - vD1
    _debugDraw.draw_line(carb.Float3(pp[0], pp[1], pp[2]), color, carb.Float3(P2[0], P2[1], P2[2]), color)
    pp = P2 - vD2
    _debugDraw.draw_line(carb.Float3(pp[0], pp[1], pp[2]), color, carb.Float3(P2[0], P2[1], P2[2]), color)
