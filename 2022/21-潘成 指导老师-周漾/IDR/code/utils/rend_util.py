import cv2
import numpy as np
import imageio
import skimage
import jittor as jt

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def load_rgb(path):
    # print("path:          ", path)
    img = imageio.imread(path)
    # print("099999999999999999")
    
    img = skimage.img_as_float32(img)
    # print(img.shape)
    # pixel values between [-1,1]
    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    # print(img.shape)
    
    return img

def load_mask(path):
    # /print("path:          ", path)
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    object_mask = alpha > 127.5

    return object_mask

def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape

    cam_loc = cam_loc.unsqueeze(-1)
    # print("jt.nn.bmm(ray_directions, cam_loc).shape: ", jt.nn.bmm(ray_directions, cam_loc).shape)
    ray_cam_dot = jt.nn.bmm(ray_directions, cam_loc).squeeze(0).squeeze(1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0
    sphere_intersections = jt.zeros((n_imgs * n_pix, 2)).float()
    sphere_intersections[mask_intersect] = jt.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * jt.Var([-1, 1]).float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_(min_v=0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect


def get_camera_params(uv, pose, intrinsics):
    # if pose.shape[1] == 7: #In case of quaternion vector representation
    #     cam_loc = pose[:, 4:]
    #     R = quat_to_rot(pose[:,:4])
    #     p = torch.eye(4).repeat(pose.shape[0],1,1).float()
    #     p[:, :3, :3] = R
    #     p[:, :3, 3] = cam_loc
    # else: # In case of pose matrix representation
    cam_loc = pose[:, :3, 3]
    p = pose

    batch_size, num_samples, _ = uv.shape

    depth = jt.ones((batch_size, num_samples))
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = jt.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = jt.misc.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc

def lift(x, y, z, intrinsics):
    # parse intrinsics
    # intrinsics = intrinsics
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return jt.stack((x_lift, y_lift, z, jt.ones_like(z)), dim=-1)

def get_depth(points, pose):
    ''' Retruns depth from 3D points according to camera pose '''
    batch_size, num_samples, _ = points.shape
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        pose = jt.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).float()
        pose[:, :3, 3] = cam_loc
        pose[:, :3, :3] = R

    points_hom = jt.concat((points, jt.ones((batch_size, num_samples, 1))), dim=2)
    
    # permute for batch matrix product
    points_hom = points_hom.permute(0, 2, 1)
    print("rend_util points_hom.shape: ", points_hom.shape)
    print("rend_util pose.shape: ", pose.shape)
    
    print("rend_util points_hom[:, :, 0]: ", points_hom[:, :, 0])
    print("rend_util pose: ", pose)
    print("rend_util jt.linalg.inv(pose): ", jt.linalg.inv(pose))
    
    points_cam = jt.bmm(jt.linalg.inv(pose),points_hom)
    depth = points_cam[:, 2, :][:, :, None]
    print("rend_util depth.shape : ", depth.shape)
    print("rend_util points_cam.shape", points_cam.shape)
    print("rend_util points_cam[:, :, 0] = ", points_cam[:, :, 0])
    print("rend_util points_cam[:, 2, :][:, :, None].shape : ", points_cam[:, 2, :][:, :, None].shape)
    
    print("rend_util depth[:, 0, :] : ",depth[:, 0, :] )
    return depth

def quat_to_rot(q):
    batch_size, _ = q.shape
    q = jt.misc.normalize(q, dim=1)
    R = jt.ones((batch_size, 3,3))
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

# jt.Var([[[ 0.30679905,  0.5421411,  -0.78227705,  2.2613852 ]
#          [-0.8241403  , 0.5624584 ,  0.06658301, -0.06799692]
#          [ 0.47609568  ,0.6242784  , 0.61936206, -1.6888463 ]
#          [ 0.          ,0. ,         0. ,         1.        ]]])