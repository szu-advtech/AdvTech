# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn.functional as F
import numpy as np

from fcn.test_common import _vis_minibatch_segmentation_final
from utils.mean_shift import mean_shift_smart_init
import utils.mask as util_

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def clustering_features(features, num_seeds=100):
    metric = 'euclidean' # NOTE
    height = features.shape[2]
    width = features.shape[3]
    out_label = torch.zeros((features.shape[0], height, width))

    # mean shift clustering
    kappa = 20
    selected_pixels = []
    for j in range(features.shape[0]):
        X = features[j].view(features.shape[1], -1)
        X = torch.transpose(X, 0, 1)
        cluster_labels, selected_indices = mean_shift_smart_init(X, kappa=kappa, num_seeds=num_seeds, max_iters=10, metric=metric)
        out_label[j] = cluster_labels.view(height, width)
        selected_pixels.append(selected_indices)
    return out_label, selected_pixels


def crop_rois(rgb, initial_masks, depth):

    device = torch.device('cuda:0')

    N, H, W = initial_masks.shape
    crop_size = 224 # NOTE
    padding_percentage = 0.25

    mask_ids = torch.unique(initial_masks[0])
    if mask_ids[0] == 0:
        mask_ids = mask_ids[1:]
    num = mask_ids.shape[0]
    rgb_crops = torch.zeros((num, 3, crop_size, crop_size), device=device)
    rois = torch.zeros((num, 4), device=device)
    mask_crops = torch.zeros((num, crop_size, crop_size), device=device)
    if depth is not None:
        depth_crops = torch.zeros((num, 3, crop_size, crop_size), device=device)
    else:
        depth_crops = None

    for index, mask_id in enumerate(mask_ids):
        mask = (initial_masks[0] == mask_id).float() # Shape: [H x W]
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(mask)
        x_padding = int(torch.round((x_max - x_min).float() * padding_percentage).item())
        y_padding = int(torch.round((y_max - y_min).float() * padding_percentage).item())

        # pad and be careful of boundaries
        x_min = max(x_min - x_padding, 0)
        x_max = min(x_max + x_padding, W-1)
        y_min = max(y_min - y_padding, 0)
        y_max = min(y_max + y_padding, H-1)
        rois[index, 0] = x_min
        rois[index, 1] = y_min
        rois[index, 2] = x_max
        rois[index, 3] = y_max

        # crop
        rgb_crop = rgb[0, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1] # [crop_H x crop_W]
        if depth is not None:
            depth_crop = depth[0, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]

        # resize
        new_size = (crop_size, crop_size)
        rgb_crop = F.upsample_bilinear(rgb_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
        rgb_crops[index] = rgb_crop
        mask_crop = F.upsample_nearest(mask_crop.unsqueeze(0).unsqueeze(0), new_size)[0,0] # Shape: [new_H, new_W]
        mask_crops[index] = mask_crop
        if depth is not None:
            depth_crop = F.upsample_bilinear(depth_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
            depth_crops[index] = depth_crop

    return rgb_crops, mask_crops, rois, depth_crops


# labels_crop is the clustering labels from the local patch
def match_label_crop(initial_masks, labels_crop, out_label_crop, rois, depth_crop):
    num = labels_crop.shape[0]
    for i in range(num):
        mask_ids = torch.unique(labels_crop[i])
        for index, mask_id in enumerate(mask_ids):
            mask = (labels_crop[i] == mask_id).float()
            overlap = mask * out_label_crop[i]
            percentage = torch.sum(overlap) / torch.sum(mask)
            if percentage < 0.5:
                labels_crop[i][labels_crop[i] == mask_id] = -1

    # sort the local labels
    sorted_ids = []
    for i in range(num):
        if depth_crop is not None:
            if torch.sum(labels_crop[i] > -1) > 0:
                roi_depth = depth_crop[i, 2][labels_crop[i] > -1]
            else:
                roi_depth = depth_crop[i, 2]
            avg_depth = torch.mean(roi_depth[roi_depth > 0])
            sorted_ids.append((i, avg_depth))
        else:
            x_min = rois[i, 0]
            y_min = rois[i, 1]
            x_max = rois[i, 2]
            y_max = rois[i, 3]
            orig_H = y_max - y_min + 1
            orig_W = x_max - x_min + 1
            roi_size = orig_H * orig_W
            sorted_ids.append((i, roi_size))

    sorted_ids = sorted(sorted_ids, key=lambda x : x[1], reverse=True)
    sorted_ids = [x[0] for x in sorted_ids]

    # combine the local labels
    refined_masks = torch.zeros_like(initial_masks).float()
    count = 0
    for index in sorted_ids:

        mask_ids = torch.unique(labels_crop[index])
        if mask_ids[0] == -1:
            mask_ids = mask_ids[1:]

        # mapping
        label_crop = torch.zeros_like(labels_crop[index])
        for mask_id in mask_ids:
            count += 1
            label_crop[labels_crop[index] == mask_id] = count

        # resize back to original size
        x_min = int(rois[index, 0].item())
        y_min = int(rois[index, 1].item())
        x_max = int(rois[index, 2].item())
        y_max = int(rois[index, 3].item())
        orig_H = int(y_max - y_min + 1)
        orig_W = int(x_max - x_min + 1)
        mask = label_crop.unsqueeze(0).unsqueeze(0).float()
        resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0, 0]

        # Set refined mask
        h_idx, w_idx = torch.nonzero(resized_mask).t()
        refined_masks[0, y_min:y_max+1, x_min:x_max+1][h_idx, w_idx] = resized_mask[h_idx, w_idx].cpu()

    return refined_masks, labels_crop


# filter labels on zero depths
def filter_labels_depth(labels, depth, threshold):
    labels_new = labels.clone()
    for i in range(labels.shape[0]):
        label = labels[i]
        mask_ids = torch.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]

        for index, mask_id in enumerate(mask_ids):
            mask = (label == mask_id).float()
            roi_depth = depth[i, 2][label == mask_id]
            depth_percentage = torch.sum(roi_depth > 0).float() / torch.sum(mask)
            if depth_percentage < threshold:
                labels_new[i][label == mask_id] = 0

    return labels_new


# filter labels inside boxes
def filter_labels(labels, bboxes):
    labels_new = labels.clone()
    height = labels.shape[1]
    width = labels.shape[2]
    for i in range(labels.shape[0]):
        label = labels[i]
        bbox = bboxes[i].numpy()

        bbox_mask = torch.zeros_like(label)
        for j in range(bbox.shape[0]):
            x1 = max(int(bbox[j, 0]), 0)
            y1 = max(int(bbox[j, 1]), 0)
            x2 = min(int(bbox[j, 2]), width-1)
            y2 = min(int(bbox[j, 3]), height-1)
            bbox_mask[y1:y2, x1:x2] = 1

        mask_ids = torch.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]

        for index, mask_id in enumerate(mask_ids):
            mask = (label == mask_id).float()
            percentage = torch.sum(mask * bbox_mask) / torch.sum(mask)
            if percentage > 0.8:
                labels_new[i][label == mask_id] = 0

    return labels_new


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = util_.build_matrix_of_indices(height, width)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img

def transform_by_matrix( points, mat, is_vec=False, is_point_image=False ):
    """
    Args:
        points: np.array [N, 3]
        mat: np.array [4, 4]
        is_vec: bool
        
    Returns:
        trans_points: np.array [N, 3]
    """
    rot = mat[:3, :3]

    w, h = mat.shape
    if w == 3 and h == 3:
        m = np.identity(4)
        m[:3,:3] = rot
        mat = m

    if is_point_image:
        trans_points = np.einsum('ij,abj->abi', rot, points )
    else:
        trans_points = np.einsum('ij,aj->ai', rot, points )

    if not is_vec:
        trans = mat[:3, 3]
        trans_points += trans
    
    return trans_points

def depth_to_pc(depth, cam_intrinsic, cam_to_world=None):
    """
    Args:
        depth: np.array [w, h, 3]
        cam_intrinsic: np.array [3, 3]
        cam_to_world: np.array [3, 3]
        with_noise: bool
    
    Returns:
        pointcloud: np.array [w, h, 3]
    """
    
    depth = depth.transpose(1, 0)
    w, h = depth.shape

    u0 = cam_intrinsic[0,2]
    v0 = cam_intrinsic[1,2]
    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    
    v, u = np.meshgrid( range(h), range(w) )
    z = depth
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy

    z = z.reshape(w, h, 1)
    x = x.reshape(w, h, 1)
    y = y.reshape(w, h, 1)

    depth = depth.transpose(1, 0)
    # 640 * 480 * 3
    ret = np.concatenate([x,y,z], axis=-1).astype('float32')

    # translate to world coordinate
    if cam_to_world is not None:
        ret = transform_by_matrix(ret, cam_to_world, is_point_image=True)

    ret = ret.transpose(1, 0, 2)

    return ret

def img_process(rgb, depth, camera_params):

    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    im_tensor = torch.from_numpy(rgb) / 255.0
    pixel_mean = torch.tensor(PIXEL_MEANS / 255.0).float()
    im_tensor -= pixel_mean

    image_blob = im_tensor.permute(2, 0, 1)
    image_blob = image_blob.unsqueeze(0)

    # bgr image
    # bgr ?
    # dep  xxx m
    if depth is not None:
        # height = depth.shape[0]
        # width = depth.shape[1]
        # fx = camera_params['fx']
        # fy = camera_params['fy']
        # px = camera_params['x_offset']
        # py = camera_params['y_offset']
        # xyz_img = compute_xyz(depth, fx, fy, px, py, height, width)
        xyz_img = depth_to_pc(depth, camera_params['cam'])

        depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
        depth_blob = depth_blob.unsqueeze(0)
    else:
        depth_blob = None

    return image_blob, depth_blob


def img_segment(rgb, dep, network, network_crop, device, visual=False):

    # construct input
    image = rgb.cuda(device=device)
    if dep is not None:
        depth = dep.cuda(device=device)
    else:
        depth = None
    label = None

    # run network
    features = network(image, label, depth).detach()
    out_label, selected_pixels = clustering_features(features, num_seeds=100)

    if depth is not None:
        # filter labels on zero depth
        out_label = filter_labels_depth(out_label, depth, 0.8)

    # zoom in refinement
    out_label_refined = None
    
    if network_crop is not None:
        rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
        if rgb_crop.shape[0] > 0:
            features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
            labels_crop, selected_pixels_crop = clustering_features(features_crop)
            out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)
            
    if visual:

        bbox = None
        _vis_minibatch_segmentation_final(image, depth, label, out_label, out_label_refined, features, 
            selected_pixels=selected_pixels, bbox=bbox)

    return out_label, out_label_refined
