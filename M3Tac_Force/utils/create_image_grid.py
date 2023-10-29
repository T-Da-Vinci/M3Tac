'''Functions for reading and saving EXR images using OpenEXR.
'''

import sys
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn as nn
from utils.api_utils import depth2rgb, label2color
import imageio
import os

sys.path.append('../..')


def seg_mask_to_rgb(seg_mask, num_classes):
    l2c = label2color(num_classes + 1)
    seg_mask_color = np.zeros((seg_mask.shape[0], 3, seg_mask.shape[2], seg_mask.shape[3]))
    for i in range(seg_mask.shape[0]):
        color = l2c.single_img_color(seg_mask[i])  # .squeeze(2).transpose(2,0,1).unsqueeze(0)
        color = np.squeeze(color, axis=2)
        color = color.transpose((2, 0, 1))
        color = color[np.newaxis, :, :, :]
        seg_mask_color[i] = color
    seg_mask_color = torch.from_numpy(seg_mask_color)
    return seg_mask_color


def xyz_to_rgb(xyz_map):
    xyz_rgb = torch.ones_like(xyz_map)
    for i in range(xyz_rgb.shape[0]):
        xyz_rgb[i] = torch.div((xyz_map[i] - xyz_map[i].min()),
                               (xyz_map[i].max() - xyz_map[i].min()).item())
    return xyz_rgb


def normal_to_rgb(normals_to_convert):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords,
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).

    Args:
        normals_to_convert (numpy.ndarray): Surface normals, dtype float32, range [-1, 1]
    '''
    camera_normal_rgb = (normals_to_convert + 1) / 2
    return camera_normal_rgb


def create_grid_image(rgb=None, inf=None, mask=None, output=None, labels=None, max_num_images_to_save=3):
    force_x, force_y, force_z = output
    label_x, label_y, label_z = labels
    res_x, res_y, res_z = torch.abs(label_x - force_x), torch.abs(label_y - force_y), torch.abs(label_z - force_z)

    rgb_tensor = rgb[:max_num_images_to_save].detach().cpu()
    inf_tensor = inf[:max_num_images_to_save].detach().cpu()
    mask_tensor = mask[:max_num_images_to_save].detach().cpu()
    force_x_tensor = force_x[:max_num_images_to_save].detach().cpu()
    force_y_tensor = force_y[:max_num_images_to_save].detach().cpu()
    force_z_tensor = force_z[:max_num_images_to_save].detach().cpu()
    res_x_tensor = res_x[:max_num_images_to_save].detach().cpu()
    res_y_tensor = res_y[:max_num_images_to_save].detach().cpu()
    res_z_tensor = res_z[:max_num_images_to_save].detach().cpu()

    mask_tensor = mask_tensor.repeat(1, 3, 1, 1)
    force_x_tensor = force_x_tensor.repeat(1, 3, 1, 1)
    force_y_tensor = force_y_tensor.repeat(1, 3, 1, 1)
    force_z_tensor = force_z_tensor.repeat(1, 3, 1, 1)
    res_x_tensor = res_x_tensor.repeat(1, 3, 1, 1)
    res_y_tensor = res_y_tensor.repeat(1, 3, 1, 1)
    res_z_tensor = res_z_tensor.repeat(1, 3, 1, 1)

    images = torch.cat((rgb_tensor, inf_tensor, mask_tensor, force_x_tensor, force_y_tensor, force_z_tensor, res_x_tensor, res_y_tensor, res_z_tensor), dim=3)
    grid_image = make_grid(images, 1, normalize=False, scale_each=False)

    return grid_image
