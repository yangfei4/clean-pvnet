"""
Purpose: This is a non-ROS script that contains the pose estimation pipeline. Pose estimation is a two stage process: 1) Mask R-CNN + 2) PVNet
Author: Hameed Abdul (hameeda2@illinois.edu) and Yangfei Dai (yangfei4@illinois.edu)
"""
import os
import json
import argparse
import yaml
import torch
import cv2
import gin
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

from yacs.config import CfgNode as CN
from lib.config import args, cfgs
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from lib.datasets.transforms import make_transforms
from lib.datasets import make_data_loader
from lib.utils.pvnet import pvnet_pose_utils
from mrcnn.utils.maskrcnnWrapper import MaskRCNNWrapper


# Configs/Models in the order 0: mainshell, 1: topshell, 2: insert_mold 
def make_and_load_pvnet(cfg):
    net = make_network(cfg).cuda()
    load_network(net, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    return net

def call_pvnet(data, is_vis=True):
    cam_u = 2694.112343
    cam_v = 1669.169773

    W, H = int(5472), int(3648)
    crop_size = 128
    # shift the uv from original camera uv to cropped image center
    shifted_u = cam_u + (W//2 - data['uv'][0]) - (W//2 - crop_size//2)
    shifted_v = cam_v + (H//2 - data['uv'][1]) - (H//2 - crop_size//2)

    K_cam = np.array([[10704.062350, 0, shifted_u],
                [0, 10727.438047, shifted_v],
                [0, 0, 1]])

    cat_idx = data['class']
    cur_pvnet = pvnets[cat_idx]
    cur_cfg = cfgs[cat_idx]
    cur_roi = data['image_128x128']
    return run_inference(cur_pvnet, cur_cfg, cur_roi, K_cam, is_vis)


def predict_to_pose(pvnet_output, cfg, K_cam, input_img, is_vis: bool=False, is_pose_H: bool=True):
    kpt_3d = np.concatenate([cfg.fps_3d, [cfg.center_3d]], axis=0)
    kpt_2d = pvnet_output['kpt_2d'][0].detach().cpu().numpy()
    pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K_cam)

    def flip_yz_axes(pose_matrix):
        # Create a transformation matrix for flipping y and z axes by 180 degrees
        flip_matrix = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        # Multiply the original pose matrix by the flip matrix
        flipped_pose = np.dot(pose_matrix, flip_matrix)
        return flipped_pose
    
    pose_pred = flip_yz_axes(pose_pred)

    if is_vis:
        visualize_pose(input_img, cfg, pvnet_output, K_cam, pose_pred)

    if is_pose_H:
        # return pose as 4x4 matrix
        return np.c_[pose_pred.T, np.array([0, 0, 0, 1])].T
    # return pose as 3x4 matrix

    return pose_pred

def draw_axis(img, R, t, K, scale=0.006, dist=None):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array
    :R - Rotation matrix, 3x3
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :scale - factor to control the axis lengths
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    img = img.astype(np.float32)
    rotation_vec, _ = cv2.Rodrigues(R) #euler rotations
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)
    
    axis_points = axis_points.astype(int)
    corner = tuple(axis_points[3].ravel())
    img = cv2.line(img, corner, tuple(axis_points[0].ravel()), (255, 0, 0), 1)
    # img = cv2.putText(img, "X", tuple(axis_points[0].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 1)

    img = cv2.line(img, corner, tuple(axis_points[1].ravel()), (0, 255, 0), 1)
    # img = cv2.putText(img, "Y", tuple(axis_points[1].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 1)

    img = cv2.line(img, corner, tuple(axis_points[2].ravel()), (0, 0, 255), 1)

    img = img.astype(np.uint8)
    return img

def visualize_pose(input_img, cfg, pvnet_output, K_cam, pose_pred):
    corner_3d = cfg.corner_3d
    kpt_2d = pvnet_output['kpt_2d'][0].detach().cpu().numpy()
    segmentation = pvnet_output['seg'][0].detach().cpu().numpy()
    mask = pvnet_output['mask'][0].detach().cpu().numpy()
    corner_2d_pred = pvnet_pose_utils.project(corner_3d, K_cam, pose_pred)

    ###########################
    # overall result
    ###########################
    plt.figure(0)
    plt.subplot(221)
    plt.imshow(input_img)
    plt.axis('off')
    plt.title('Input Image')

    plt.subplot(222)
    plt.imshow(mask)
    plt.axis('off')
    plt.title('Predicted Mask')

    plt.subplot(223)
    plt.imshow(input_img)
    plt.scatter(kpt_2d[:8, 0], kpt_2d[:8, 1], color='red', s=10)
    plt.axis('off')
    plt.title('Key points detection')

    ax = plt.subplot(224)
    # ax.imshow(input_img)
    ax.imshow(draw_axis(input_img, pose_pred[:3, :3], pose_pred[:3, 3], K_cam))

    plt.axis('off')
    plt.title('Pose Prediction')
    # plt.savefig("/pvnet/data/evaluation/topshell.png")

    from scipy.spatial.transform import Rotation
    R = pose_pred[:3, :3]
    euler_angles = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    euler_angles_rounded = [int(angle) for angle in euler_angles]
    print("Euler angles for Estimated Pose in camera frame:", euler_angles_rounded)

    plt.show()

def run_inference(pvnet, cfg, image, K_cam, is_vis=False):
    pvnet.eval()

    transform = make_transforms(cfg, is_train=False)
    processed_image, _, _ = transform(image)
    processed_image = np.array(processed_image).astype(np.float32)

    # Convert the preprocessed image to a tensor and move it to GPU
    input_tensor = torch.from_numpy(processed_image).unsqueeze(0).cuda().float()

    with torch.no_grad():
        pvnet_output = pvnet(input_tensor)
    return predict_to_pose(pvnet_output, cfg, K_cam, image, is_vis)


if __name__ == '__main__':
    # load instance segmentaino results from mask r-cnn
    """
        'class': an integer representing the class.
        'uv': a numpy array containing two float values.
        'score': a float representing the score.
        'image_128x128': a 2D numpy array representing an image.
    """
    # Load all need models and configs
    gin.parse_config_file('./mrcnn/simple_output.gin')
    T_tagboard_in_cam = np.load("./T_tagboard_in_cam.npy")
    mrcnn = MaskRCNNWrapper()
    pvnets = tuple([make_and_load_pvnet(c) for c in cfgs])
    # Dataset can be downloaded on box: https://uofi.box.com/s/s81bn3nulxi18rlml1vnjwqmf4kyonal
    img_path = Path("./11_23_image_dataset")

    for img_path in img_path.glob("*.png"):
        print(img_path)
        img = cv2.imread(str(img_path))[:,:,::-1]
        data_for_pvnet, _, _ = mrcnn(img, is_vis=True)
        poses = [call_pvnet(data, is_vis=True) for data in data_for_pvnet]

        [d.update({'T_part_in_cam': T_part_in_cam,  'T_part_in_tag': np.linalg.inv(T_tagboard_in_cam) @ T_part_in_cam}) for d, T_part_in_cam in zip(data_for_pvnet, poses)]

        data = {str(i):d for i, d in enumerate(data_for_pvnet)}

        output_path = img_path.parent / (img_path.name[:-4] + ".npz")
        np.savez_compressed(output_path, **data)