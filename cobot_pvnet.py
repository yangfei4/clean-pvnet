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

    if is_vis:
        visualize_pose(input_img, cfg, pvnet_output, K_cam, pose_pred)

    if is_pose_H:
        # return pose as 4x4 matrix
        return np.c_[pose_pred.T, np.array([0, 0, 0, 1])].T
    # return pose as 3x4 matrix
    return pose_pred


def visualize_pose(input_img, pvnet_output, K_cam, pose_pred):
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
    ax.imshow(input_img)

    # Calculate center of bounding box
    center_x = np.mean(corner_2d_pred[:, 0])
    center_y = np.mean(corner_2d_pred[:, 1])
    shift_x = center_x - corner_2d_pred[6, 0]
    shift_y = center_y - corner_2d_pred[6, 1]
    # Plot X-axis
    ax.plot([center_x , corner_2d_pred[2, 0]+shift_x], [center_y, corner_2d_pred[2, 1]+shift_y], color='r', linewidth=1)
    # Plot Y-axis
    ax.plot([center_x, corner_2d_pred[4, 0]+shift_x], [center_y, corner_2d_pred[4, 1]+shift_y], color='g', linewidth=1)
    # Plot Z-axis
    ax.plot([center_x, corner_2d_pred[7, 0]+shift_x], [center_y, corner_2d_pred[7, 1]+shift_y], color='b', linewidth=1)
    # Add patches for corner_2d_gt and corner_2d_pred
    ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
    ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
    plt.axis('off')
    plt.title('Pose Prediction')
    # plt.savefig("/pvnet/data/evaluation/topshell.png")

    ###########################
    # vertex: currently makes no sense
    ###########################
    # plt.figure(1)
    # from torchvision.utils import make_grid
    # import torchvision
    # Grid = make_grid(output['vertex'].permute(1,0,2,3), nrow=9, padding=25)
    # vector_map = torchvision.transforms.ToPILImage()(Grid.cpu())
    # vector_map.show()
    # plt.imshow(vector_map)

    ###########################
    # segmentaion map, note:
    # mask = torch.argmax(output['seg'], 1)
    ###########################
    plt.figure(2)
    plt.subplot(121)
    plt.imshow(segmentation[0])
    plt.axis('off')
    plt.title('Segmentaion 1')

    plt.subplot(122)
    plt.imshow(segmentation[1])
    plt.axis('off')
    plt.title('Segmentaion 2')

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
    return predict_to_pose(pvnet_output, cfg, K_cam, image)


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
    # Dataset can be downloaded on box: https://uofi.box.com/s/s81bn3nulxi18rlml1vnjwqmf4kyonal
    img_path = Path("./11_23_image_dataset")

    for img_path in img_path.glob("*.png"):
        print(img_path)
        img = cv2.imread(str(img_path))
        data_for_pvnet, _, _ = mrcnn(img, is_vis=False)
        pvnets = tuple([make_and_load_pvnet(c) for c in cfgs])
        poses = [call_pvnet(data, is_vis=False) for data in data_for_pvnet]

        [d.update({'T_part_in_cam': T_part_in_cam,  'T_part_in_tag': np.linalg.inv(T_tagboard_in_cam) @ T_part_in_cam}) for d, T_part_in_cam in zip(data_for_pvnet, poses)]

        data = {str(i):d for i, d in enumerate(data_for_pvnet)}

        output_path = img_path.parent / (img_path.name[:-4] + ".npz")
        np.savez_compressed(output_path, **data)