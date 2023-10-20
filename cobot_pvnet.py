import os
import json
import argparse
import yaml
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from yacs.config import CfgNode as CN
from lib.config import args, cfgs
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from lib.datasets.transforms import make_transforms
from lib.datasets import make_data_loader
from lib.utils.pvnet import pvnet_pose_utils

def predict_to_pose(pvnet_output, K_cam, input_img, is_vis: bool=False):
    kpt_3d = np.concatenate([cfg.fps_3d, [cfg.center_3d]], axis=0)
    kpt_2d = pvnet_output['kpt_2d'][0].detach().cpu().numpy()
    pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K_cam)

    print("Camera Intrinsics:")
    print(K_cam)
    print("Predicted Pose wrt camera:")
    print(pose_pred)

    if is_vis:
        visualize_pose(input_img, pvnet_output, K_cam, pose_pred)
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

def run_inference(cfg, image, K_cam):
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)

    network.eval()

    transform = make_transforms(cfg, is_train=False)
    processed_image, _, _ = transform(image)
    processed_image = np.array(processed_image).astype(np.float32)

    # Convert the preprocessed image to a tensor and move it to GPU
    input_tensor = torch.from_numpy(processed_image).unsqueeze(0).cuda().float()

    with torch.no_grad():
        pvnet_output = network(input_tensor)
    return predict_to_pose(pvnet_output, K_cam, image, is_vis=True)


if __name__ == '__main__':
    # load instance segmentaino results from mask r-cnn
    """
        'class': an integer representing the class.
        'uv': a numpy array containing two float values.
        'score': a float representing the score.
        'image_128x128': a 2D numpy array representing an image.
    """
    load_path = 'maskrcnn_topshell.json'

    # Load the data from the specified path
    with open(load_path, 'r') as f:
        instances = json.load(f)

    # Convert lists back to numpy arrays after loading
    for item in instances:
        item['uv'] = np.array(item['uv'])
        item['image_128x128'] = np.array(item['image_128x128'])

    z = []
    pose_list = []
    for instance in instances:
        category = instance['class']
        cfg = cfgs[category]
        K_cam = np.array([[10704.062350, 0, item['uv'][0]],
                    [0, 10727.438047, item['uv'][1]],
                    [0, 0, 1]])
        pose = globals()['run_'+args.type](cfg, instance['image_128x128'], K_cam)
        pose_list.append(pose)
        if(category == 1): # 0: mainshell, 1: topshell, 2: insert_mold
            z.append(pose[2,3])
    
    plt.figure()
    # plot z distance distrubution
    plt.hist(z, bins=20)
    plt.xlabel('z estimation (m)')
    plt.ylabel('occurrence count')
    plt.title('z estimation distribution')
    # plot a vertical distance line for a ground truth with text
    plt.axvline(x=1.473, color='r', linestyle='--')
    plt.show()