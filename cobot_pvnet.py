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

def run_inference(cfg, image, K_cam):
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    data_loader = make_data_loader(cfg, is_train=False)
    
    batch_example = None # will be used to load kpts annotation
    for batch in data_loader:
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        batch_example = batch
        break

    network.eval()

    # image = Image.open(image_path).convert('RGB')
    # Preprocess the image

    transform = make_transforms(cfg, is_train=False)
    processed_image, _, _ = transform(image)
    processed_image = np.array(processed_image).astype(np.float32)

    # Convert the preprocessed image to a tensor and move it to GPU
    input_tensor = torch.from_numpy(processed_image).unsqueeze(0).cuda().float()

    with torch.no_grad():
        output = network(input_tensor)

    visualizer = make_visualizer(cfg)
    pose, visualization = visualizer.visualize_output(image, output, batch_example, K_cam)
    return pose, visualization


if __name__ == '__main__':
    # load instance segmentaino results from mask r-cnn
    """
        'class': an integer representing the class.
        'uv': a numpy array containing two float values.
        'score': a float representing the score.
        'image_128x128': a 2D numpy array representing an image.
    """
    load_path = 'maskrcnn.json'

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
        pose, visualization = globals()['run_'+args.type](cfg, instance['image_128x128'], K_cam)
        pose_list.append(pose)
        if(category == 2):
            z.append(pose[2,3])
    
    plt.figure()
    # plot z distance distrubution
    plt.hist(z, bins=20)
    plt.xlabel('z estimation (m)')
    plt.ylabel('occurrence count')
    plt.title('z estimation distribution for inset_mold')
    # plot a vertical distance line for a ground truth with text
    plt.axvline(x=1.473, color='r', linestyle='--')
    plt.show()