import numpy as np
import os
import json
import argparse
import yaml
from yacs.config import CfgNode as CN
from lib.config import args, cfgs

def run_inference(cfg, image):
    import torch
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from PIL import Image
    from lib.visualizers import make_visualizer
    from lib.datasets.transforms import make_transforms
    from lib.datasets import make_data_loader

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
    visualizer.visualize_output(image, output, batch_example)


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

    # print(data)
    for instance in instances:
        category = instance['class']
        cfg = cfgs[category]
        # run_inference(cfg, instance['image_128x128'])
        globals()['run_'+args.type](cfg, instance['image_128x128'])