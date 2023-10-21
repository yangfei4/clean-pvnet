import os
import argparse
import time
from pathlib import Path

import cv2
import detectron2
import numpy as np
import torch

from utils.mrcnn import Detector
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_path", help="Path to coco annotations")
    parser.add_argument("--img_path", help="Path to corresponding imgs")
    parser.add_argument("input_path", help="Path to images to be used for inference")
    parser.add_argument("ckpt", help="Path to trained model")
    parser.add_argument("config", help="Path to config")
    parser.add_argument("dataset_name", help="Name of dataset")
    return parser.parse_args()

if __name__ == '__main__':
    args = setup_args()

    # Load Dataset
    ############################################
    ann_path = args.coco_path
    if args.img_path is None: 
        img_path = str(Path(ann_path).parent)
    else:
        img_path = args.img_path

    model_name = Path(args.ckpt).name.split(".pth")[0]
    coco_name = Path(ann_path).name.split(".json")[0]

    ds_name = args.dataset_name
    output_dir = f"eval_results/{ds_name}" 
    log_path = f"{output_dir}/log.txt"

    setup_logger(output=log_path)
    register_coco_instances(ds_name, {}, ann_path, img_path)
    _meta_data = MetadataCatalog.get(ds_name)


    # Inference Config
    ############################################
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    print(cfg)

    cfg.MODEL.WEIGHTS = str(args.ckpt)
    cfg.OUTPUT_DIR = output_dir

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"


    detector = DefaultPredictor(cfg)
    # detector = Detector(ann_path, img_path, ckpt_path=args.ckpt,input_path,)

    part_name = args.ckpt.split('/')[0].split('.pt')[0]
    pred_output_path = Path(args.input_path) / "inference_pred" / args.dataset_name / part_name 
    pred_output_path.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(Path(args.input_path).glob('*.png')):
        out_path = pred_output_path / img_path.name
        wdw_name = f"{img_path.name} Predictions"
        img_path = str(img_path)
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]

        prediction = detector(img)
        v = Visualizer(img, metadata=_meta_data)
        output = v.draw_instance_predictions(prediction["instances"].to("cpu"))
        output = output.get_image()[:, :, ::-1]
        cv2.imwrite(str(out_path), output)

        # out = detector.predict(img_path, visualize=False, no_mask=True)
        # out = detector(img_path)
        # cv2.imwrite(str(out_path), out)
        # print(f"Writing image to {out_path}")

        # cv2.destroyWindow(wdw_name)
