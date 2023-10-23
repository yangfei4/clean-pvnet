import os
import argparse
import time
from pathlib import Path

import cv2
import detectron2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
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


class MaskRCNNWrapper(object):

    def __init__(self, 
                 cfg_path,
                 ckpt_path,
                 coco_path,
                 img_path,
                 ds_name,
                ):

        self.__dict__.update(locals())
        self._load_dataset()
        self._load_model()

    def _load_dataset(self):
        # Load Dataset
        ############################################
        if self.img_path is None: 
            img_path = str(Path(self.coco_path).parent)

        self.output_dir = f"eval_results/{self.ds_name}" 

        register_coco_instances(self.ds_name, {}, self.coco_path, self.img_path)
        self._meta_data = MetadataCatalog.get(self.ds_name)


    def _load_model(self):
        # Inference Config
        ############################################
        cfg = get_cfg()
        cfg.merge_from_file(self.cfg_path)

        cfg.MODEL.WEIGHTS = str(self.ckpt_path)
        cfg.OUTPUT_DIR = self.output_dir

        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"

        self.model = DefaultPredictor(cfg)

    def __call__(self, img):

        # TODO: crop image around tagboard center:

        # Use the model to run inference on the img
        prediction = self.model(img)
        v = Visualizer(img, metadata=self._meta_data)
        return prediction, v
        # TODO: crop rois around detected bbox centers

        # TODO: store uv info, rois by class pred


if __name__ == '__main__':
    args = setup_args()

    mrcnn = MaskRCNNWrapper(args.config, args.ckpt, args.coco_path, args.img_path, args.dataset_name)

    part_name = args.ckpt.split('/')[0].split('.pt')[0]
    pred_output_path = Path(args.input_path) / "inference_pred" / args.dataset_name / part_name 
    pred_output_path.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(Path(args.input_path).glob('*.png')):
        out_path = pred_output_path / img_path.name
        wdw_name = f"{img_path.name} Predictions"
        img_path = str(img_path)
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]

        prediction, v = mrcnn(img)
        output = v.draw_instance_predictions(prediction["instances"].to("cpu"))
        output = output.get_image()[:, :, ::-1]
        cv2.imwrite(str(out_path), output)
