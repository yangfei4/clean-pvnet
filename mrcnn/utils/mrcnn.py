import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask


class Detector:
    def __init__(self, 
                 ann_path: str='./coco_data/coco_annotations.json',
                 img_path: str='./coco_data/rgb_0000.png',
                 part_name: str='topshell',
                 ckpt_path: str='./insert_mold.pth',
                 input_path: str='./cobot/imgs',
                 output_path: str='./mrcnn_output'):
        self.cfg = get_cfg()
        
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95

        if torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = 'cuda'
        else:
            self.cfg.MODEL.DEVICE = 'cpu'
            
        
        self.load_mrcnn(ann_path, img_path, part_name, ckpt_path, input_path, output_path)
        
        self.predictor = DefaultPredictor(self.cfg)
        self._mask = None
        self._mask2 = None
        
    def predict(self, img_path: str, visualize: bool=True, no_mask: bool=False):
        start_time = time.time()
        img = cv2.imread(img_path)
        if not no_mask:
            masked_img = cv2.bitwise_and(self._mask, img)
        
        if visualize:
            cv2.namedWindow("Masked", cv2.WINDOW_NORMAL)
            cv2.imshow("Masked", masked_img)

            cv2.namedWindow("Mask 2", cv2.WINDOW_NORMAL)
            cv2.imshow("Mask 2", cv2.bitwise_and(self._mask2, img))
        
        
        prediction = self.predictor(img)
        stop_time = time.time()
        v = Visualizer(img[:, :, ::-1], metadata=self._meta_data)
        
        output = v.draw_instance_predictions(prediction["instances"].to("cpu"))

        output_masked = output.get_image()[:, :, ::-1]
        if no_mask:
            return output_masked
        
        outworkpace = np.greater(self._mask2, 0)
        output_masked[outworkpace] = img[outworkpace]
        
        
        fps = 1 / (stop_time - start_time)
        print(f"FPS: {fps}")

        if visualize:
            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.imshow("Test", output_masked)
            cv2.waitKey(0)
        return output_masked
        
    def is_occluded() -> bool:
        pass
        
    def load_mrcnn(self, ann_path: str, img_path: str, part_name: str, ckpt_path: str, input_path: str, output_path: str):
        # Load Dataset
        ds_name = f"{part_name}_ros"
        register_coco_instances(ds_name, {}, ann_path, img_path)
        self._meta_data = MetadataCatalog.get(ds_name)
        dataset = DatasetCatalog.get(ds_name)
        
        self.cfg.DATASETS.TRAIN = (ds_name)
        
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        
        Path(output_path).mkdir(exist_ok=True, parents=True)
        self.cfg.MODEL.WEIGHTS = str(Path.cwd() / ckpt_path)
        
    def create_mask(self, img_path, visualize: bool=True):
        img = cv2.imread(img_path)
        img_reg = img.copy()
        img = cv2.GaussianBlur(img, (7,7), cv2.BORDER_DEFAULT)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(hsv, (40, 100, 60), (90, 255,255))
        
        imask_green = mask_green>0
        green = np.zeros_like(img, np.uint8)
        green[imask_green] = img[imask_green]
        
        gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
        self._mask = green

        
        if visualize:
            cv2.namedWindow("Green", cv2.WINDOW_NORMAL)
            cv2.imshow("Green", green)

        (contours, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        area = []
        area.append(cv2.contourArea(contour) for contour in contours)
        cnt = contours[area.index(max(area))]
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(self._mask, [box], 0, (255, 255, 255), -1)
        
        self._mask2 = cv2.bitwise_not(self._mask)

