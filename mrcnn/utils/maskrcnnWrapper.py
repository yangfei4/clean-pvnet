from pathlib import Path
from typing import Union

import cv2
import detectron2
import gin
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog

from matplotlib import pyplot as plt
def plot_im(img: Union[str, Path, np.ndarray], output_name, figsize=(10,10)):
    
    if not isinstance(img, np.ndarray):
        assert Path(img).exists(), f"{img} is not a valid path"
        img = cv2.imread(str(img))
    
    try:
        if img.shape[-1] == 3:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pass
    except Exception as e:
        pass
    
    fig = plt.figure(figsize=figsize,tight_layout=True)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    fig.set_tight_layout(True)
    plt.imshow(img)
    plt.savefig(output_name)


def concat_images(img_list, n_rows=4):
    """
    Concatenate numpy images to 2xn grid.
    :param img_list: list of images as numpy arrays.
    :return: Concatenated image as numpy array.
    """
    num_images = len(img_list)
    if(num_images == 0):
        return
    elif(num_images == 1):
        return img_list[0]

    # Calculate the number of columns (n)
    n_cols = -(-num_images // n_rows)  # This is equivalent to ceil(num_images / 2)
    # Empty list to store the concatenated images row-wise
    rows = []
    for i in range(0, n_rows):
        row = []
        for j in range(0, n_cols):
            ind = i*n_cols + j

            if(ind<num_images):
                row.append(img_list[ind])
            else:
                black_img = np.zeros_like(img_list[0])
                row.append(black_img)
        concatenated_row = np.hstack(row)
        rows.append(concatenated_row)

    # Concatenate all the rows vertically
    concatenated_img = np.vstack(rows)
    
    return concatenated_img


def crop_roi(im: np.ndarray, cent, size) -> np.ndarray:
    if isinstance(size, list) or  isinstance(size, tuple):
        w, h = size
    elif isinstance(size, int):
        w = h = size

    x, y = cent
    x_off = slice(x - w // 2, x + w // 2)
    y_off = slice(y - h // 2, y + h // 2)
    
    if len(im.shape) == 2:
        return im[y_off, x_off]
    return im[y_off, x_off, :]



@gin.configurable
class MaskRCNNWrapper(object):

    def __init__(self, 
                 cfg_path=gin.REQUIRED,
                 ckpt_path=gin.REQUIRED,
                 coco_path=None,
                 img_path=None,
                 ds_name=None,
                 output_dir="./"
                ):

        self.__dict__.update(locals())
        if coco_path is not None:
            self._load_dataset()
        self._load_model()

        # TODO: Update this for the number of tagboards to consider
        self.TAGBOARD_CENT = (2000, 2200)
        self.model_input_wh = (2208, 1242)
        self.call_count = 0

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
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"

        self.model = DefaultPredictor(cfg)

    def process_pred_for_pvnet(self, predictions, full_res_img, mrcnn_pred_img, crop_dim=128):
        instances = predictions['instances']
        threshold = 0.5
        data_for_pvnet = []
        img_list = []
        pred_list = []

        for i in range(len(instances)):
            instance = instances[i]
            score = instance.scores.item()

            cls = instance.pred_classes.item()
            uv = instance.pred_boxes.get_centers().cpu().numpy()[0]

            # Be careful with order of u & v !!!
            if(score > threshold):
                # Crop roi around each part & pred mask wrt to cropped input image to mask rcnn
                u_cropped = int(uv[0])
                v_cropped = int(uv[1])
                # iamge_pred_128X128 = result[(v_out-dim//2):(v_out+dim//2), (u_out-dim//2):(u_out+dim//2)]
                cent_cropped = (u_cropped, v_cropped)
                roi_pred_128 = crop_roi(mrcnn_pred_img, cent_cropped, crop_dim)
                pred_list.append(roi_pred_128)
                
                # Crop roi around each part  wrt to full resolution input image (before crop)
                u_tag_cent, v_tag_cent = self.TAGBOARD_CENT
                width_tag, height_tag = self.model_input_wh
                u_5k = u_cropped + (u_tag_cent - width_tag  //2)
                v_5k = v_cropped + (v_tag_cent - height_tag //2)
                # iamge_128X128 = im[(v-dim//2):(v+dim//2), (u-dim//2):(u+dim//2)]
                cent_5k = (u_5k,v_5k)
                roi_5k_to_128 = crop_roi(full_res_img, cent_5k, crop_dim)
                img_list.append(roi_5k_to_128 )
                pvnet_input= {"class": cls, "uv": cent_5k, "score": score, "image_128x128": roi_5k_to_128 }
                data_for_pvnet.append(pvnet_input)
            else:
                print(f"instance {i} is in low confidence score")
        return data_for_pvnet, pred_list, img_list


    def __call__(self, full_res_img, is_vis=True):
        # TODO: crop image around tagboard center:
        img = crop_roi(full_res_img, self.TAGBOARD_CENT, self.model_input_wh)

        # Use the model to run inference on the img
        prediction = self.model(img)
        if hasattr(self, "_meta_data"):
            v = Visualizer(img, metadata=self._meta_data)
        else:
            v = Visualizer(img)
        output_img = v.draw_instance_predictions(prediction["instances"].to("cpu"))
        output_img = output_img.get_image()[:, :, ::-1]

        # TODO: crop rois around detected bbox centers
        # TODO: store uv info, rois by class pred
        data_for_pvnet, pred_list, img_list = self.process_pred_for_pvnet(prediction, full_res_img, output_img)
        if is_vis:
            img_grid = concat_images(img_list)
            plot_im(img_grid, f"roi_5k_to_128_{self.call_count}.png")

            pred_grid = concat_images(pred_list)
            plot_im(pred_grid, f"roi_preds_128_{self.call_count}.png")
        self.call_count += 1
        return data_for_pvnet, prediction, output_img

