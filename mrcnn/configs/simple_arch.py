import json

from omegaconf import OmegaConf
from detectron2 import model_zoo
from detectron2.config import get_cfg, LazyConfig
from detectron2.model_zoo import get_config



model_ref = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_ref))

# NOTE: epoch  = MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = 20000
cfg.SOLVER.BASE_LR = 1e-3

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_ref)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
 

print(cfg.dump())
with open("./configs/simple_arch.yaml", "w") as f:
      f.write(cfg.dump())   # save config to file

