import json

from omegaconf import OmegaConf
from detectron2 import model_zoo
from detectron2.config import get_cfg, LazyConfig
from detectron2.model_zoo import get_config


model_ref = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_ref))

# NOTE: epoch  = MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
# Number of images per batch across all machines(GPUs) /
# Also Number of trianing images per step

epoch = 5
TOTAL_NUM_IMAGES = 10000

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = int(epoch * TOTAL_NUM_IMAGES / cfg.SOLVER.IMS_PER_BATCH)
cfg.SOLVER.BASE_LR = 1e-3
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.CHECKPOINT_PERIOD = 1000

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_ref)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
# number of regions per image used to train RPN
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 300
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256]]

cfg.INPUT.MIN_SIZE_TRAIN = (2208, 1242)
cfg.INPUT.MIN_SIZE_TEST = 1242
cfg.INPUT.MAX_SIZE_TRAIN = 2208
cfg.INPUT.MAX_SIZE_TEST = 2208

print(cfg.dump())
with open("./configs/res101_1009.yaml", "w") as f:
      f.write(cfg.dump())   # save config to file