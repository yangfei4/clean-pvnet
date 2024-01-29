
from lib.config import cfg as default_cfg


class DatasetCatalog(object):
    def __init__(self, cfg_new=None) -> None:
        self.cfg = cfg_new or default_cfg
        self.dataset_attrs = self._create_dataset_attrs()

    def _create_dataset_attrs(self):
        return {
        'FitTrain': {
            'id': 'fit',
            'data_root': 'data/{}_train'.format(self.cfg.cls_type),
            'ann_file': 'data/{}_train/train.json'.format(self.cfg.cls_type),
            'split': 'train'
        },
        'FitTest': {
            'id': 'fit',
            'data_root': 'data/{}_test'.format(self.cfg.cls_type),
            'ann_file': 'data/{}_test/train.json'.format(self.cfg.cls_type),
            'split': 'test'
        },
        'LinemodTest': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(self.cfg.cls_type),
            'ann_file': 'data/linemod/{}/test.json'.format(self.cfg.cls_type),
            'split': 'test'
        },
        'LinemodTrain': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(self.cfg.cls_type),
            'ann_file': 'data/linemod/{}/train.json'.format(self.cfg.cls_type),
            'split': 'train'
        }
    }

    def get(self, name):
        attrs = self.dataset_attrs[name]
        return attrs.copy()