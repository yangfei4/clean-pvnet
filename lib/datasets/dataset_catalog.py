
from lib.config import cfg as default_cfg


class DatasetCatalog(object):
    def __init__(self, cfg_new=None) -> None:
        self.cfg = cfg_new or default_cfg
        self.dataset_attrs = self._create_dataset_attrs()

    def _create_dataset_attrs(self):
        return {
        'LinemodTrain': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(self.cfg.cls_type),
            'ann_file': 'data/linemod/{}/train.json'.format(self.cfg.cls_type),
            'split': 'train'
        },
        'LinemodTest': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(self.cfg.cls_type),
            'ann_file': 'data/linemod/{}/test.json'.format(self.cfg.cls_type),
            'split': 'test'
        },
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
        'InsertMoldTrain': {
            'id': 'custom',
            'data_root': 'data/insert_mold_train',
            'ann_file': 'data/insert_mold_train/train.json',
            'split': 'train'
        },
        'InsertMoldTest': {
            'id': 'custom',
            'data_root': 'data/insert_mold_test',
            'ann_file': 'data/insert_mold_test/train.json',
            # 'data_root': 'data/custom',
            # 'ann_file': 'data/custom/train.json',
            'split': 'test'
        },
        'MainshellTrain': {
            'id': 'custom',
            'data_root': 'data/mainshell_train',
            'ann_file': 'data/mainshell_train/train.json',
            'split': 'train'
        },
        'MainshellTest': {
            'id': 'custom',
            'data_root': 'data/mainshell_test',
            'ann_file': 'data/mainshell_test/train.json',
            # 'data_root': 'data/custom',
            # 'ann_file': 'data/custom/train.json',
            'split': 'test'
        },
        'TopshellTrain': {
            'id': 'custom',
            'data_root': 'data/topshell_train',
            'ann_file': 'data/topshell_train/train.json',
            'split': 'train'
        },
        'TopshellTest': {
            'id': 'custom',
            'data_root': 'data/topshell_test',
            'ann_file': 'data/topshell_test/train.json',
            # 'data_root': 'data/custom',
            # 'ann_file': 'data/custom/train.json',
            'split': 'test'
        },
	    'CustomTrain': {
            'id': 'custom',
            'data_root': 'data/custom',
            'ann_file': 'data/custom/train.json',
            'split': 'train'
        },
        'CustomTest': {
            'id': 'custom',
            'data_root': 'data/custom_test',
            'ann_file': 'data/custom_test/train.json',
            # 'data_root': 'data/custom',
            # 'ann_file': 'data/custom/train.json',
            'split': 'test'
        }
    }

    def get(self, name):
        attrs = self.dataset_attrs[name]
        return attrs.copy()