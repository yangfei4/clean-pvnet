from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'LinemodTrain': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/train.json'.format(cfg.cls_type),
            'split': 'train'
        },
        'LinemodTest': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/test.json'.format(cfg.cls_type),
            'split': 'test'
        },
        'FitTrain': {
            'id': 'fit',
            'data_root': 'data/{}_train'.format(cfg.cls_type),
            'ann_file': 'data/{}_train/train.json'.format(cfg.cls_type),
            'split': 'train'
        },
        'FitTest': {
            'id': 'fit',
            'data_root': 'data/{}_test'.format(cfg.cls_type),
            'ann_file': 'data/{}_test/train.json'.format(cfg.cls_type),
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

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()