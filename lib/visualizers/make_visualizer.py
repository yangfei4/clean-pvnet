import os
import imp
from lib.datasets.dataset_catalog import DatasetCatalog


def make_visualizer(cfg):
    task = cfg.task
    dataset_log = DatasetCatalog()
    data_source = dataset_log.get(name=cfg.test.dataset)['id']
    module = '.'.join(['lib.visualizers', data_source, task])
    path = os.path.join('lib/visualizers', data_source, task+'.py')
    visualizer = imp.load_source(module, path).Visualizer(cfg)
    return visualizer
