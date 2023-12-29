import imp
import os
from lib.datasets.dataset_catalog import DatasetCatalog


def _evaluator_factory(cfg):
    task = cfg.task
    # make a instance of DatasetCatalog
    # dataset_log = DatasetCatalog(cfg_new=cfg)
    # args = dataset_log.get(name=dataset_name)
    # data_source = dataset_log.get(cfg.test.dataset)['id']

    dataset_log = DatasetCatalog()
    # args = DatasetCatalog.get(dataset_name)
    args = dataset_log.get(name = cfg.test.dataset)
    data_source = args['id']

    module = '.'.join(['lib.evaluators', data_source, task])
    path = os.path.join('lib/evaluators', data_source, task+'.py')
    evaluator = imp.load_source(module, path).Evaluator(cfg.result_dir)
    return evaluator


def make_evaluator(cfg):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg)
