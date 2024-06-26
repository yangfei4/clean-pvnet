from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
from tqdm import tqdm, trange
import os
import torch.multiprocessing
import time

import torch
import numpy as np
import random

import wandb


def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(cfg, network):
    time_start = time.time()

    if cfg.train.dataset[:4] != 'City':
        torch.multiprocessing.set_sharing_strategy('file_system')
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    if cfg.train.warmup and not cfg.train.cosine:
        set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg, is_train=False)

    trainer.set_fixed_batch(make_data_loader(cfg, is_train=False))
    
    wandb.watch(network, log="all",log_freq=1)

    for epoch in trange(begin_epoch, cfg.train.epoch):        
        recorder.epoch = epoch

        trainer.train(epoch, train_loader, optimizer, recorder, scheduler)

        # Multistep and warmup step schedulers need to be updated every epoch not batch iteration
        if cfg.train.warmup and not cfg.train.cosine:
            scheduler.step()

        # Preform cross validation of the model periodically
        if epoch % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder, scheduler, optimizer)
        
    # Save and upload best model to wandb for reference
    best_epoch = trainer.model_ckpt_data['epoch'] + 1
    path_to_ckpt = os.path.join(cfg.model_dir, '{}.pth'.format(best_epoch))
    torch.save(trainer.model_ckpt_data, path_to_ckpt)
    if cfg.train.save_to_wandb:
        wandb.save(path_to_ckpt)

    print(f"{'='*100}\nFinal Cross Validation Results\n{'='*100}")
    # load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    trainer.network.load_state_dict(torch.load(path_to_ckpt)['net'])
    trainer.val(best_epoch, val_loader, evaluator, recorder, scheduler, optimizer)

    print(f"[Timing] Training for {cfg.train.epoch - begin_epoch} epoch:")
    print(f"{time.time() - time_start} seconds \n{(time.time() - time_start)/3600} hours")
    print()
    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def main(cfg=cfg):
    with wandb.init(config=cfg):
        if cfg.train.wandb_sweep:
            cfg.train.t_0          = wandb.config.t_0
            cfg.train.t_mult       = wandb.config.t_mult
            cfg.train.lr           = wandb.config.lr

        network = make_network(cfg)
        if args.test:
            test(cfg, network)
        else:
            train(cfg, network)
    wandb.finish()

if cfg.train.wandb_sweep:
    if cfg.train.deterministic:
        set_seed() 

    sweep_config = {'method': 'bayes'} #'random'}

    # metric = {'name': '/Eval/eval_result.kpt_error','goal': 'minimize'}
    metric = {'name': '/Eval/eval_result.z_err_mm','goal': 'minimize'}

    parameters = {'lr' : {'max': 0.01, 'min': 1e-4},
                  't_0': {'max': 40, 'min': 10},
                  't_mult': {'max': 10, 'min': 1}}

    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters

    sweep_id = wandb.sweep(sweep_config, entity="cobot_illfit", project="pvnet_cobot")
    wandb.agent(sweep_id, main)


if __name__ == "__main__":
    if cfg.train.deterministic:
        set_seed() 
    main()
