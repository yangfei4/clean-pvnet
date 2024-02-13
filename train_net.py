from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch.multiprocessing
import time

import torch
import numpy as np
import random

import wandb
wandb.init(entity="cobot_illfit", project="pvnet_cobot", name=cfg.train.exp_name, 
           config={"learning_rate": cfg.train.lr, "epochs": cfg.train.epoch, "batch_size": cfg.train.batch_size})

class NoneScheduler(object):
    def __init__(self, optim):
        self.optim = optim

    def step(self):
        pass

    def get_last_lr(self):
        return [self.optim.param_groups[0]['lr']]


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
    if cfg.train.nosched:
        scheduler = NoneScheduler(optimizer)
    else:
        scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)
    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    # set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg, is_train=False)

    trainer.set_fixed_batch(make_data_loader(cfg, is_train=False))
    
    wandb.watch(
            network,
            log="all",
            log_freq=1
    )

    for epoch in range(begin_epoch, cfg.train.epoch):        
        recorder.epoch = epoch

        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        # Evaluate and save model periodically
        # TODO: Add proper cross validation. 
        if epoch % cfg.save_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder, scheduler, optimizer)
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)
        

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


def main():
    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    if cfg.train.deterministic:
        set_seed() 
    main()
