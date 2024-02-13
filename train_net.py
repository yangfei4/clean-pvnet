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
wandb.init(project="pvnet", name="32-no-aug", 
           config={"learning_rate": cfg.train.lr, "epochs": cfg.train.epoch, "batch_size": cfg.train.batch_size})

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using mutiple gpu

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
    # set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg, is_train=False)
    # train_loader = make_data_loader(cfg, is_train=True, max_iter=100)
    
    wandb.watch(
            network,
            log="all",
            log_freq=1
    )

    for epoch in range(begin_epoch, cfg.train.epoch):        
        recorder.epoch = epoch

        if epoch % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        # if epoch<=25 or epoch%cfg.eval_ep == 0:
            # trainer.val(epoch, val_loader, evaluator, recorder)
        
        # evaluate model every epoch
        trainer.val(epoch, val_loader, evaluator, recorder, scheduler, optimizer)

        if epoch % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)
        
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

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
    # set_seed() # to be removed
    main()
