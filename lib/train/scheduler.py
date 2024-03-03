from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR 
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class NoneScheduler(object):
    def __init__(self, optim):
        self.optim = optim

    def step(self):
        pass

    def get_last_lr(self):
        return [self.optim.param_groups[0]['lr']]


def make_lr_scheduler(cfg, optimizer):
    if cfg.train.nosched:
        scheduler = NoneScheduler(optimizer)
        return scheduler

    if cfg.train.cosine:
        T_0 = cfg.train.t_0 
        T_mult = cfg.train.t_mult
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, verbose=True)
        return scheduler

    if cfg.train.warmup:
        scheduler = WarmupMultiStepLR(optimizer, cfg.train.milestones, cfg.train.gamma, 1.0 / 3, 5, 'linear')
    else:
        scheduler = MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma)
    return scheduler


def set_lr_scheduler(cfg, scheduler):
    if cfg.train.warmup:
        scheduler.milestones = cfg.train.milestones
    else:
        scheduler.milestones = Counter(cfg.train.milestones)
    scheduler.gamma = cfg.train.gamma
