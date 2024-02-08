import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np

from lib.config import cfg, args
from lib.utils.pvnet import pvnet_pose_utils
from lib.utils import img_utils
from lib.utils.pvnet import pvnet_config
from lib.datasets.transforms import make_transforms
from lib.visualizers import make_visualizer

mean = pvnet_config.mean
std = pvnet_config.std

import wandb

class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        network = DataParallel(network)
        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):

        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            # recorder.step += 1

            # batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                
                # recorder.record('train')
        # Record the training statistics at the end of each epoch
        recorder.record('train')
        recorder.step += 1

    def val(self, epoch, data_loader, evaluator=None, recorder=None, scheduler=None, optimizer=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        batch_num = 0
        batch_exmaple = None
        for batch in tqdm.tqdm(data_loader):
            if batch_num==2: # choose the third image as example(no specifc reason)
                batch_exmaple = batch
            batch_num += 1
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network.module(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size*batch_num
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)
        
        if recorder and scheduler and optimizer:
            visualizer = make_visualizer(cfg)
            img = visualizer.get_image_and_tensor_for_batch(batch_exmaple)

            with torch.no_grad():
                one_output = self.network(batch_exmaple)[0]

            img_id = int(batch_exmaple['img_id'][0])
            fig = visualizer.make_figure_for_training(img, one_output, img_id)
            wandb.log({"epoch": epoch, 
                       "total_loss": val_loss_stats['loss'], 
                       "seg_loss": val_loss_stats['seg_loss'], 
                       "vote_loss": val_loss_stats['vote_loss'], 
                       "eval_result": result,
                       "visual": wandb.Image(fig),
                       "lr_scheduler": scheduler.get_last_lr()[0],
                       "lr_optimizer": optimizer.param_groups[0]['lr']})
            recorder.record('val', epoch, val_loss_stats, image_stats)
    
