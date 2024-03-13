import io
import cv2
import time
import datetime
import torch
import tqdm
import shutil
from torch import nn
from torch.nn import DataParallel
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np

from lib.config import cfg, args
from lib.utils.pvnet import pvnet_pose_utils
from lib.utils import img_utils
from lib.utils.pvnet import pvnet_config
from lib.datasets.transforms import make_transforms
from lib.visualizers import make_visualizer
from lib.utils.net_utils import save_model


mean = pvnet_config.mean
std = pvnet_config.std

import wandb
import torch.onnx.symbolic_helper as sym_help

def diag_symbolic(g, input):
    # Retrieve the size of the input tensor
    input = input[0]
    input_shape = g.op("Shape", input)
    input_shape_size = g.op("Shape", input_shape, axis_i=torch.tensor([0], dtype=torch.int64))
    size = g.op("Squeeze", input_shape_size)

    # Create a zero tensor with the same device and dtype as the input
    # zeros = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))

                    
    # Create a zero tensor with the same device and dtype as the input
    zeros = g.op("ConstantOfShape", input_shape, value_t=torch.tensor(0, dtype=torch.float32))
    zero_tensor = g.op("Unsqueeze", zeros, axes_i=torch.tensor([0], dtype=torch.int64))
    zero_tensor = g.op("Unsqueeze", zero_tensor, axes_i=torch.tensor([0], dtype=torch.int64))
    zero_tensor = g.op("Expand", zero_tensor, size)

    # Create an identity matrix with size=size
    identity = g.op("EyeLike", zero_tensor)
    
    # Element-wise multiplication to get the diagonal elements
    output = g.op("Mul", input, identity)

    return output

# Register the symbolic function for the diag operation
OPSET_VER=12
def diag_symbolic_wrapper(g, *input):
     return diag_symbolic(g, input)

torch.onnx.register_custom_op_symbolic('::diag', diag_symbolic_wrapper, opset_version=OPSET_VER)

class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        network = DataParallel(network)
        self.network = network
        self.batch_to_vis = None
        self.max_kpt_projection_err = 1E3

    def set_fixed_batch(self, fixed_batch, num_samples=cfg.train.batch_size):
        self.fixed_batch = fixed_batch

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

    def train(self, epoch, data_loader, optimizer, recorder, scheduler):

        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            # recorder.step += 1

            # batch = self.to_cuda(batch)
            if cfg.train.vis_train_input and isinstance(self.batch_to_vis, type(None)):
                from torchvision.utils import make_grid
                input_grid =  make_grid(batch['inp']).cpu().permute(1, 2, 0).numpy()
                self.batch_to_vis = input_grid
                wandb.log({'Batch_Input/':  wandb.Image(input_grid),
                           "epoch": epoch})
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            # If using cosine annealing scheduler then steps need to happen every batch iteration
            if cfg.train.cosine:
                scheduler.step(epoch + (iteration - 1)  / max_iter)

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
            # Loss should be divid by all validation images which is = number of batches * batch size
            val_loss_stats[k] /= len(data_loader) * cfg.test.batch_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)
        
        # Save model checkpoint based
        cur_pix_err = result['kpt_error']
        if cur_pix_err < self.max_kpt_projection_err:
            self.model_ckpt_data = save_model(self.network.module, optimizer, scheduler, recorder, epoch, cfg.model_dir)
            print(f"Saving model... 2D Reprojection error decreased from {self.max_kpt_projection_err:3f} ---> {cur_pix_err}")
            self.max_kpt_projection_err = cur_pix_err

        if recorder and scheduler and optimizer:
            visualizer = make_visualizer(cfg)

            fixed_batch_visuals = {}
            for sample in self.fixed_batch:
                img = visualizer.get_image_and_tensor_for_batch(sample)
                with torch.no_grad():
                    one_output = self.network(sample)[0]
                img_id = int(sample['img_id'][0])
                fig = visualizer.make_figure_for_training(img, one_output, img_id)

                wandb_path = f"{img_id:0>4}"
                img_path = f'./output/{wandb_path.replace("/","_")}.png'
                # Use shutil.rmtree to remove the directory and all its contents
                path = Path(img_path)
                if path.exists():
                    shutil.rmtree(str(path.parent))

                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(img_path))
                plt.close(fig)
                fixed_batch_visuals.update({wandb_path: wandb.Image(str(img_path))})

            stats = {"epoch": epoch, 
                       "/Losses/total_loss": val_loss_stats['loss'], 
                       "/Losses/seg_loss": val_loss_stats['seg_loss'], 
                       "/Losses/vote_loss": val_loss_stats['vote_loss'], 
                       "/Eval/eval_result": result,
                       "/Lr/lr_scheduler": scheduler.get_last_lr()[0],
                       "/Lr/lr_optimizer": optimizer.param_groups[0]['lr']}

            fixed_batch_visuals.update(**stats)
            wandb.log(fixed_batch_visuals)
            recorder.record('val', epoch, val_loss_stats, image_stats)
    
