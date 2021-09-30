import numpy as np
import torch

from torch import optim
from torch.nn.functional import mse_loss

import tb_utils
from data import DataLoader
from model import EncodeModule, LateralModule, MartingaleModule, SquashModule
# from losses import *
from grads import alpha_grad_fn
from torch.utils.tensorboard import SummaryWriter

# cuda
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set seeds
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # data_loader
    dots = 16
    ratio_min_1 = 1 / dots
    ratio_max_1 = 5 / dots
    batch_size = 256
    dataloader = DataLoader(dots, ratio_min_1, ratio_max_1, batch_size, device, brown_speed=0.0)
    # modules
    units = 2
    ratio_min_2 = 0.01
    ratio_max_2 = 0.1
    squash_factor = 0.5
    encode_module = EncodeModule(dots, units, ratio_min_2, ratio_max_2).to(device)
    lateral_module = LateralModule(units, ratio_min_2, ratio_max_2).to(device)
    martingale_module = MartingaleModule(units, ratio_min_2, ratio_max_2).to(device)
    squash_module = SquashModule(ratio_min_2, ratio_max_2, squash_factor).to(device)
    # optim
    lr1 = 1e-1
    lr2 = 2.5e-2
    weight_decay = 4e-2
    optimizer = optim.SGD(
        [{'params': (*lateral_module.parameters(), *martingale_module.parameters()), 'lr': lr1, 'momentum': 0.5},
         {'params': encode_module.parameters(), 'lr': lr2, 'momentum': 0.5}])
    # writer
    writer = SummaryWriter()

    ratio_prior = encode_module.ratio_prior
    pr_last = torch.ones(batch_size, units, device=device) * ratio_prior
    for step, data in enumerate(dataloader):
        alpha, pr = encode_module(data, pr_last)  # (batch_size * units)
        q = lateral_module(pr)  # (batch_size, units)
        pr_mean = martingale_module()  # (units,)

        grad_alpha = alpha_grad_fn(pr_last, pr, q) + 5 * (pr_mean - ratio_prior)
        grad_weight_decay = weight_decay * encode_module.weight
        grad_q = (q - pr)
        grad_pr_mean = (pr_mean - pr).sum(0)

        optimizer.zero_grad()
        alpha.backward(grad_alpha / batch_size)
        encode_module.weight.backward(grad_weight_decay)
        q.backward(grad_q / batch_size)
        pr_mean.backward(grad_pr_mean / batch_size)
        optimizer.step()

        pr_last = squash_module(pr)

        if tb_utils.is_posting(step):
            loss_q = mse_loss(q, pr)
            writer.add_scalar(tag='loss_q', scalar_value=loss_q, global_step=step)
            writer.add_histogram(tag='q', values=q, global_step=step)
            writer.add_histogram(tag='pr_mean', values=pr_mean, global_step=step)
            writer.add_histogram(tag='alpha', values=alpha, global_step=step)
            writer.add_histogram(tag='pr', values=pr, global_step=step)
            writer.add_histogram(tag='weight', values=encode_module.weight, global_step=step)
            writer.add_histogram(tag='l_weight', values=lateral_module.weight, global_step=step)
            tb_utils.add_diff_histogram(writer, tag='pos_diff', values=encode_module.weight, global_step=step)
            for i in range(units):
                tb_utils.add_plot(writer, tag='dot_weight/' + str(i), values=encode_module.weight[:, i],
                                  global_step=step)
    pass
