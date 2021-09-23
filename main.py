import numpy as np
import torch

from torch import nn, optim

import utils
from data import DataLoader
from model import ForwardModule, AuxiliaryModule
from losses import MutualInfoLoss, L2Regularization
from torch.utils.tensorboard import SummaryWriter

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
    dataloader = DataLoader(dots, ratio_min_1, ratio_max_1, batch_size, brown_speed=0.0)
    # modules
    units = 1
    ratio_min_2 = 0.01
    ratio_max_2 = 0.1
    squash_factor = 0.5
    q_module = AuxiliaryModule(units, ratio_min_2, ratio_max_2)
    alpha_module = ForwardModule(dots, units, ratio_min_2, ratio_max_2, squash_factor)
    # loss
    weight_decay = 1e-2
    loss_q_fn = nn.MSELoss()
    loss_alpha_fn = MutualInfoLoss()
    l2_alpha_fn = L2Regularization(alpha_module.weight, weight_decay)
    # optim
    lr1 = 1
    lr2 = 1e-2
    optim_q = optim.SGD(q_module.parameters(), lr=lr1, momentum=0.5)
    optim_alpha = optim.SGD(alpha_module.parameters(), lr=lr2)
    # writer
    writer = SummaryWriter()

    pr_last = None
    for step, data in enumerate(dataloader):
        alpha, pr = alpha_module(data, pr_last)
        q = q_module(pr)
        if pr_last is not None:
            loss_alpha = loss_alpha_fn(pr_last, pr, q.detach(), alpha) + l2_alpha_fn()
            optim_alpha.zero_grad()
            loss_alpha.backward()
            # l2_alpha_fn()
            optim_alpha.step()
        loss_q = loss_q_fn(q, pr)
        optim_q.zero_grad()
        loss_q.backward()
        optim_q.step()

        # writer.add_scalar(tag='loss_alpha', scalar_value=loss_alpha, global_step=step)
        writer.add_scalar(tag='loss_q', scalar_value=loss_q, global_step=step)
        if step % 10000 == 0:
            writer.add_histogram(tag='q', values=q, global_step=step)
            writer.add_histogram(tag='alpha', values=alpha, global_step=step)
            writer.add_histogram(tag='pr', values=pr, global_step=step)
            writer.add_histogram(tag='weight', values=alpha_module.weight, global_step=step)
            utils.add_plot(writer, tag='dot_weight', values=alpha_module.weight, global_step=step)
        pr_last = pr
    pass
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
