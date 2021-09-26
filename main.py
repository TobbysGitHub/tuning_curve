import numpy as np

from torch import optim
from torch.nn.functional import mse_loss

import utils
from data import DataLoader
from model import EncodeModule, LateralModule, MartingaleModule, SquashModule
from losses import *
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
    encode_module = EncodeModule(dots, units, ratio_min_2, ratio_max_2)
    lateral_module = LateralModule(units, ratio_min_2, ratio_max_2)
    martingale_module = MartingaleModule(units, ratio_min_2, ratio_max_2)
    squash_module = SquashModule(ratio_min_2, ratio_max_2, squash_factor)
    # loss
    weight_decay = 1e-2
    l2_alpha_fn = L2Regularization(encode_module.weight, weight_decay)
    # optim
    lr1 = 1
    lr2 = 1e-2
    optim_q = optim.SGD(lateral_module.parameters(), lr=lr1, momentum=0.5)
    optim_alpha = optim.SGD(encode_module.parameters(), lr=lr2)
    optim_mean = optim.SGD(martingale_module.parameters(), lr=lr1, momentum=0.5)
    optim_alpha_mean = optim.Adam(encode_module.parameters())
    # writer
    writer = SummaryWriter()

    pr_last = None
    for step, data in enumerate(dataloader):
        alpha, pr = encode_module(data, pr_last)
        q = lateral_module(pr)
        pr_mean = martingale_module()  # (units,)
        if pr_last is not None:
            loss_alpha = alpha_loss_fn(pr_last, pr, q.detach(), alpha) + l2_alpha_fn()
            optim_alpha.zero_grad()
            loss_alpha.backward(retain_graph=True)
            optim_alpha.step()
        loss_q = mse_loss(q, pr)
        optim_q.zero_grad()
        loss_q.backward()
        optim_q.step()

        loss_mean = mse_loss(pr_mean.unsqueeze(0).expand(batch_size, -1), pr)
        optim_mean.zero_grad()
        loss_mean.backward()
        optim_mean.step()
        loss_alpha_mean = alpha_mean_loss_fn(pr_mean.detach(), martingale_module.pr_prior, alpha)
        optim_alpha_mean.zero_grad()
        loss_alpha_mean.backward()
        optim_alpha_mean.step()

        pr_last = squash_module(pr)

        writer.add_scalar(tag='loss_q', scalar_value=loss_q, global_step=step)
        writer.add_scalar(tag='loss_mean', scalar_value=loss_mean, global_step=step)
        if utils.needs_posting(step):
            writer.add_histogram(tag='q', values=q, global_step=step)
            writer.add_histogram(tag='pr_mean', values=pr_mean, global_step=step)
            writer.add_histogram(tag='alpha', values=alpha, global_step=step)
            writer.add_histogram(tag='pr', values=pr, global_step=step)
            writer.add_histogram(tag='weight', values=encode_module.weight, global_step=step)
            utils.add_plot(writer, tag='dot_weight', values=encode_module.weight, global_step=step)
    pass
