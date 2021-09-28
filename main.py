import numpy as np

from torch import optim
from torch.nn.functional import mse_loss

import utils
from data import DataLoader
from model import EncodeModule, LateralModule, MartingaleModule, SquashModule
from losses import *
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
    # loss
    weight_decay = 1e-2
    l2_alpha_fn = L2Regularization(encode_module.weight, weight_decay)
    # optim
    lr1 = 1e-1
    lr2 = 1e-2
    optimizer = optim.SGD(
        [{'params': encode_module.parameters(), 'lr': lr2},
         {'params': (*lateral_module.parameters(), *martingale_module.parameters()), 'lr': lr1, 'momentum': 0.5}])

    # writer
    writer = SummaryWriter()

    pr_last = torch.ones(batch_size, units, device=device) * encode_module.ratio_prior
    for step, data in enumerate(dataloader):
        alpha, pr = encode_module(data, pr_last)
        q = lateral_module(pr)
        pr_mean = martingale_module()  # (units,)

        loss = 0
        loss_alpha = alpha_loss_fn(pr_last, pr, q.detach(), alpha) + l2_alpha_fn()
        loss = loss + loss_alpha
        loss_q = mse_loss(q, pr)
        loss = loss + loss_q
        loss_mean = mse_loss(pr_mean.unsqueeze(0).expand(batch_size, -1), pr)
        loss = loss + loss_mean
        loss_alpha_mean = alpha_mean_loss_fn(pr_mean.detach(), martingale_module.pr_prior, alpha)
        loss = loss + 5 * loss_alpha_mean
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pr_last = squash_module(pr)

        if utils.is_posting(step):
            writer.add_scalar(tag='loss_q', scalar_value=loss_q, global_step=step)
            writer.add_histogram(tag='q', values=q, global_step=step)
            writer.add_histogram(tag='pr_mean', values=pr_mean, global_step=step)
            writer.add_histogram(tag='alpha', values=alpha, global_step=step)
            writer.add_histogram(tag='pr', values=pr, global_step=step)
            writer.add_histogram(tag='weight', values=encode_module.weight, global_step=step)
            writer.add_histogram(tag='l_weight', values=lateral_module.weight, global_step=step)
            utils.add_diff_histogram(writer, tag='pos_diff', values=encode_module.weight, global_step=step)
            for i in range(units):
                utils.add_plot(writer, tag='dot_weight/' + str(i), values=encode_module.weight[:, i], global_step=step)
    pass
