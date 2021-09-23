import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModule(nn.Module):
    def __init__(self, dots, units, ratio_min, ratio_max, squash_factor=2):
        super().__init__()
        self.dots = dots
        self.units = units
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.squash_factor = squash_factor
        self.weight = nn.Parameter(torch.empty(dots, units))
        # self.bias = nn.Parameter(torch.ones(units))
        self.register_buffer('bias', torch.ones(units))

        self.reset_parameter()

    def reset_parameter(self):
        with torch.no_grad():
            return self.weight.normal_(0, 0.1)

    def forward(self, si: torch.Tensor, pr_last: torch.Tensor):
        """
        :param si: stimuli # (batch_size * dots)
        :param pr_last: last presentation # (batch_size * units)
        :return:
        """
        if pr_last is None:
            pr_last = torch.ones(si.shape[0], self.units) * _estimate_prior(self.ratio_min, self.ratio_max)
        # else:
        #     pr_last = _squash(pr_last, self.ratio_min, self.ratio_max, self.linearity)
        alpha = si @ self.weight + self.bias  # (batch_size * units)
        # alpha = F.softplus(alpha)
        pr = pr_last * alpha.detach()  # (batch_size * units)
        pr = _squash(pr, self.ratio_min, self.ratio_max, self.squash_factor)
        output = [alpha, pr]
        return output


class AuxiliaryModule(nn.Module):
    def __init__(self, units, ratio_min, ratio_max):
        super().__init__()
        self.units = units
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.bias = nn.Parameter(torch.ones(units) * _estimate_prior(ratio_min, ratio_max))

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        self.bias.data.clamp_(self.ratio_min, self.ratio_max)
        return self.bias.expand(batch_size, self.units)


def _squash(x, value_min: float, value_max: float, factor=0.9):
    def half_squash(x_):
        if factor < 1:
            return (1 - factor) * (1 - torch.relu(1 - x_)) + factor * torch.tanh(x_)
        else:
            # return torch.nan_to_num(torch.tanh((torch.sqrt(1 + 4 * factor * x_) - 1) / (2 * factor)))
            return torch.tanh(x / factor)

    def rst_op(func, scale, trans, reflect=False):
        def reflect_op(f):
            return lambda val: -f(-val)

        def scale_op(f):
            return lambda val: scale * f(val / scale)

        def trans_op(f):
            return lambda val: trans + f(val - trans)

        if reflect:
            return trans_op(scale_op(reflect_op(func)))
        else:
            return trans_op(scale_op(func))

    value_fixed = _estimate_prior(value_min, value_max)
    return (x >= value_fixed) * rst_op(half_squash, value_max - value_fixed, value_fixed)(x) + \
           (x < value_fixed) * rst_op(half_squash, value_fixed - value_min, value_fixed, reflect=True)(x)


def _estimate_prior(ratio_min, ratio_max):
    # the estimate of prior probability
    return 1 / (1 + math.pow(
        (math.pow(ratio_min, ratio_min) * math.pow(1 - ratio_min, 1 - ratio_min)) /
        (math.pow(ratio_max, ratio_max) * math.pow(1 - ratio_max, 1 - ratio_max)),
        1 / (ratio_max - ratio_min)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    z = torch.linspace(0, 1, 100)
    y = _squash(z, 0.0, 1, 10)
    plt.plot(z.numpy(), y.numpy())
    plt.show()
    pass
