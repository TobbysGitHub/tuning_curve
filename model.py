import math
import torch
import torch.nn as nn


class EncodeModule(nn.Module):
    def __init__(self, dots, units, ratio_min, ratio_max):
        super().__init__()
        self.dots = dots
        self.units = units
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.ratio_prior = _estimate_prior(ratio_min, ratio_max)
        self.weight = nn.Parameter(torch.empty(dots, units))

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
        alpha = si @ self.weight + 1  # (batch_size * units)
        alpha = self.reciprocal_relu(alpha)
        pr = pr_last * alpha.detach()  # (batch_size * units)
        output = [alpha, pr]
        return output

    @staticmethod
    def reciprocal_relu(x: torch.Tensor):
        x_ = torch.masked_fill(x, x == 2, float('inf'))
        return (x < 1) * (1 / (2 - x_)) + (x >= 1) * x


class LateralModule(nn.Module):
    def __init__(self, units, ratio_min, ratio_max):
        super().__init__()
        self.units = units
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.weight = nn.Parameter(torch.zeros(units, units))  # (units, units)
        self.bias = nn.Parameter(torch.ones(units) * _estimate_prior(ratio_min, ratio_max))
        self.register_buffer('eye_mask', torch.ones(units, units) - torch.eye(units, units))  # (units, units)

    def forward(self, pr: torch.Tensor):
        pulse = torch.bernoulli(pr.clamp(0, 1))  # (batch_size, units)
        output = pulse @ (self.weight * self.eye_mask) + self.bias
        output.data.clamp_(self.ratio_min, self.ratio_max)  # (batch_size, units)
        return output


class MartingaleModule(nn.Module):
    def __init__(self, units, ratio_min, ratio_max):
        super().__init__()
        self.pr_prior = _estimate_prior(ratio_min, ratio_max)
        self.pr_mean = nn.Parameter(torch.ones(units) * self.pr_prior)

    def forward(self):
        return self.pr_mean


class SquashModule(nn.Module):
    def __init__(self, ratio_min, ratio_max, squash_factor=2):
        super().__init__()
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.ratio_prior = _estimate_prior(ratio_min, ratio_max)
        self.squash_factor = squash_factor

    def forward(self, x):
        return self._squash(x, self.ratio_min, self.ratio_max, self.ratio_prior, self.squash_factor)

    @staticmethod
    def _squash(x, value_min: float, value_max: float, value_inflection: float, factor=0.9):
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

        return (x >= value_inflection) * rst_op(half_squash, value_max - value_inflection, value_inflection)(x) + \
               (x < value_inflection) * rst_op(half_squash, value_inflection - value_min, value_inflection,
                                               reflect=True)(x)


def _estimate_prior(ratio_min, ratio_max):
    # the estimate of prior probability
    return 1 / (1 + math.pow(
        (math.pow(ratio_min, ratio_min) * math.pow(1 - ratio_min, 1 - ratio_min)) /
        (math.pow(ratio_max, ratio_max) * math.pow(1 - ratio_max, 1 - ratio_max)),
        1 / (ratio_max - ratio_min)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    z = torch.linspace(0, 1, 100)
    y = SquashModule._squash(z, 0.0, 1, 0.5, 10)
    plt.plot(z.numpy(), y.numpy())
    plt.show()
    pass
