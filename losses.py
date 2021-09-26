import torch
import torch.nn as nn


def alpha_loss_fn(pr_last, pr, q, alpha):
    return -(pr_last * torch.log(pr / q) * alpha).mean()


def squared_loss_fn(x1, x2):
    return torch.pow(x1 - x2, 2).mean()


def alpha_mean_loss_fn(pr_mean, pr_prior, alpha):
    return -((pr_prior - pr_mean) * alpha).mean()


class L2Regularization(nn.Module):
    def __init__(self, weight, decay=1e-5):
        super().__init__()
        self.weight = weight
        self.decay = decay

    def forward(self):
        return self.decay * (torch.pow(self.weight, 2)).sum()
        # return self.decay * (torch.pow(torch.clamp(torch.abs(self.weight) - 0.1, min=0), 2)).sum()
