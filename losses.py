import torch
import torch.nn as nn


class MutualInfoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pr_last, pr, q, alpha):
        return -(pr_last * torch.log(pr / q) * alpha).mean()


class L2Regularization(nn.Module):
    def __init__(self, weight, decay=1e-5):
        super().__init__()
        self.weight = weight
        self.decay = decay

    def forward(self):
        return self.decay * (torch.pow(self.weight, 2)).sum()
        # return self.decay * (torch.pow(torch.clamp(torch.abs(self.weight) - 0.1, min=0), 2)).sum()
