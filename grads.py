import torch


def alpha_grad_fn(pr_last, pr, q):
    return -(pr_last * torch.log(pr / q))
