import math
from random import random

import numpy as np
import torch
from tensorboard.compat.proto.summary_pb2 import Summary, HistogramProto
from torch.utils.tensorboard import SummaryWriter


def is_posting(step):
    assert step >= 0
    if step < 10:
        return True
    return random() < 1 / math.sqrt(step)


def add_plot(writer: SummaryWriter, tag, values, global_step):
    torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
    bins = [float(i) for i in range(len(values) + 1)]
    values = values.detach().cpu().numpy()
    counts = np.concatenate([[0], values])
    writer._get_file_writer().add_summary(
        _histogram(tag, counts, bins), global_step, None)
    pass


def _histogram(name, counts, bins):
    hist = _make_histogram(counts, bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def _make_histogram(counts, bins):
    return HistogramProto(min=0.5,
                          max=len(bins) - 0.5,
                          num=counts.sum(),
                          sum=(counts * bins).sum(),
                          sum_squares=(counts * bins * bins).sum(),
                          bucket_limit=bins,
                          bucket=counts.tolist())


def add_diff_histogram(writer: SummaryWriter, tag, values, global_step):
    """
    :param writer:
    :param tag:
    :param values: (dots, units)
    :param global_step:
    """
    dots = values.size()[0]
    values = values.detach().cpu().numpy()
    center_point = np.argmax(values, axis=0)  # (units,)
    center_point.sort()
    diff = np.diff(center_point, append=center_point[0] + dots)
    writer.add_histogram(tag=tag, values=diff, global_step=global_step)
