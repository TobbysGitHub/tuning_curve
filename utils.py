import numpy as np
import torch
from tensorboard.compat.proto.summary_pb2 import Summary, HistogramProto
from torch.utils.tensorboard import SummaryWriter


def needs_posting(step):
    assert step >= 0
    return step % int(np.sqrt(step) + 1) == 0


def add_plot(writer: SummaryWriter, tag, values, global_step):
    torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
    bins = [float(i) for i in range(len(values) + 1)]
    values = values.detach().numpy().reshape(-1)
    counts = np.concatenate([[0], values])
    writer._get_file_writer().add_summary(
        histogram(tag, counts, bins), global_step, None)
    pass


def histogram(name, counts, bins):
    hist = make_histogram(counts, bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram(counts, bins):
    return HistogramProto(min=0.5,
                          max=len(bins) - 0.5,
                          num=counts.sum(),
                          sum=(counts * bins).sum(),
                          sum_squares=(counts * bins * bins).sum(),
                          bucket_limit=bins,
                          bucket=counts.tolist())
