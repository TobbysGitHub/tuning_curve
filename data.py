import math

import numpy as np
import torch


class DataLoader:
    def __init__(self, dots, ratio_min, ratio_max, batch_size, device, brown_speed=0.0):
        self.dots = dots
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.batch_size = batch_size
        self.brown_speed = brown_speed
        self.device = device
        self.si_pos = torch.tensor(np.linspace(start=0.0, stop=1.0, num=self.batch_size, endpoint=False),
                                   device=device)  # (batch_size,)
        self.dots_pos = torch.tensor(np.linspace(start=0.0, stop=1.0, num=self.dots, endpoint=False),
                                     device=device)  # (dots,)

    def _brown_motion(self):
        if self.brown_speed == 0.0:
            return
        brown_motion = torch.randn(self.batch_size) * self.brown_speed
        self.si_pos += brown_motion
        self.si_pos = self.si_pos.fmod(1)
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self._brown_motion()
        dots_phase = 2 * math.pi * (self.dots_pos - self.si_pos.view(-1, 1))  # (batch_size, dots)
        dots_ratio = (self.ratio_max * torch.cos(dots_phase)).clamp(min=self.ratio_min)
        dots_spike = torch.bernoulli(dots_ratio)
        return dots_spike


if __name__ == '__main__':
    torch.manual_seed(1)
    dots = 16
    ratio_min_1 = 1 / dots
    ratio_max_1 = 5 / dots
    dataloader = DataLoader(dots, ratio_min_1, ratio_max_1, batch_size=1, device=None)
    spikes = torch.zeros(dots)
    for _, data in zip(range(500), dataloader):
        spikes = spikes + data.cpu().numpy()[0]

    import matplotlib.pyplot as plt

    plt.plot(spikes)
    plt.show()
