import numpy as np
import torch


class DataLoader:
    def __init__(self, dots, ratio_min, ratio_max, batch_size, brown_speed=0.0):
        self.dots = dots
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.batch_size = batch_size
        self.brown_speed = brown_speed
        self.si_pos = np.linspace(start=0.0, stop=1.0, num=self.batch_size, endpoint=False)  # (batch_size,)
        self.dots_pos = np.linspace(start=0.0, stop=1.0, num=self.dots, endpoint=False)  # (dots,)

    def _brown_motion(self):
        if self.brown_speed == 0.0:
            return
        brown_motion = np.random.randn(self.batch_size) * self.brown_speed
        self.si_pos += brown_motion
        self.si_pos = np.mod(self.si_pos, 1)
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self._brown_motion()
        dots_phase = 2 * np.pi * (self.dots_pos - self.si_pos.reshape(-1, 1))  # (batch_size, dots)
        dots_ratio = np.maximum(self.ratio_min, self.ratio_max * np.cos(dots_phase))
        dots_spike = np.random.binomial(1, p=dots_ratio).astype(np.float32)
        return torch.tensor(dots_spike)


if __name__ == '__main__':
    np.random.seed(1)
    dots = 16
    ratio_min_1 = 1 / dots
    ratio_max_1 = 5 / dots
    batch_size = 1
    dataloader = DataLoader(dots, ratio_min_1, ratio_max_1, batch_size)
    spikes = np.zeros(dots)
    for _, data in zip(range(500), dataloader):
        spikes = spikes + data.numpy()[0]

    import matplotlib.pyplot as plt

    plt.plot(spikes)
    plt.show()
