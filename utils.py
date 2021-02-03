import random
import numpy as np
from collections import namedtuple
import copy


def exponential_moving_average(x, beta=0.9):
    average = 0
    ema_x = x.copy()
    for i, o in enumerate(ema_x):
        average = average * beta + (1 - beta) * o
        ema_x[i] = average / (1 - beta**(i+1))
    return ema_x


def simple_moving_average(x, window=100):
    x_avg, N = [], len(x)
    for i in range(N):
        n = max(0, i-window+1)
        x_avg.append(sum(x[n:i+1]) / len(x[n:i+1]))
    return np.array(x_avg)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class FrameHistory(object):

    def __init__(self, history_length):
        self.history_length = history_length
        self.history = []

    def push(self, frame):
        """Saves a frame."""
        self.history.append(frame)
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def clone(self):
        new = FrameHistory(self.history_length)
        new.history = self.history.copy()
        return new

    def __len__(self):
        return len(self.history)