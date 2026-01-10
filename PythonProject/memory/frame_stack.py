from collections import deque
import numpy as np

class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        for _ in range(self.k):
            self.frames.append(frame)
        return self._get()

    def step(self, frame):
        self.frames.append(frame)
        return self._get()

    def _get(self):
        return np.stack(self.frames, axis=0)
