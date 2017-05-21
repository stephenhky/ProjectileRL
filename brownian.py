import numpy as np

class BrownianMotioner:
    def __init__(self, D=0.1, x0=0.0, y0=0.0):
        self.D = D
        self.x = x0
        self.y = y0
        self.nbsteps = 0

    def next(self):
        delta_x = np.random.normal(scale=self.D)
        delta_y = np.random.normal(scale=self.D)

        self.x += delta_x
        self.y += delta_y

        self.nbsteps += 1

        return self.x, self.y