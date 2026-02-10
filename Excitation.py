import math
import numpy as np

class Excitation_sin(object):
    def __init__(self, T):
        self.T = T

    def fun(self, k, amplitude):

        sin_ref = amplitude * np.sin((2 * np.pi / self.T) * 0.1 * k)

        return sin_ref

