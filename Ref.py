import math
import numpy as np

class Ref_Wave(object):
    def __init__(self, T):
        self.T = T

    def fun(self, k, amplitude):
        if k % 30 == 0:
            #amplitude = amplitude - 1* 180/np.pi
            amplitude = np.random.uniform(-30 * np.pi / 180, 30 * np.pi / 180)
        #print('amplitude',amplitude)
        sin_ref = np.sin((2 * np.pi / self.T) * 0.1 * k)
        wave_ref = amplitude * math.copysign(1, sin_ref)

        return amplitude, wave_ref

class Ref_Sin(object):
    def __init__(self, T):
        self.T = T

    def fun(self, k, amplitude):

        sin_ref = amplitude * np.sin((2 * np.pi / self.T) * 0.1 * k)

        return sin_ref

