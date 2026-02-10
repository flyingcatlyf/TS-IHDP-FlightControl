import math
import numpy as np


class second_order_filter(object):
    def __init__(self, step, omega, epsilon):
        self.step = step
        self.omega = omega
        self.epsilon = epsilon

    def solver(self, t, x0, u0):  # the input x0,u0 are required to be np.arrary. #function solver solves x1 and reward
        k1 = self.system(t, x0, u0)
        k2 = self.system(t + self.step / 2.0, x0 + self.step * np.transpose(k1) / 2.0, u0)
        k3 = self.system(t + self.step / 2.0, x0 + self.step * np.transpose(k2) / 2.0, u0)
        k4 = self.system(t + self.step / 2.0, x0 + self.step * np.transpose(k3), u0)

        x1 = x0 + self.step * np.transpose(k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        terminated = False
        return np.array(x1, dtype=np.float32), terminated, False, {}

    def system(self, t, x0, u0):
        #print('x0',x0)

        v0 = x0[0]
        v1 = x0[1]

        dv0 = v1
        dv1 = -2 * self.epsilon * self.omega * v1 + self.omega * self.omega * (u0[0] - v0)

        dx = np.array([dv0, dv1])

        return dx

