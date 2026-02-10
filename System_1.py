import numpy as np
import math


class RK4_1(object):
    def __init__(self, step):
        self.step = step

    def solver(self, t, x0, u0, d0):  # the input x0,u0 are required to be np.arrary. #function solver solves x1 and reward
        k1 = self.system(t, x0, u0, d0)
        k2 = self.system(t + self.step / 2.0, x0 + self.step * np.transpose(k1) / 2.0, u0, d0)
        k3 = self.system(t + self.step / 2.0, x0 + self.step * np.transpose(k2) / 2.0, u0, d0)
        k4 = self.system(t + self.step / 2.0, x0 + self.step * np.transpose(k3), u0, d0)

        x1 = x0 + self.step * np.transpose(k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        terminated = False
        return np.array(x1, dtype=np.float32), terminated, False, {}

    def system(self, t, x0, u0, d0): #alpha dynamics
        alpha = x0[0]
        q = u0[0]
        delta = d0[0]

        bz = -0.034
        phiz = 0.000103*(alpha)**3 - 0.00945*alpha*abs(alpha) - 0.170 * alpha

        g = 32.2 * 0.3048
        W = 450 * 0.454
        V = 3109.3 * 0.3048
        f = 180/np.pi
        Q = 6132.8 * 0.454/(0.3048**2)
        S = 0.44 * 0.3048**2

        dalpha = (f*g*Q*S/(W*V)) * math.cos(alpha*np.pi/180) * (phiz+bz*delta) + q

        s = (f*g*Q*S/(W*V)) * math.cos(alpha*np.pi/180) * (phiz+bz*delta)
        print('s',s)

        dxt = np.array([dalpha])



        return dxt



