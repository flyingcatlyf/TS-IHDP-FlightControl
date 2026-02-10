import numpy as np
import math

class RK4_2(object):
    def __init__(self, step):
        self.step = step

    def solver(self, t, s0, a0, disturbance0):  # the input x0,u0 are required to be np.arrary. #function solver solves x1 and reward
        k1 = self.system(t, s0, a0, disturbance0)
        k2 = self.system(t + self.step / 2.0, s0 + self.step * np.transpose(k1) / 2.0, a0, disturbance0)
        k3 = self.system(t + self.step / 2.0, s0 + self.step * np.transpose(k2) / 2.0, a0, disturbance0)
        k4 = self.system(t + self.step / 2.0, s0 + self.step * np.transpose(k3), a0, disturbance0)

        s1 = s0 + self.step * np.transpose(k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        terminated = False
        return np.array(s1, dtype=np.float32), terminated, False, {}

    def system(self, t, s0, a0, disturbance0):
        q = s0[0]
        delta = a0[0]
        alpha = disturbance0[0]

        bm = -0.206
        phim = 0.000215*(alpha)**3 - 0.01950*alpha*abs(alpha) + 0.051 * alpha


        Iyy = 182.5 * 14.59396 * 0.3048**2
        f = 180/np.pi
        Q = 6132.8 * 0.454/(0.3048**2)
        S = 0.44 * 0.3048**2
        d = 0.75 * 0.3048

        dq = (f*Q*S*d/Iyy) * (phim + bm * delta)


        dst = np.array([dq])



        return dst



