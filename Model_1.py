import os
import numpy as np
import torch as T
import torch.nn as nn

class IncrementalModel_1(nn.Module): #Q network
    def __init__(self, kappa, state_dims, action_dims, disturb_dims, name, chkpt_dir='tmp/ddpg'):
        super(IncrementalModel_1, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        self.kappa = kappa

        self.x_1 = np.zeros((state_dims, 1))
        self.u_1 = np.zeros((action_dims,1))
        self.d_1 = np.zeros((disturb_dims, 1))

        self.F_1 = np.eye(state_dims)
        self.G_1 = np.zeros((state_dims, action_dims))
        self.D_1 = np.zeros((state_dims, disturb_dims))

        self.F_1_tensor = T.tensor(self.F_1, dtype=T.float, requires_grad=False)
        self.G_1_tensor = T.tensor(self.G_1, dtype=T.float, requires_grad=False)
        self.D_1_tensor = T.tensor(self.D_1, dtype=T.float, requires_grad=False)
        print('self.F_1',self.F_1)
        print('self.G_1',self.G_1)
        print('self.D_1',self.D_1)


        self.Theta_1 = np.concatenate((self.F_1.T, self.G_1.T, self.D_1), axis=0)
        self.Tau_1 = 10 ** 8 * np.eye(state_dims + action_dims + disturb_dims)

    def forward(self, x_1, u_1, d_1, x0, u0, d0):

        x_error = T.add(x0, -x_1)
        u_error = T.add(u0, -u_1)
        d_error = T.add(d0, -d_1)

        y1 = T.add(T.matmul(x_error, self.F_1_tensor.T),T.matmul(u_error, self.G_1_tensor.T))
        y2 = T.add(y1,T.matmul(d_error, self.D_1_tensor.T))

        x1 = T.add(y2, x0)

        return x1

    def update(self, x_1, u_1, d_1, x0, u0, d0, x1):

        delta_x0 = x0 - x_1
        delta_u0 = u0 - u_1
        delta_d0 = d0 - d_1

        X0 = np.concatenate((delta_x0, delta_u0, delta_d0),axis=0).reshape(3,1)

        delta_x1_estimate = np.dot(X0.T, self.Theta_1)

        delta_x1 = x1 - x0

        epsilon0 = delta_x1 - delta_x1_estimate

        Theta0 = self.Theta_1 + (np.dot(np.dot(self.Tau_1, X0), epsilon0)) / (self.kappa + np.dot(np.dot(X0.T,self.Tau_1),X0))

        Tau0 = 1 / (self.kappa) * (self.Tau_1 - (np.dot(np.dot(np.dot(self.Tau_1, X0),X0.T),self.Tau_1)) / (self.kappa + np.dot(np.dot(X0.T,self.Tau_1),X0)))

        F0 = Theta0[0:1].T
        G0 = Theta0[1:2].T
        D0 = Theta0[2:3].T


        self.F_1 = F0
        self.G_1 = G0
        self.D_1 = D0

        self.Tau_1 = Tau0
        self.Theta_1 = Theta0

        self.F_1_tensor = T.tensor(self.F_1, dtype=T.float, requires_grad=False)
        self.G_1_tensor = T.tensor(self.G_1, dtype=T.float, requires_grad=False)
        self.D_1_tensor = T.tensor(self.D_1, dtype=T.float, requires_grad=False)




    def save_checkpoint(self):
        print('... saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))


#note:
#1.转置 .T 和 .transponse() 的区别
#当你需要简单地获取数组的转置时，可以使用.T属性。
#当你需要更复杂的转置操作，特别是当需要指定新的维度顺序时，应使用numpy.transpose()函数，并可能为其提供一个参数来指定新的维度顺序