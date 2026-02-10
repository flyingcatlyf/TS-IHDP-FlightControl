import os
import torch as T
import torch.nn as nn
import torch.optim as optim

class CriticNetwork_1(nn.Module):
    def __init__(self, alpha, input_dims, lay1_dims, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork_1, self).__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.lay1_dims = lay1_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        self.lay1 = nn.Linear(self.input_dims, self.lay1_dims, bias=False)
        self.lay2 = nn.Linear(self.lay1_dims, 1, bias=False)
        self.bn1 = nn.LayerNorm(self.lay1_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda：0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.lay1.weight.data.uniform_(-0.01, 0.01)
        self.lay2.weight.data.uniform_(-0.01, 0.01)

        self.lay1weight = self.lay1.weight.data
        self.lay2weight = self.lay2.weight.data

        #self.lay1.weight.data = 0.001 * T.ones((3, 2))
        #self.lay2.weight.data = 0.001 * T.ones((3, 2))

        self.lay2_out_data = T.zeros((1,1),requires_grad=False)

    def forward(self, state):
        lay1_out = self.lay1(state)
        lay1_out = T.tanh(lay1_out)

        lay2_out = self.lay2(lay1_out)

        self.lay2_out_data = lay2_out

        lay2_out = T.abs(lay2_out)

        return lay2_out

    def save_checkpoint(self):
        print('... saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork_1(nn.Module):
    def __init__(self, beta, input_dims, lay1_dims, action_dims, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork_1,self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.lay1_dims = lay1_dims
        self.actions_dims = action_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        self.lay1 = nn.Linear(self.input_dims, self.lay1_dims, bias=False)
        self.lay2 = nn.Linear(self.lay1_dims, action_dims, bias=False)
        self.bn1 = nn.LayerNorm(self.input_dims)
        self.bn2 = nn.LayerNorm(self.actions_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda：0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.lay1.weight.data.uniform_(-0.01, 0.01)
        self.lay2.weight.data.uniform_(-0.01, 0.01)

        self.lay1weight = self.lay1.weight.data
        self.lay2weight = self.lay2.weight.data

        #self.lay1.bias.data.uniform_(-0.0001, 0.0001)
        #self.lay2.bias.data.uniform_(-0.0001, 0.0001)

        #self.lay1.weight.data = 0.001 * T.ones((3, 2))
        #self.lay2.weight.data = 0.001 * T.ones((3, 2))

        self.lay2_out_data = T.zeros((1,1),requires_grad=False)

    def forward(self, state):

        lay1_out = self.lay1(state)
        lay1_out = T.tanh(lay1_out)

        #print('lay1_out',lay1_out)
        #print('self.lay2.weight.data',self.lay2.weight.data)
        #lay1_out_numpy = lay1_out.detach().numpy().flatten()
        #lay2_weight_data_numpy = self.lay2.weight.data.detach().numpy().flatten()
        #lay2_in_vector = lay1_out_numpy * lay2_weight_data_numpy
        #self.lay2_in_set =
        #print('lay2_in_vector',lay2_in_vector)

        lay2_out = self.lay2(lay1_out)

        self.lay2_out_data = lay2_out


        lay2_out = 20 * T.tanh(lay2_out)

        return lay2_out

    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

