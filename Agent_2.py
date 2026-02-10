import numpy as np
import torch as T
import torch.nn.functional as F

from Networks_2 import CriticNetwork_2, ActorNetwork_2
from Model_2 import IncrementalModel_2

class Agent_2(object):
    def __init__(self, alpha, beta, state_dims, action_dims, disturb_dims, lay1_dims, gamma, tau, kappa):
        self.alpha = alpha
        self.beta = beta
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.disturb_dims = disturb_dims

        self.lay1_dims = lay1_dims
        self.gamma = gamma
        self.tau = tau
        self.kappa = kappa

        self.critic = CriticNetwork_2(self.alpha, 2 * self.state_dims, self.lay1_dims, name='Critic')
        self.target_critic = CriticNetwork_2(self.alpha, 2 * self.state_dims, self.lay1_dims, name='TargetCritic')
        self.actor = ActorNetwork_2(self.beta, 2 * self.state_dims + self.disturb_dims, self.lay1_dims, self.action_dims, name='Actor')
        self.target_actor = ActorNetwork_2(self.beta, 2 * self.state_dims + disturb_dims, self.lay1_dims, self.action_dims, name='TargetActor')
        self.incrementalmodel = IncrementalModel_2(self.kappa, self.state_dims, self.action_dims, self.disturb_dims, name='IncrementalModel')
        self.update_critic_network_parameters(tau=1) #run this function to initialize target networks.
        self.update_actor_network_parameters(tau=1) #run this function to initialize target networks.

        self.k_index = 0
        self.Actor_2_Lay1_grad_history = np.zeros((150000, 21))
        self.Actor_2_Lay2_grad_history = np.zeros((150000, 7))

        self.y_index = 0
        self.lay2_out_history = np.zeros((150000, 1))

        self.grad_history = np.zeros((150000, 3))
        self.grad_index = 0

        self.error_memory = np.zeros((150000, 1))
        self.error_index = 0
        self.averaged_threshold = 0.2

        self.terminal_memory = np.zeros((150000, 1))
        self.terminal_index = 0

    def choose_action(self, s0, error0, disturbance0):

        self.actor.eval()
        s0_aug = T.tensor(np.concatenate((s0, error0, disturbance0)), dtype=T.float).to(self.actor.device)

        action = self.actor.forward(s0_aug).to(self.actor.device)  # 计算控制器输出

        action_noise = action
        #action_noise = action + T.tensor(self.noise(), dtype=T.float).to(self.actor.device) #控制器加噪声

        self.lay2_out_history[self.y_index] = self.actor.lay2_out_data.detach().numpy()
        self.y_index = self.y_index + 1

        return action_noise.cpu().detach().numpy()

    def learn(self, s_1, a_1, disturbance_1, s0, error0, disturbance0, t0, s0_ref, s1_ref):

        if self.terminal_memory[self.terminal_index - 1] == 1:
            return

        #compute target
        s_1_tensor = T.tensor(s_1, dtype=T.float, requires_grad=False).to(self.critic.device)
        a_1_tensor = T.tensor(a_1, dtype=T.float, requires_grad=False).to(self.critic.device)
        disturbance_1_tensor = T.tensor(disturbance_1, dtype=T.float, requires_grad=False).to(self.critic.device)
        s0_tensor = T.tensor(s0, dtype=T.float, requires_grad=False).to(self.critic.device)
        error0_tensor = T.tensor(error0, dtype=T.float, requires_grad=False).to(self.critic.device)
        disturbance0_tensor = T.tensor(disturbance0, dtype=T.float, requires_grad=False).to(self.critic.device)

        s0_ref_tensor = T.tensor(s0_ref, dtype=T.float,requires_grad=False).to(self.critic.device)
        s1_ref_tensor = T.tensor(s1_ref, dtype=T.float,requires_grad=False).to(self.critic.device)
        #print('s0_tensor',s0_tensor)
        s0_aug = T.tensor(np.concatenate((s0, error0, disturbance0)), dtype=T.float, requires_grad=False).to(self.critic.device)
        s0_aug_critic = T.tensor(np.concatenate((s0, error0)), dtype=T.float, requires_grad=False).to(self.critic.device)

        self.actor.eval()
        a0_tensor = self.actor.forward(s0_aug)
        #u0_tensor = T.tensor([1],dtype=T.float,requires_grad=True)
        #One-step interaction with model
        disturbance1_tensor = disturbance0_tensor
        s1_tensor = self.incrementalmodel.forward(s_1_tensor,a_1_tensor, disturbance_1_tensor, s0_tensor, a0_tensor, disturbance0_tensor)
        error1_tensor = s1_tensor - s1_ref_tensor

        s1_aug = T.cat([s1_tensor,error1_tensor,disturbance1_tensor]).to(self.critic.device)
        s1_aug_critic = T.cat([s1_tensor,error1_tensor]).to(self.critic.device)

        a1_tensor = self.target_actor.forward(s1_aug)

        #reward
        r = T.pow(error1_tensor, 2) + 0.00001 * T.pow(a0_tensor, 2)
        #- 0.0001 * T.abs(a1_tensor-a0_tensor)
        #- 0.00051 * T.abs(u0_tensor)
        #- 0.00005 * T.abs(u1_tensor-u0_tensor)
        #r = -T.matmul(e1_tensor**2, mat)


        #value
        s1_error1 = T.cat([s1_tensor, error1_tensor], dim=0)
        value = self.target_critic(s1_error1)

        target = r + self.gamma * value

        #========================================train critic===============================================

        i = 1
        while True:
            self.actor.eval()
            self.critic.eval()
            critic_out = self.critic.forward(s0_aug_critic)
            self.critic.train()
            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(critic_out, target)
            print('critic2_loss', critic_loss)

            if critic_loss < 0.0001:
                break

            critic_loss.backward(retain_graph=True)
            self.critic.optimizer.step()
            i = i + 1
            if i > 50:
                break

        self.update_critic_network_parameters(tau=1)

        #=========================================train actor==================================================
        j = 1
        loss_1 = T.tensor([10000])
        while True:
            self.actor.train()
            self.actor.optimizer.zero_grad()

            self.actor.eval()
            a0_tensor = self.actor.forward(s0_aug)
            s1_tensor = self.incrementalmodel.forward(s_1_tensor, a_1_tensor, disturbance_1_tensor, s0_tensor, a0_tensor, disturbance0_tensor)
            error1_tensor = s1_tensor - s1_ref_tensor

            s1_aug = T.cat([s1_tensor, error1_tensor, disturbance1_tensor], dim=0).to(self.critic.device)
            s1_aug_critic = T.cat([s1_tensor, error1_tensor], dim=0).to(self.critic.device)

            a1_tensor = self.target_actor.forward(s1_aug)

            #0.0114/0.0115
            r = T.pow(error1_tensor, 2) + 0.00001 * T.pow(a0_tensor, 2)
            #r = -T.abs(error1_tensor) - 0.00001 * T.abs(a1_tensor-a0_tensor)
            #- 0.00001* T.abs(u1_tensor-u0_tensor)
            #- 0.00000001 * T.abs(u0_tensor)
            #- 0.00000001 * T.abs(u0_tensor)
            #r = -T.matmul(e1_tensor ** 2, mat)
            #s1_aug = T.cat([s1_tensor, error1_tensor, disturbance1_tensor], dim=0)
            value = self.target_critic(s1_aug_critic)
            #target = r + self.gamma * value + 0.00001 * T.abs(a1_tensor - a0_tensor)
            target = r + self.gamma * value

            #lya = self.target_critic(x1_e1) + 0.5 -self.target_critic(x0_aug)

            actor_loss = target
            print('actor2_loss', actor_loss)

            if actor_loss - loss_1 > 0:
                break

            self.actor.train()
            actor_loss.backward()

            if j == 1:
                self.Actor_2_Lay1_grad_history[self.k_index] = self.actor.lay1.weight.grad.detach().numpy().flatten()
                self.Actor_2_Lay2_grad_history[self.k_index] = self.actor.lay2.weight.grad.detach().numpy().flatten()
                self.k_index = self.k_index + 1

            self.actor.optimizer.step()
            loss_1 = actor_loss
            j = j + 1

            if j > 50:
                break

        self.update_actor_network_parameters(tau=1) #run this function to initialize target networks.


    def update_critic_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_params = self.critic.named_parameters()
        critic_params_dict = dict(critic_params)

        target_critic_params = self.target_critic.named_parameters()
        target_critic_params_dict = dict(target_critic_params)

        for name in critic_params_dict:
            critic_params_dict[name] = tau*critic_params_dict[name].clone() + \
                                    (1-tau)*target_critic_params_dict[name].clone()

        self.target_critic.load_state_dict(critic_params_dict)

    def update_actor_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        actor_params_dict = dict(actor_params)

        target_actor_params = self.target_actor.named_parameters()
        target_actor_params_dict = dict(target_actor_params)

        for name in actor_params_dict:
            actor_params_dict[name] = tau*actor_params_dict[name].clone() + \
                                    (1-tau)*target_actor_params_dict[name].clone()

        self.target_actor.load_state_dict(actor_params_dict)

    def save_models(self):
        self.critic.save_checkpoint()
        self.actor.save_checkpoint()

    def load_models(self):
        self.critic.load_checkpoint()
        self.actor.load_checkpoint()

    def gain(self, s0, error0, disturbance0):

        state_aug = T.tensor(np.concatenate((s0, error0, disturbance0)), dtype=T.float, requires_grad=True).to(self.critic.device)

        loss = self.actor.forward(state_aug)

        self.actor.optimizer.zero_grad()

        loss.backward()

        #print('x0_aug.grad',state_aug.grad)

        self.grad_history[self.grad_index] = state_aug.grad.detach().numpy().flatten()
        self.grad_index = self.grad_index + 1

    def memory_error(self, error0):

        self.error_memory[self.error_index] = error0
        self.error_index = self.error_index + 1

    def termination_evaluation(self):

        error_history = self.error_memory[:self.error_index]
        error_evaluated = error_history[-5000:]
        size_error_evaluated = error_evaluated.size

        threshold = size_error_evaluated * self.averaged_threshold

        sum_error_evaluated = np.sum(np.abs(error_evaluated))

        terminal = np.array((sum_error_evaluated - threshold) < 0, dtype=int)

        self.terminal_memory[self.terminal_index] = terminal
        self.terminal_index = self.terminal_index + 1

    def choose_action_zero(self, x0, e0, d0):
        self.actor.eval()
        x0_aug = T.tensor(np.concatenate((x0, e0, d0)), dtype=T.float).to(self.actor.device)

        action = self.actor.forward(x0_aug).to(self.actor.device)  # 计算控制器输出

        action_noise = action
        # action_noise = action + T.tensor(self.noise(), dtype=T.float).to(self.actor.device) #控制器加噪声

        # self.lay2_out_history[self.x_index] = self.actor.lay2_out_data.detach().numpy()
        # self.x_index = self.x_index + 1

        return action_noise.cpu().detach().numpy()




