import numpy as np
import torch as T
import torch.nn.functional as F

from Networks_1 import CriticNetwork_1, ActorNetwork_1
from Model_1 import IncrementalModel_1

class Agent_1(object):
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

        self.critic = CriticNetwork_1(self.alpha, 2 * self.state_dims, self.lay1_dims, name='Critic')
        self.target_critic = CriticNetwork_1(self.alpha, 2 * self.state_dims, self.lay1_dims, name='TargetCritic')
        self.actor = ActorNetwork_1(self.beta, 2 * self.state_dims + self.disturb_dims, self.lay1_dims, self.action_dims, name='Actor')
        self.target_actor = ActorNetwork_1(self.beta, 2 * self.state_dims + disturb_dims, self.lay1_dims, self.action_dims, name='TargetActor')
        self.incrementalmodel = IncrementalModel_1(self.kappa, self.state_dims, self.action_dims, self.disturb_dims, name='IncrementalModel')
        self.update_critic_network_parameters(tau=1) #run this function to initialize target networks.
        self.update_actor_network_parameters(tau=1) #run this function to initialize target networks.

        self.v_memory = np.zeros((500000, 6))
        self.index = 0

        self.i_index = 0
        self.Actor_1_Lay1_grad_history = np.zeros((150000, 21))
        self.Actor_1_Lay2_grad_history = np.zeros((150000, 7))

        self.x_index = 0
        self.x_disturb_index = 0

        self.lay2_out_history = np.zeros((150000, 1))

        self.grad_history = np.zeros((150000, 3))
        self.grad_index = 0

        self.e_memory = np.zeros((150000, 1))
        self.e_index = 0
        self.averaged_threshold = 0.5

        self.terminal_memory = np.zeros((150000, 1))
        self.terminal_index = 0

    def choose_action(self, x0, e0, d0):

        self.actor.eval()
        x0_aug = T.tensor(np.concatenate((x0, e0, d0)), dtype=T.float).to(self.actor.device)

        action = self.actor.forward(x0_aug).to(self.actor.device)  # 计算控制器输出

        action_noise = action
        #action_noise = action + T.tensor(self.noise(), dtype=T.float).to(self.actor.device) #控制器加噪声

        self.lay2_out_history[self.x_index] = self.actor.lay2_out_data.detach().numpy()
        self.x_index = self.x_index + 1

        return action_noise.cpu().detach().numpy()

    def learn(self, x_1, u_1, d_1, x0, e0, d0, t0, x0_ref, x1_ref):

        if self.terminal_memory[self.terminal_index - 1] == 1:
            return

        #compute target
        x_1_tensor = T.tensor(x_1, dtype=T.float, requires_grad=False).to(self.critic.device)
        u_1_tensor = T.tensor(u_1, dtype=T.float, requires_grad=False).to(self.critic.device)
        d_1_tensor = T.tensor(d_1, dtype=T.float, requires_grad=False).to(self.critic.device)
        x0_tensor = T.tensor(x0, dtype=T.float, requires_grad=False).to(self.critic.device)
        e0_tensor = T.tensor(e0, dtype=T.float, requires_grad=False).to(self.critic.device)
        d0_tensor = T.tensor(d0, dtype=T.float, requires_grad=False).to(self.critic.device)
        d1_tensor = T.tensor(d0, dtype=T.float, requires_grad=False).to(self.critic.device)


        x0_ref_tensor = T.tensor(x0_ref, dtype=T.float,requires_grad=False).to(self.critic.device)
        x1_ref_tensor = T.tensor(x1_ref, dtype=T.float,requires_grad=False).to(self.critic.device)

        x0_aug = T.tensor(np.concatenate((x0, e0, d0)), dtype=T.float, requires_grad=False).to(self.critic.device)
        x0_aug_critic = T.tensor(np.concatenate((x0, e0)), dtype=T.float, requires_grad=False).to(self.critic.device)

        self.actor.eval()
        u0_tensor = self.actor.forward(x0_aug)



        #One-step interaction with model
        x1_tensor = self.incrementalmodel.forward(x_1_tensor,u_1_tensor,d_1_tensor,x0_tensor,u0_tensor,d0_tensor)
        e1_tensor = x1_tensor - x1_ref_tensor

        x1_aug = T.cat([x1_tensor,e1_tensor,d1_tensor]).to(self.critic.device)
        x1_aug_critic = T.cat([x1_tensor,e1_tensor]).to(self.critic.device)

        u1_tensor = self.target_actor.forward(x1_aug)

        #reward
        r = T.pow(e1_tensor,2) + 0.000005 * T.pow(u0_tensor, 2)
        #- 0.0011 * T.abs(u0_tensor)
        #- 0.00095 * T.abs(u1_tensor-u0_tensor)
        # - 0.00001 * T.abs(u0_tensor)

        #value
        value = self.target_critic(x1_aug_critic)

        target = r + self.gamma * value

        #========================================train critic===============================================

        i = 1
        while True:
            self.actor.eval()
            self.critic.eval()
            critic_out = self.critic.forward(x0_aug_critic)
            self.critic.train()
            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(critic_out, target)
            print('critic_loss', critic_loss)

            if critic_loss < 0.0005:
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
            u0_tensor = self.actor.forward(x0_aug)
            x1_tensor = self.incrementalmodel.forward(x_1_tensor,u_1_tensor,d_1_tensor,x0_tensor,u0_tensor,d0_tensor)
            e1_tensor = x1_tensor - x1_ref_tensor

            x1_aug = T.cat([x1_tensor, e1_tensor, d1_tensor]).to(self.critic.device)
            x1_aug_critic = T.cat([x1_tensor, e1_tensor]).to(self.critic.device)

            u1_tensor = self.target_actor.forward(x1_aug)

            noise = 0.3 * np.random.random([1])
            x0_noise = x0 + noise
            e0_noise = e0 + noise

            #noise = 0.0001 * np.sign(e0)
            #x0_noise = x0 - noise
            #e1_noise = e0 - noise
            #np.array([0.0001])
            #noise = np.array([0.0001])
            #noise = 0.004 * np.random.random(1)
            #print('noise',noise)
            #print('x0',x0)
            #print('x0 + noise', x0 + noise)
            #print('e0',e0)
            #print('e0+noise',e0+noise)
            x0_aug_noise = T.tensor(np.concatenate((x0_noise, e0_noise, d0)), dtype=T.float, requires_grad=False).to(self.critic.device)
            #print('x0_aug_noise',x0_aug_noise)
            u0_noise_tensor = self.target_actor.forward(x0_aug_noise)

            #0.0114/0.0115
            #r = -T.abs(e1_tensor) - 0.00098 * T.abs(u1_tensor - u0_tensor)

            r = T.pow(e1_tensor,2) + 0.000005 * T.pow(u0_tensor, 2)
            #r = T.pow(e1_tensor,2) + 0.00001 * T.pow(u0_tensor, 2) + 0.00098 * T.abs(u1_tensor - u0_tensor)

            #- 0.001 * T.pow(u0_tensor, 2)
            #- 0.000133 * T.abs(u0_noise_tensor - u0_tensor)
            #- 0.00098 * T.abs(u1_tensor - u0_tensor)
            #- 0.00000002 * T.abs(u0_noise_tensor- u0_tensor)
            #- 0.00098 * T.abs(u1_tensor - u0_tensor)
            #- 0.00001 * T.abs(u0_tensor) - 0.00095 * T.abs(u1_tensor- u0_tensor)

            #- 0.00001 * T.abs(u0_tensor)

            #value
            value = self.target_critic(x1_aug_critic)
            #target = r + self.gamma * value + 0.00098 * T.abs(u1_tensor - u0_tensor)
            target = r + self.gamma * value


            #lya = self.target_critic(x1_e1) + 0.5 -self.target_critic(x0_aug)

            actor_loss = target
            print('actor_loss', actor_loss)

            if actor_loss - loss_1 > 0:
                break

            self.actor.train()
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)

            if j == 1:
                self.Actor_1_Lay1_grad_history[self.i_index] = self.actor.lay1.weight.grad.detach().numpy().flatten()
                self.Actor_1_Lay2_grad_history[self.i_index] = self.actor.lay2.weight.grad.detach().numpy().flatten()
                self.i_index = self.i_index + 1

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
        self.actor_disturb.save_checkpoint()


    def load_models(self):
        self.critic.load_checkpoint()
        self.actor.load_checkpoint()
        self.actor_disturb.load_checkpoint()

    def store_transition(self, V0, V1, Ve, V0_system, V1_system, Ve_system):

        #print('V0',V0.detach().numpy())
        self.v_memory[self.index] = np.array([V0.detach().numpy()[0],V1.detach().numpy()[0], Ve.detach().numpy()[0],V0_system.detach().numpy()[0],V1_system.detach().numpy()[0],Ve_system.detach().numpy()[0]])
        self.index = self.index + 1

    def gain(self, x0, e0, d0):

        state_aug = T.tensor(np.concatenate((x0, e0, d0)), dtype=T.float, requires_grad=True).to(self.critic.device)

        loss = self.actor.forward(state_aug)

        self.actor.optimizer.zero_grad()

        loss.backward()

        #print('x0_aug.grad',state_aug.grad)

        self.grad_history[self.grad_index] = state_aug.grad.detach().numpy().flatten()
        self.grad_index = self.grad_index + 1

    def memory_error(self, e0):

        self.e_memory[self.e_index] = e0
        self.e_index = self.e_index + 1

    def termination_evaluation(self):

        e_history = self.e_memory[:self.e_index]
        e_evaluated = e_history[-10000:]
        size_e_evaluated = e_evaluated.size

        threshold = size_e_evaluated * self.averaged_threshold

        sum_e_evaluated = np.sum(np.abs(e_evaluated))

        terminal = np.array((sum_e_evaluated - threshold) < 0, dtype=int)

        #print('terminal',terminal)

        self.terminal_memory[self.terminal_index] = terminal
        self.terminal_index = self.terminal_index + 1

        #if evaluated_e_sum < threshold:
        #        terminal = 1
        #        agent_1.terminal_memory[agent_1.terminal_index] = terminal
        #        agent_1.terminal_index = agent_1.terminal_index + 1
        #else:
        #        terminal = 0
        #        agent_1.terminal_memory[agent_1.terminal_index] = terminal
        #        agent_1.terminal_index = agent_1.terminal_index + 1

    def choose_action_zero(self, x0, e0, d0):
        self.actor.eval()
        x0_aug = T.tensor(np.concatenate((x0, e0, d0)), dtype=T.float).to(self.actor.device)

        action = self.actor.forward(x0_aug).to(self.actor.device)  # 计算控制器输出

        action_noise = action
        # action_noise = action + T.tensor(self.noise(), dtype=T.float).to(self.actor.device) #控制器加噪声

        # self.lay2_out_history[self.x_index] = self.actor.lay2_out_data.detach().numpy()
        # self.x_index = self.x_index + 1

        return action_noise.cpu().detach().numpy()










