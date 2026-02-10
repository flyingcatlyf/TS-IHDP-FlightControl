import numpy as np
import torch as T
import math
from System_1 import RK4_1
from System_2 import RK4_2
from Agent_1 import Agent_1
from Agent_2 import Agent_2
from Ref import Ref_Sin
from Excitation import Excitation_sin

from filter import second_order_filter

env_aircraft_1 = RK4_1(step=0.001)
env_aircraft_2 = RK4_2(step=0.001)

agent_1 = Agent_1(alpha=0.1, beta=0.0000005, state_dims=1, action_dims=1, disturb_dims=1, lay1_dims=7, gamma=0.6, tau=0.01, kappa=1.0)
agent_2 = Agent_2(alpha=0.1, beta=0.0000001, state_dims=1, action_dims=1, disturb_dims=1, lay1_dims=7, gamma=0.6, tau=0.01, kappa=1.0)
ref = Ref_Sin(T=1000)
excitation = Excitation_sin(T=1000)
filter = second_order_filter(step=0.001, omega = 20, epsilon = 0.7)

#print('agent_1.critic.lay1.weight_origion',agent_1.critic.lay1.weight)
#print('agent_1.critic.lay2.weight_origion',agent_1.critic.lay2.weight)
#print('agent_2.critic.lay1.weight_origion',agent_2.critic.lay1.weight)
#print('agent_2.critic.lay2.weight_origion',agent_2.critic.lay2.weight)
#print('agent_1.actor.lay1.weight_origion',agent_1.actor.lay1.weight)
#print('agent_1.actor.lay2.weight_origion',agent_1.actor.lay2.weight)
#print('agent_2.actor.lay1.weight_origion',agent_2.actor.lay1.weight)
#print('agent_2.actor.lay2.weight_origion',agent_2.actor.lay2.weight)

critic1_lay1_weight = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/Critic_1_Lay1_history')
critic1_lay2_weight = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/Critic_1_Lay2_history')
critic2_lay1_weight = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/Critic_2_Lay1_history')
critic2_lay2_weight = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/Critic_2_Lay2_history')
actor1_lay1_weight = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/Actor_1_Lay1_history')
actor1_lay2_weight = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/Actor_1_Lay2_history')
actor2_lay1_weight = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/Actor_2_Lay1_history')
actor2_lay2_weight = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/Actor_2_Lay2_history')


#critic1_lay1_weight0 = np.reshape(critic1_lay1_weight[:1,:],(7,2))
#critic1_lay2_weight0 = np.reshape(critic1_lay2_weight[:1,:],(1,7))
#print('critic1_lay1_weight_new',critic1_lay1_weight0)
#print('critic1_lay2_weight_new',critic1_lay2_weight0)

critic1_lay1_weight0 = np.reshape(critic1_lay1_weight[:1,:],(7,2))
critic1_lay2_weight0 = np.reshape(critic1_lay2_weight[:1,:],(1,7))
critic2_lay1_weight0 = np.reshape(critic2_lay1_weight[:1,:],(7,2))
critic2_lay2_weight0 = np.reshape(critic2_lay2_weight[:1,:],(1,7))
actor1_lay1_weight0 = np.reshape(actor1_lay1_weight[:1,:],(7,3))
actor1_lay2_weight0 = np.reshape(actor1_lay2_weight[:1,:],(1,7))
actor2_lay1_weight0 = np.reshape(actor2_lay1_weight[:1,:],(7,3))
actor2_lay2_weight0 = np.reshape(actor2_lay2_weight[:1,:],(1,7))

#print('critic1_lay1_weight_new',critic1_lay1_weight0)
#print('critic1_lay1_weight_new',critic1_lay1_weight0)
#print('critic1_lay1_weight_new',critic1_lay1_weight0)
#print('critic1_lay1_weight_new',critic1_lay1_weight0)


agent_1.critic.lay1.weight.data = T.tensor(critic1_lay1_weight0,dtype=T.float)
agent_1.critic.lay2.weight.data = T.tensor(critic1_lay2_weight0,dtype=T.float)
agent_2.critic.lay1.weight.data = T.tensor(critic2_lay1_weight0,dtype=T.float)
agent_2.critic.lay2.weight.data = T.tensor(critic2_lay2_weight0,dtype=T.float)
agent_1.actor.lay1.weight.data = T.tensor(actor1_lay1_weight0,dtype=T.float)
agent_1.actor.lay2.weight.data = T.tensor(actor1_lay2_weight0,dtype=T.float)
agent_2.actor.lay1.weight.data = T.tensor(actor2_lay1_weight0,dtype=T.float)
agent_2.actor.lay2.weight.data = T.tensor(actor2_lay2_weight0,dtype=T.float)

agent_1.target_critic.lay1.weight.data = T.tensor(critic1_lay1_weight0,dtype=T.float)
agent_1.target_critic.lay2.weight.data = T.tensor(critic1_lay2_weight0,dtype=T.float)
agent_2.target_critic.lay1.weight.data = T.tensor(critic2_lay1_weight0,dtype=T.float)
agent_2.target_critic.lay2.weight.data = T.tensor(critic2_lay2_weight0,dtype=T.float)
agent_1.target_actor.lay1.weight.data = T.tensor(actor1_lay1_weight0,dtype=T.float)
agent_1.target_actor.lay2.weight.data = T.tensor(actor1_lay2_weight0,dtype=T.float)
agent_2.target_actor.lay1.weight.data = T.tensor(actor2_lay1_weight0,dtype=T.float)
agent_2.target_actor.lay2.weight.data = T.tensor(actor2_lay2_weight0,dtype=T.float)

#print('agent_1.critic.lay1.weight_origion',agent_1.critic.lay1.weight)
#print('agent_1.critic.lay2.weight_origion',agent_1.critic.lay2.weight)
#print('agent_2.critic.lay1.weight_origion',agent_2.critic.lay1.weight)
#print('agent_2.critic.lay2.weight_origion',agent_2.critic.lay2.weight)
#print('agent_1.actor.lay1.weight_origion',agent_1.actor.lay1.weight)
#print('agent_1.actor.lay2.weight_origion',agent_1.actor.lay2.weight)
#print('agent_2.actor.lay1.weight_origion',agent_2.actor.lay1.weight)
#print('agent_2.actor.lay2.weight_origion',agent_2.actor.lay2.weight)


#print('agent_1.critic.lay1.weight ',agent_1.critic.lay1.weight )

np.random.seed(0)


t0 = np.array([0])
Index0 = 0
step = 0.001
score0 = -10000
score_history = []
average_score_history = []

state_history = np.zeros((50000, 2))
action_history = np.zeros((50000, 2))
alpha_ref_history = np.zeros((50000, 1))
q_ref_noise_history = np.zeros((50000, 1))
q_ref_filtered_history = np.zeros((50000, 1))
actions_zero_error_history = np.zeros((50000, 6))


Critic_1_Lay1_history = np.zeros((50000,14))
Critic_1_Lay2_history = np.zeros((50000,7))
Critic_2_Lay1_history = np.zeros((50000,14))
Critic_2_Lay2_history = np.zeros((50000,7))
Actor_1_Lay1_history = np.zeros((50000,21))
Actor_1_Lay2_history = np.zeros((50000,7))
Actor_2_Lay1_history = np.zeros((50000,21))
Actor_2_Lay2_history = np.zeros((50000,7))
Theta_1_history = np.zeros((50000,3))
Theta_2_history = np.zeros((50000,3))

for i in range(1):

    score = 0

    amplitude = 10

    x0 = np.array([0]); s0 = np.array([0])
    x_1 = np.array([0]);s_1 = np.array([0])
    u_1 = np.array([0]);a_1 = np.array([0])

    xf0 = np.array([0,0])
    #print('xf0',xf0)

    a0 = np.array([0])
    #a_1 = np.array([0])
    #a_2 = np.array([0])
    #a_3 = np.array([0])
    #a_4 = np.array([0])
    #a_5 = np.array([0])

    q_1_ref = 0

    e0_constant = np.array([0])
    error0_constant = np.array([0])

    k = 0
    while k < 40000:
#====================================攻角参考指令===========================================================
        alpha0_ref = ref.fun(k, amplitude)
        alpha1_ref = ref.fun(k + 1, amplitude)

        x0_ref = np.array([alpha0_ref])
        x1_ref = np.array([alpha1_ref])
        e0 = np.subtract(x0, x0_ref)

#=====================================数据记录=========================================================
        Critic_1_Lay1_history[k + i * 300] = agent_1.critic.lay1.weight.data.detach().numpy().flatten()
        Critic_1_Lay2_history[k + i * 300] = agent_1.critic.lay2.weight.data.detach().numpy().flatten()
        Critic_2_Lay1_history[k + i * 300] = agent_2.critic.lay1.weight.data.detach().numpy().flatten()
        Critic_2_Lay2_history[k + i * 300] = agent_2.critic.lay2.weight.data.detach().numpy().flatten()
        Actor_1_Lay1_history[k + i * 300] = agent_1.actor.lay1.weight.data.detach().numpy().flatten()
        Actor_1_Lay2_history[k + i * 300] = agent_1.actor.lay2.weight.data.detach().numpy().flatten()
        Actor_2_Lay1_history[k + i * 300] = agent_2.actor.lay1.weight.data.detach().numpy().flatten()
        Actor_2_Lay2_history[k + i * 300] = agent_2.actor.lay2.weight.data.detach().numpy().flatten()



#===================================判断误差范围(agent 1)======================================================
        agent_1.memory_error(e0)
        agent_1.termination_evaluation()

#===================================训练1次  agent_1======================================================
        d0 = a0; d_1 = a_1

        agent_1.learn(x_1, u_1, d_1, x0, e0, d0, t0, x0_ref, x1_ref)  # [alpha_1, q_1, delta_1, alpha0, alpha0 - alpha_ref, delta0, delta0_ref, alpha1_ref]
        agent_1.learn(x_1, u_1, d_1, x0, e0, d0, t0, x0_ref, x1_ref)  # [alpha_1, q_1, delta_1, alpha0, alpha0 - alpha_ref, delta0, delta0_ref, alpha1_ref]
        agent_1.learn(x_1, u_1, d_1, x0, e0, d0, t0, x0_ref, x1_ref)  # [alpha_1, q_1, delta_1, alpha0, alpha0 - alpha_ref, delta0, delta0_ref, alpha1_ref]
        agent_1.gain(x0, e0, d0)

        q0_ref = agent_1.choose_action(x0,e0,d0)

        q0_ref_zero_error = agent_1.choose_action_zero(x0, e0_constant, d0)
        q0_ref_zero_alpha = agent_1.choose_action_zero(e0_constant, e0, d0)
        q0_ref_zero_delta = agent_1.choose_action_zero(x0, e0, e0_constant)

        #print('k',k)

#q0_ref = agent_1.actor.forward(T.tensor(np.concatenate((x0, e0, d0)), dtype = T.float, requires_grad=False)).detach().numpy() #tensor
        #noise = 0 * np.sin(0.01 * k)
        #noise = 5 * np.exp(-0.0005 * k) * np.sin(0.01 * k)
        #+ 3 * np.sin(-0.015 * k + 0.001)
        #+ 3 * np.sin(- 0.02 * k)
        #noise = 5 * np.exp(-0.0001 * k) * np.random.random([1])
        #q0_ref = q0_ref + noise

        #2-order filter
        #xf1, done, info, empty = filter.solver(t0, xf0, q0_ref)  # 计算当前一步状态
        #q0_ref_filtered = xf0[0]
        #q1_ref = xf1[0]

        #no filter
        q0_ref_filtered = q0_ref
        q1_ref = q0_ref

        error0 = np.subtract(s0, q0_ref_filtered)


#===================================判断误差范围(agent 2)======================================================
        agent_2.memory_error(error0)
        agent_2.termination_evaluation()

#===================================训练1次 agent_2========================================================
        disturbance_1 = x_1; disturbance0 = x0
        agent_2.learn(s_1, a_1, disturbance_1, s0, error0, disturbance0, t0, q0_ref_filtered, q1_ref) #[q_1, delta_1, alpha_1, q0, q0-q0_ref, alpha0, q0_ref, q1_ref]
        agent_2.learn(s_1, a_1, disturbance_1, s0, error0, disturbance0, t0, q0_ref_filtered, q1_ref)  # [q_1, delta_1, alpha_1, q0, q0-q0_ref, alpha0, q0_ref, q1_ref]
        agent_2.learn(s_1, a_1, disturbance_1, s0, error0, disturbance0, t0, q0_ref_filtered, q1_ref)  # [q_1, delta_1, alpha_1, q0, q0-q0_ref, alpha0, q0_ref, q1_ref]
        Theta_1_history[k + i * 300] = agent_1.incrementalmodel.Theta_1.reshape((3,))
        Theta_2_history[k + i * 300] = agent_2.incrementalmodel.Theta_1.reshape((3,))

        agent_2.gain(s0, error0, disturbance0)


#==================================选择动作======================================================
        a0 = agent_2.choose_action(s0, error0, disturbance0)

        a0_zero_error = agent_2.choose_action_zero(s0, error0_constant, disturbance0)
        a0_zero_q = agent_2.choose_action_zero(error0_constant, error0, disturbance0)
        a0_zero_alpha = agent_2.choose_action_zero(s0, error0, error0_constant)



#noise2 = 5 * np.exp(-0.0005 * k) * np.sin(0.01 * k)

        #a0 = a0 + noise2

        if a0[0] - a_1[0] > 0.6:
            a0[0] = a_1[0] + 0.6
        if a0[0] - a_1[0] < -0.6:
            a0[0] = a_1[0] - 0.6

        u0 = s0

#==================================数据记录2==========================================================
        state_history[k + i * 300] = np.concatenate((x0, s0))
        action_history[k + i * 300] = np.concatenate((u0, a0))
        alpha_ref_history[k + i * 300] = alpha0_ref
        q_ref_noise_history[k + i * 300] = q0_ref
        q_ref_filtered_history[k + i * 300] = q0_ref_filtered

        actions_zero_error_history[k + i * 300] = np.concatenate((q0_ref_zero_error,q0_ref_zero_alpha,q0_ref_zero_delta, a0_zero_error,a0_zero_q,a0_zero_alpha))


#==================================仿真环境更新======================================================
        x1, done, info, empty = env_aircraft_1.solver(t0, x0, u0, d0)  # 计算当前一步状态
        s1,done1,info1,empty1 = env_aircraft_2.solver(t0, s0, a0, disturbance0)  # 计算当前一步状态

        agent_1.incrementalmodel.update(x_1, u_1, d_1, x0, u0, d0, x1)  # [alpha_1, q_1, delta_1, alpha0, q0, delta0, alpha1]
        agent_2.incrementalmodel.update(s_1, a_1, disturbance_1, s0, a0, disturbance0, s1)  # [q_1, delta_1, alpha_1, q0, delta0, alpha0, q1]

        #compute V0 V1 Ve
        x_1_tensor = T.tensor(x_1, dtype=T.float, requires_grad=False).to(agent_1.critic.device)
        u_1_tensor = T.tensor(u_1, dtype=T.float, requires_grad=False).to(agent_1.critic.device)
        d_1_tensor = T.tensor(d_1, dtype=T.float, requires_grad=False).to(agent_1.critic.device)
        x0_tensor = T.tensor(x0, dtype=T.float, requires_grad=False).to(agent_1.critic.device)
        x0_aug = T.tensor(np.concatenate((x0, e0, d0)), dtype=T.float, requires_grad=False).to(agent_1.critic.device)
        x0_aug_critic = T.tensor(np.concatenate((x0, e0)), dtype=T.float, requires_grad=False).to(agent_1.critic.device)
        x1_ref_tensor = T.tensor(x1_ref, dtype=T.float, requires_grad=False).to(agent_1.critic.device)
        d0_tensor = T.tensor(d0, dtype=T.float, requires_grad=False).to(agent_1.critic.device)
        d1_tensor = T.tensor(d0, dtype=T.float, requires_grad=False).to(agent_1.critic.device)




        u0_tensor = agent_1.actor.forward(x0_aug)

        x1_tensor = agent_1.incrementalmodel.forward(x_1_tensor,u_1_tensor,d_1_tensor, x0_tensor,u0_tensor,d0_tensor)
        #x1_tensor = agent_1.incrementalmodel.forward(x_1_tensor, u_1_tensor, d_1_tensor, x0_tensor, u0_tensor, d0_tensor)
        e1_tensor = x1_tensor - x1_ref_tensor
        #x1_aug = T.cat([x1_tensor, e1_tensor, d1_tensor]).to(agent_1.critic.device)
        x1_aug_critic = T.cat([x1_tensor, e1_tensor]).to(agent_1.critic.device)

        V0 = agent_1.critic.forward(x0_aug_critic)
        V1 = agent_1.critic.forward(x1_aug_critic)
        Ve = V1-V0

        x1_tensor_system = T.tensor(x1)
        e1_tensor_system = x1_tensor_system - x1_ref_tensor
        x1_aug_system = T.cat([x1_tensor_system, e1_tensor_system]).to(agent_1.critic.device)

        V1_system = agent_1.critic.forward(x1_aug_system)
        Ve_system = V1_system-V0

        agent_1.store_transition(V0, V1, Ve, V0, V1_system, Ve_system)


        t1 = t0 + step
        k = k + 1

#==================================计算奖励==========================================================
        reward = -abs(x1[0]-x1_ref[0])
        score = score + reward

#=================================数据更新========================================================
        #a_5 = a_4; a_4 = a_3
        #a_3 = a_2; a_2 = a_1

        t0 = t1
        x_1 = x0; s_1 = s0
        u_1 = u0; a_1 = a0
        x0 = x1;  s0 = s1


        #xf0 = xf1

        q_1_ref = q0_ref

        score_history.append(score)
        average_score = np.mean(score_history[-100:])
        average_score_history.append(average_score)

        print('k',k)


if score > score0:
    score0 = score
    Index0 = i + 1
    print('save index', i + 1)
    print('episode', i + 1, 'score % .2f' % score,'100 game average %.2f' % np.mean(score_history[-100:]), 'maxscore', score0, 'no.', Index0)
#if i == 2999:
agent_1.save_models()
agent_2.save_models()

np.savetxt('state_history',state_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('action_history',action_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('alpha_ref_history',alpha_ref_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
#np.savetxt('q_ref_noise_history',q_ref_noise_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('q_ref_noise_history', q_ref_noise_history, delimiter=' ', newline='\n', header='', footer='', comments='#', encoding=None)
np.savetxt('q_ref_filtered_history', q_ref_filtered_history, delimiter=' ', newline='\n', header='', footer='', comments='#', encoding=None)
np.savetxt('Critic_1_Lay1_history',Critic_1_Lay1_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)
np.savetxt('Critic_1_Lay2_history',Critic_1_Lay2_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)
np.savetxt('Critic_2_Lay1_history',Critic_2_Lay1_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)
np.savetxt('Critic_2_Lay2_history',Critic_2_Lay2_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)
np.savetxt('Actor_1_Lay1_history',Actor_1_Lay1_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)
np.savetxt('Actor_1_Lay2_history',Actor_1_Lay2_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)
np.savetxt('Actor_2_Lay1_history',Actor_2_Lay1_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)
np.savetxt('Actor_2_Lay2_history',Actor_2_Lay2_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)
np.savetxt('Theta_1_history',Theta_1_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)
np.savetxt('Theta_2_history',Theta_2_history,delimiter=' ',newline='\n',header='',footer='',comments='#', encoding=None)

np.savetxt('agent_1_v_memory',agent_1.v_memory,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)

np.savetxt('Agent_1_Lay2_out_history',agent_1.lay2_out_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('Agent_2_Lay2_out_history',agent_2.lay2_out_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)

np.savetxt('Agent_1_grad_gain_history',agent_1.grad_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('Agent_2_grad_gain_history',agent_2.grad_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)

np.savetxt('Actor_1_Lay1_grad_history',agent_1.Actor_1_Lay1_grad_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('Actor_1_Lay2_grad_history',agent_1.Actor_1_Lay2_grad_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('Actor_2_Lay1_grad_history',agent_2.Actor_2_Lay1_grad_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('Actor_2_Lay2_grad_history',agent_2.Actor_2_Lay2_grad_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)

np.savetxt('agent_1_error_memory',agent_1.e_memory,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('agent_2_error_memory',agent_2.error_memory,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)

np.savetxt('agent_1_terminal_memory',agent_1.terminal_memory,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('agent_2_terminal_memory',agent_2.terminal_memory,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)

np.savetxt('actions_zero_error_history',actions_zero_error_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)


#one-order filter
# a = 0.3
# q0_ref = q_1_ref + a * (q0_ref_noise - q_1_ref)