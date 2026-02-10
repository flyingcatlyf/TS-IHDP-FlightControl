import numpy as np
import matplotlib.pyplot as plt
import matplotlib


state_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/state_history')
action_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/action_history')
alpha_ref_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/alpha_ref_history')
q_ref_noise_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/q_ref_noise_history')
q_ref_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/q_ref_filtered_history')
Critic_1_Lay1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Critic_1_Lay1_history')
Critic_1_Lay2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Critic_1_Lay2_history')
Critic_2_Lay1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Critic_2_Lay1_history')
Critic_2_Lay2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Critic_2_Lay2_history')
Actor_1_Lay1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Actor_1_Lay1_history')
Actor_1_Lay2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Actor_1_Lay2_history')
Actor_2_Lay1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Actor_2_Lay1_history')
Actor_2_Lay2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Actor_2_Lay2_history')
Theta_1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Theta_1_history')
Theta_2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Theta_2_history')

#terminal_agent1 = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/agent_1_terminal_memory')
#terminal_agent2 = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/agent_2_terminal_memory')


#Bias_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss/IHDP_nn/IHDP/Bias_history')

#F_reshape = F_matrix[:20000,:]
#G_reshape = G_matrix[:20000,:]
state_history_reshape = state_history[:40000,:]
action_history_reshape = action_history[:40000,]

alpha_ref_history = np.reshape(alpha_ref_history,(-1,1))
alpha_ref_history_reshape = alpha_ref_history[:40000,:]
q_ref_noise_history = np.reshape(q_ref_noise_history,(-1,1))
q_ref_noise_history_reshape = q_ref_noise_history[:40000,:]
q_ref_history = np.reshape(q_ref_history,(-1,1))
q_ref_history_reshape = q_ref_history[:40000,:]

#terminal_agent1 = np.reshape(terminal_agent1,(-1,1))
#terminal_agent1_reshape = terminal_agent1[:40000,:]
#terminal_agent2 = np.reshape(terminal_agent2,(-1,1))
#terminal_agent2_reshape = terminal_agent2[:40000,:]


#state_history_reshape_1 = np.reshape(state_history_reshape[:,0],(-1,1)) - q_ref_history_reshape

Critic_1_Lay1_history_reshape = Critic_1_Lay1_history[:40000,:]
Critic_1_Lay2_history_reshape = Critic_1_Lay2_history[:40000,:]
Critic_2_Lay1_history_reshape = Critic_2_Lay1_history[:40000,:]
Critic_2_Lay2_history_reshape = Critic_2_Lay1_history[:40000,:]
Actor_1_Lay1_history_reshape = Actor_1_Lay1_history[:40000,:]
Actor_1_Lay2_history_reshape = Actor_1_Lay2_history[:40000,:]
Actor_2_Lay1_history_reshape = Actor_2_Lay1_history[:40000,:]
Actor_2_Lay2_history_reshape = Actor_2_Lay1_history[:40000,:]

Theta_1_history_reshape = Theta_1_history[:40000,:]
Theta_2_history_reshape = Theta_2_history[:40000,:]


#Bias_history_reshape = Bias_history[:20000,:]

time = np.array(range(0,40000,1))

#fig1 = plt.figure(figsize=(18.0,9.0))
#matplotlib.rcParams['pdf.fonttype'] = 42
#plt.plot(F_reshape[:,0],linewidth=1.0,color = 'C0',label = 'F11')
#plt.plot(F_reshape[:,1],linewidth=1.0,color='C1',label = 'F12')
#plt.plot(F_reshape[:,2],linewidth=1.0,color = 'C2',label = 'F13')
#plt.plot(F_reshape[:,3],linewidth=1.0,color='C3',label = 'F14')
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel(r'$F$',fontdict={'size':35})
#plt.xticks(fontsize=35)
#plt.yticks(fontsize=35)
#plt.title('F Matrix')
#plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)
#plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

#fig2 = plt.figure(figsize=(18.0,9.0))
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype']  = 42
##plt.plot(G_reshape[:,1],linewidth=1.0,color='C1')
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel(r'$G$',fontdict={'size':35})
#plt.xticks(fontsize=35)
#plt.yticks(fontsize=35)
#plt.title('G Matrix')
#plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)
#plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0', '5', '10', '15', '20', '25', '30'])

fig1 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(state_history_reshape[:,0],linewidth=1.0,color = 'C0',label=r'$\alpha$')
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{ref}$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\alpha Tracking$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig2 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(state_history_reshape[:,1],linewidth=1.0,color = 'C0',label='q')
plt.plot(q_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref}$')
plt.plot(q_ref_noise_history_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q^{noise}_{ref}$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$pitch rate q$ [deg/s]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Pitch rate')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig3 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
#plt.plot(action_history_reshape[:,0],linewidth=1.0,color = 'C0',label=r'$q$')
plt.plot(action_history_reshape[:,1],linewidth=1.0,color = 'C0',label=r'$\delta$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title(r'$\delta history$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig4 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Critic_1_Lay1_history_reshape[:,0],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,1],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,2],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,3],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,4],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,5],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,6],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,7],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,8],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,9],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,10],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,11],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,12],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay1_history_reshape[:,13],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,14],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,15],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,16],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,17],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,18],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,19],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,20],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,21],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,22],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,23],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,24],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,25],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,26],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,27],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,28],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay1_history_reshape[:,29],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Critic_1_Lay1_history',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Critic_1_Lay1_history')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig5 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Critic_1_Lay2_history_reshape[:,0],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay2_history_reshape[:,1],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay2_history_reshape[:,2],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay2_history_reshape[:,3],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay2_history_reshape[:,4],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay2_history_reshape[:,5],linewidth=1.0,color = 'C0')
plt.plot(Critic_1_Lay2_history_reshape[:,6],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay2_history_reshape[:,7],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay2_history_reshape[:,8],linewidth=1.0,color = 'C0')
#plt.plot(Critic_1_Lay2_history_reshape[:,9],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Critic_1_Lay2_history',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Critic_1_Lay1_Weights')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig6 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Critic_2_Lay1_history_reshape[:,0],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,1],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,2],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,3],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,4],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,5],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,6],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,7],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,8],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,9],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,10],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,11],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,12],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay1_history_reshape[:,13],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,14],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,15],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,16],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,17],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,18],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,19],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,20],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,21],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,22],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,23],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,24],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,25],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,26],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,27],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,28],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay1_history_reshape[:,29],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Critic_2_Lay1_history',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Critic_2_Lay1_history')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig7= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Critic_2_Lay2_history_reshape[:,0],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay2_history_reshape[:,1],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay2_history_reshape[:,2],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay2_history_reshape[:,3],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay2_history_reshape[:,4],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay2_history_reshape[:,5],linewidth=1.0,color = 'C0')
plt.plot(Critic_2_Lay2_history_reshape[:,6],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay2_history_reshape[:,7],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay2_history_reshape[:,8],linewidth=1.0,color = 'C0')
#plt.plot(Critic_2_Lay2_history_reshape[:,9],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Critic_2_Lay2_history',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Critic_2_Lay1_Weights')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig8= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Actor_1_Lay1_history_reshape[:,0],linewidth=1.0,color = 'C0',label = r'$\alpha$')
plt.plot(Actor_1_Lay1_history_reshape[:,1],linewidth=1.0,color = 'C1',label = r'$e_{\alpha}$')
plt.plot(Actor_1_Lay1_history_reshape[:,2],linewidth=1.0,color = 'C2',label = r'\delta_{e}')
plt.plot(Actor_1_Lay1_history_reshape[:,3],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay1_history_reshape[:,4],linewidth=1.0,color = 'C1')
plt.plot(Actor_1_Lay1_history_reshape[:,5],linewidth=1.0,color = 'C2')
plt.plot(Actor_1_Lay1_history_reshape[:,6],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay1_history_reshape[:,7],linewidth=1.0,color = 'C1')
plt.plot(Actor_1_Lay1_history_reshape[:,8],linewidth=1.0,color = 'C2')
plt.plot(Actor_1_Lay1_history_reshape[:,9],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay1_history_reshape[:,10],linewidth=1.0,color = 'C1')
plt.plot(Actor_1_Lay1_history_reshape[:,11],linewidth=1.0,color = 'C2')
plt.plot(Actor_1_Lay1_history_reshape[:,12],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay1_history_reshape[:,13],linewidth=1.0,color = 'C1')
plt.plot(Actor_1_Lay1_history_reshape[:,14],linewidth=1.0,color = 'C2')
plt.plot(Actor_1_Lay1_history_reshape[:,15],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay1_history_reshape[:,16],linewidth=1.0,color = 'C1')
plt.plot(Actor_1_Lay1_history_reshape[:,17],linewidth=1.0,color = 'C2')
plt.plot(Actor_1_Lay1_history_reshape[:,18],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay1_history_reshape[:,19],linewidth=1.0,color = 'C1')
plt.plot(Actor_1_Lay1_history_reshape[:,20],linewidth=1.0,color = 'C2')
#plt.plot(Actor_1_Lay1_history_reshape[:,21],linewidth=1.0,color = 'C0')
#plt.plot(Actor_1_Lay1_history_reshape[:,22],linewidth=1.0,color = 'C1')
#plt.plot(Actor_1_Lay1_history_reshape[:,23],linewidth=1.0,color = 'C2')
#plt.plot(Actor_1_Lay1_history_reshape[:,24],linewidth=1.0,color = 'C0')
#plt.plot(Actor_1_Lay1_history_reshape[:,25],linewidth=1.0,color = 'C1')
#plt.plot(Actor_1_Lay1_history_reshape[:,26],linewidth=1.0,color = 'C2')
#plt.plot(Actor_1_Lay1_history_reshape[:,27],linewidth=1.0,color = 'C0')
#plt.plot(Actor_1_Lay1_history_reshape[:,28],linewidth=1.0,color = 'C1')
#plt.plot(Actor_1_Lay1_history_reshape[:,29],linewidth=1.0,color = 'C2')
idx = 2000
y_mark = Actor_1_Lay1_history_reshape[2000,0]
plt.scatter(idx, y_mark, s = 50)
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Actor_1_Lay1_history',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Actor_1_Lay1_history')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig20= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Actor_1_Lay1_history_reshape[:,0],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,1],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,2],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,3],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,4],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,5],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,6],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,7],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,8],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,9],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,10],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,11],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,12],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,13],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,14],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,15],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,16],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,17],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,18],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,19],linewidth=2.0)
plt.plot(Actor_1_Lay1_history_reshape[:,20],linewidth=2.0)
#plt.plot(Actor_1_Lay1_history_reshape[:,21],linewidth=2.0)
#plt.plot(Actor_1_Lay1_history_reshape[:,22],linewidth=2.0)
#plt.plot(Actor_1_Lay1_history_reshape[:,23],linewidth=2.0)
#plt.plot(Actor_1_Lay1_history_reshape[:,24],linewidth=2.0)
#plt.plot(Actor_1_Lay1_history_reshape[:,25],linewidth=2.0)
#plt.plot(Actor_1_Lay1_history_reshape[:,26],linewidth=2.0)
#plt.plot(Actor_1_Lay1_history_reshape[:,27],linewidth=2.0)
#plt.plot(Actor_1_Lay1_history_reshape[:,28],linewidth=2.0)
#plt.plot(Actor_1_Lay1_history_reshape[:,29],linewidth=2.0)
idx = 2000
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,0], s = 50, marker = 'o', label = r'$\alpha$')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,1], s = 50, marker = '^', label = r'$e_{\alpha}$')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,2], s = 50, marker = 'D', label = r'$\delta_{e}$')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,3], s = 50, marker = 'o')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,4], s = 50, marker = '^')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,5], s = 50)
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,6], s = 50, marker = 'o')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,7], s = 50, marker = '^')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,8], s = 50)
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,9], s = 50, marker = 'o')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,10], s = 50, marker = '^')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,11], s = 50)
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,12], s = 50, marker = 'o')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,13], s = 50, marker = '^')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,14], s = 50)
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,15], s = 50, marker = 'o')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,16], s = 50, marker = '^')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,17], s = 50)
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,18], s = 50, marker = 'o')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,19], s = 50, marker = '^')
plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,20], s = 50)
#plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,21], s = 50, marker = 'o')
#plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,22], s = 50, marker = '^')
#plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,23], s = 50)
#plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,24], s = 50, marker = 'o')
#plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,25], s = 50, marker = '^')
#plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,26], s = 50)
#plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,27], s = 50, marker = 'o')
#plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,28], s = 50, marker = '^')
#plt.scatter(idx, Actor_1_Lay1_history_reshape[idx,29], s = 50)
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Actor_1_Lay1_history',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Actor_1_Lay1_history')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig9 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Actor_1_Lay2_history_reshape[:,0],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay2_history_reshape[:,1],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay2_history_reshape[:,2],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay2_history_reshape[:,3],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay2_history_reshape[:,4],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay2_history_reshape[:,5],linewidth=1.0,color = 'C0')
plt.plot(Actor_1_Lay2_history_reshape[:,6],linewidth=1.0,color = 'C0')
#plt.plot(Actor_1_Lay2_history_reshape[:,7],linewidth=1.0,color = 'C0')
#plt.plot(Actor_1_Lay2_history_reshape[:,8],linewidth=1.0,color = 'C0')
#plt.plot(Actor_1_Lay2_history_reshape[:,9],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Actor_1_Lay2_history',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Actor_1_Lay2_history')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig10= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Actor_2_Lay1_history_reshape[:,0],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,1],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,2],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,3],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,4],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,5],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,6],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,7],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,8],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,9],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,10],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,11],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,12],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,13],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,14],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,15],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,16],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,17],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,18],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,19],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay1_history_reshape[:,20],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay1_history_reshape[:,21],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay1_history_reshape[:,22],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay1_history_reshape[:,23],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay1_history_reshape[:,24],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay1_history_reshape[:,25],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay1_history_reshape[:,26],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay1_history_reshape[:,27],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay1_history_reshape[:,28],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay1_history_reshape[:,29],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Actor_2_Lay1_history',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Actor_2_Lay1_history')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)


fig11= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Actor_2_Lay2_history_reshape[:,0],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay2_history_reshape[:,1],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay2_history_reshape[:,2],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay2_history_reshape[:,3],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay2_history_reshape[:,4],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay2_history_reshape[:,5],linewidth=1.0,color = 'C0')
plt.plot(Actor_2_Lay2_history_reshape[:,6],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay2_history_reshape[:,7],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay2_history_reshape[:,8],linewidth=1.0,color = 'C0')
#plt.plot(Actor_2_Lay2_history_reshape[:,9],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Actor_2_Lay2_history',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Actor_2_Lay2_history')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig13= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Theta_1_history_reshape[:,0],linewidth=1.0,color = 'C0',label=r'$\frac{\partial \alpha}{\partial \alpha}$')
plt.plot(Theta_1_history_reshape[:,1],linewidth=1.0,color = 'C1',label=r'$\frac{\partial \alpha}{\partial q}$')
plt.plot(Theta_2_history_reshape[:,0],linewidth=1.0,color = 'C2',label=r'$\frac{\partial q}{\partial q}$')
plt.plot(Theta_2_history_reshape[:,2],linewidth=1.0,color = 'C3',label=r'$\frac{\partial q}{\partial \alpha}$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$F$',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title(r'$F$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig14= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(Theta_1_history_reshape[:,2],linewidth=1.0,color = 'C0',label=r'$\frac{\partial \alpha}{\partial \delta}$')
plt.plot(Theta_2_history_reshape[:,1],linewidth=1.0,color = 'C1',label=r'$\frac{\partial q}{\partial \delta}$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$G$',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title(r'$G$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig15= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
ax = plt.axes(projection='3d')
ax.scatter(np.reshape(state_history_reshape[:,0],(-1,1))-alpha_ref_history_reshape, np.reshape(state_history_reshape[:,0],(-1,1)), q_ref_history_reshape, c=q_ref_history_reshape)
ax.set_xlabel(r'$\alpha_{ref}$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'$q_{ref}$')
#ax.xticks(fontsize=35)
#ax.yticks(fontsize=35)
#ax.title('Actor_2_Lay2_history')
ax.grid(True)

fig16= plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
ax = plt.axes(projection='3d')
ax.plot_wireframe(np.reshape(state_history_reshape[:,0],(-1,1))-alpha_ref_history_reshape, np.reshape(state_history_reshape[:,0],(-1,1)), q_ref_history_reshape)
ax.set_xlabel(r'$\alpha_{ref}$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'$q_{ref}$')
#ax.xticks(fontsize=35)
#ax.yticks(fontsize=35)
#ax.title('Actor_2_Lay2_history')
ax.grid(True)

fig17 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(state_history_reshape[:,1],linewidth=1.0,color = 'C0',label='q')
plt.plot(q_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref}$')
#plt.plot(q_ref_noise_history_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q^{noise}_{ref}$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$pitch rate q$ [deg/s]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Pitch rate')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

a0 = np.sum(Actor_1_Lay1_history_reshape[:,0])
a1 = np.sum(Actor_1_Lay1_history_reshape[:,1])
a2 = np.sum(Actor_1_Lay1_history_reshape[:,2])
a3 = np.sum(Actor_1_Lay1_history_reshape[:,3])
a4 = np.sum(Actor_1_Lay1_history_reshape[:,4])
a5 = np.sum(Actor_1_Lay1_history_reshape[:,5])
a6 = np.sum(Actor_1_Lay1_history_reshape[:,6])
a7 = np.sum(Actor_1_Lay1_history_reshape[:,7])
a8 = np.sum(Actor_1_Lay1_history_reshape[:,8])
a9 = np.sum(Actor_1_Lay1_history_reshape[:,9])
a10 = np.sum(Actor_1_Lay1_history_reshape[:,10])
a11 = np.sum(Actor_1_Lay1_history_reshape[:,11])
a12 = np.sum(Actor_1_Lay1_history_reshape[:,12])
a13 = np.sum(Actor_1_Lay1_history_reshape[:,13])
a14 = np.sum(Actor_1_Lay1_history_reshape[:,14])
a15 = np.sum(Actor_1_Lay1_history_reshape[:,15])
a16 = np.sum(Actor_1_Lay1_history_reshape[:,16])
a17 = np.sum(Actor_1_Lay1_history_reshape[:,17])
a18 = np.sum(Actor_1_Lay1_history_reshape[:,18])
a19 = np.sum(Actor_1_Lay1_history_reshape[:,19])
a20 = np.sum(Actor_1_Lay1_history_reshape[:,20])

#x = [0]
#x=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
x = np.array(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
y = np.array([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20])
fig18 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.bar(x, y)



#fig18 = plt.figure(figsize=(18.0,9.0))
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype']  = 42
#plt.plot(terminal_agent1_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref}$')
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel(r'$terminal$',fontdict={'size':35})
#plt.xticks(fontsize=35)
#plt.yticks(fontsize=35)
#plt.title('Terminal')
#plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

#fig19 = plt.figure(figsize=(18.0,9.0))
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype']  = 42
#plt.plot(terminal_agent2_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref}$')
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel(r'$terminal2$',fontdict={'size':35})
#plt.xticks(fontsize=35)
#plt.yticks(fontsize=35)
#plt.title('Terminal')
#plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

#fig14 = plt.figure(figsize=(18.0,9.0))
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype']  = 42
#plt.plot(Bias_history_reshape[:,0],linewidth=1.0,color = 'C0')
#plt.plot(Bias_history_reshape[:,1],linewidth=1.0,color = 'C1')
#plt.plot(Bias_history_reshape[:,2],linewidth=1.0,color = 'C2')
#plt.plot(Bias_history_reshape[:,3],linewidth=1.0,color = 'C3')
#plt.plot(Bias_history_reshape[:,4],linewidth=1.0,color = 'C4')
#plt.plot(Bias_history_reshape[:,5],linewidth=1.0,color = 'C5')
#plt.plot(Bias_history_reshape[:,6],linewidth=1.0,color = 'C6')
#plt.plot(Bias_history_reshape[:,7],linewidth=1.0,color = 'C7')
#plt.plot(Bias_history_reshape[:,8],linewidth=1.0,color = 'C8')
#plt.plot(Bias_history_reshape[:,9],linewidth=1.0,color = 'C9')
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel(r'$Actor Bias Weights$',fontdict={'size':35})
#plt.xticks(fontsize=35)
#plt.yticks(fontsize=35)
#plt.title('Actor Bias Weights')
#plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)


#index = np.array(range(0,40000,50))
#alpha_history = np.reshape(state_history_reshape[:,0],(-1,1))
#alpha_history = alpha_history[index]

#alphar_history = alpha_ref_history_reshape
#alphar_history = alphar_history[index]

#qr_history = q_ref_history_reshape[index]
#fig14= plt.figure(figsize=(18.0,9.0))
#fig14 = plt.figure()
#ax = plt.axes(projection="3d")
#x = alpha_history
#y = alphar_history
#z = qr_history
#X, Y = np.meshgrid(x, y)
#ax.plot_surface(X, Y, z, alpha=0.9, cstride=1, rstride=1, cmap='rainbow')
#ax.set_xlabel(r'$\alpha$')
#ax.set_ylabel(r'$\alpha_{ref}$')
#ax.set_zlabel(r'qref')

plt.show()
