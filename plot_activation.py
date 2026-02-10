import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#Activation Function
#Vanilla
Agent_1_Lay2_out_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Agent_1_Lay2_out_history')
Agent_2_Lay2_out_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Agent_2_Lay2_out_history')

#TS
Agent_1_Lay2_out_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Agent_1_Lay2_out_history')
Agent_2_Lay2_out_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Agent_2_Lay2_out_history')

#Filterd
Agent_1_Lay2_out_history_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter2/Agent_1_Lay2_out_history')
Agent_2_Lay2_out_history_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter2/Agent_2_Lay2_out_history')
#state_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/state_history')
#action_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/action_history')
#alpha_ref_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/alpha_ref_history')
#q_ref_noise_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/q_ref_noise_history')
#q_ref_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/q_ref_filtered_history')
#Critic_1_Lay1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Critic_1_Lay1_history')
#Critic_1_Lay2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Critic_1_Lay2_history')
#Critic_2_Lay1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Critic_2_Lay1_history')
#Critic_2_Lay2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Critic_2_Lay2_history')
#Actor_1_Lay1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Actor_1_Lay1_history')
#Actor_1_Lay2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Actor_1_Lay2_history')
#Actor_2_Lay1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Actor_2_Lay1_history')
#Actor_2_Lay2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Actor_2_Lay2_history')
#Theta_1_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Theta_1_history')
#Theta_2_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/IHDP_nn/IHDP/Theta_2_history')


#Bias_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss/IHDP_nn/IHDP/Bias_history')

#F_reshape = F_matrix[:20000,:]
#G_reshape = G_matrix[:20000,:]

#vanilla
Agent_1_Lay2_out_history_vanilla = np.reshape(Agent_1_Lay2_out_history_vanilla,(-1,1))
Agent_1_Lay2_out_history_reshape_vanilla = Agent_1_Lay2_out_history_vanilla[:40000,:]

Agent_2_Lay2_out_history_vanilla = np.reshape(Agent_2_Lay2_out_history_vanilla,(-1,1))
Agent_2_Lay2_out_history_reshape_vanilla = Agent_2_Lay2_out_history_vanilla[:40000,]

x = Agent_1_Lay2_out_history_vanilla
y = np.tanh(x)

#TS
Agent_1_Lay2_out_history_TS = np.reshape(Agent_1_Lay2_out_history_TS,(-1,1))
Agent_1_Lay2_out_history_reshape_TS = Agent_1_Lay2_out_history_TS[:40000,:]

Agent_2_Lay2_out_history_TS = np.reshape(Agent_2_Lay2_out_history_TS,(-1,1))
Agent_2_Lay2_out_history_reshape_TS = Agent_2_Lay2_out_history_TS[:40000,]

#filter
Agent_1_Lay2_out_history_filter = np.reshape(Agent_1_Lay2_out_history_filter,(-1,1))
Agent_1_Lay2_out_history_reshape_filter = Agent_1_Lay2_out_history_filter[:40000,:]

Agent_2_Lay2_out_history_filter = np.reshape(Agent_2_Lay2_out_history_filter,(-1,1))
Agent_2_Lay2_out_history_reshape_filter = Agent_2_Lay2_out_history_filter[:40000,]

time = np.array(range(-40000,40000,2))

#x_tanh = np.arange(0, 4, 0.0001)

x_tanh = 0.0001 * time
y_tanh = np.tanh(x_tanh)
z_tanh = 1 - np.tanh(x_tanh)**2




#plot
fig1 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(np.abs(Agent_1_Lay2_out_history_reshape_vanilla),linewidth=1.0,color = 'C0',label = 'vanilla')
plt.plot(np.abs(Agent_1_Lay2_out_history_reshape_TS),linewidth=1.0,color = 'C1',label = 'TS')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Agent_1_Lay2_out',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Agent_1_Lay2_out')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig2 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(np.abs(Agent_2_Lay2_out_history_reshape_vanilla),linewidth=1.0,color = 'C0',label = 'vanilla')
plt.plot(np.abs(Agent_2_Lay2_out_history_reshape_TS),linewidth=1.0,color = 'C1',label = 'TS')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Agent_2_Lay2_out')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig3 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'vanilla')
plt.scatter(x,y,alpha=0.1)
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Agent_2_Lay2_out')
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig4 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

plt.subplot(2,1,1)
plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'vanilla')
#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Agent_2_Lay2_out')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(Agent_1_Lay2_out_history_reshape_vanilla,time,linewidth=1.0,color = 'C1',linestyle='-',label = 'vanilla')
plt.plot(Agent_1_Lay2_out_history_reshape_TS,time,linewidth=1.0,color = 'C0',label = 'TS')
#plt.scatter(x,y,alpha=0.1)
plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title('Agent_2_Lay2_out')
plt.grid(True)
plt.yticks([-40000, -20000, 0,  20000,  40000], ['0', '10', '20', '30', '40'])

fig5 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

plt.subplot(2,1,1)
plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'vanilla')
#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Agent_2_Lay2_out')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(Agent_2_Lay2_out_history_reshape_vanilla,time,linewidth=1.0,color = 'C1',linestyle='-',label = 'vanilla')
plt.plot(Agent_2_Lay2_out_history_reshape_TS,time,linewidth=1.0,color = 'C0',label = 'TS')
#plt.scatter(x,y,alpha=0.1)
plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title('Agent_2_Lay2_out')
plt.grid(True)
plt.yticks([-40000, -20000, 0,  20000,  40000], ['0', '10', '20', '30', '40'])


fig6 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

plt.subplot(3,2,1)
plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
#plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')
#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':10})
#plt.xlabel('tanh input',fontdict={'size':10})
plt.ylabel('tanh output',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Outer-loop actor',fontsize=18)
plt.grid(True)

plt.subplot(3,2,2)
plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
#plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
#plt.xlabel('tanh input',fontdict={'size':10})
plt.ylabel('tanh output',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Inner-loop actor',fontsize=18)
plt.grid(True)

plt.subplot(3,2,3)
#plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')
#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':10})
#plt.xlabel('tanh input',fontdict={'size':10})
plt.ylabel('tanh derivative',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.title('Actor 1')
plt.grid(True)

plt.subplot(3,2,4)
#plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
#plt.xlabel('tanh input',fontdict={'size':10})
#plt.ylabel('tanh output',fontdict={'size':10})
plt.ylabel('tanh derivative',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
x_min, x_max = plt.xlim()
#plt.title('Actor 2')
plt.grid(True)

plt.subplot(3,2,5)
plt.plot(Agent_1_Lay2_out_history_reshape_vanilla,time,linewidth=1.0,color = 'C1',linestyle='-',label = 'Baseline')
plt.plot(Agent_1_Lay2_out_history_reshape_TS,time,linewidth=1.0,color = 'C0',label = 'Temporally smoothed',linestyle='-')
#plt.scatter(x,y,alpha=0.1)
plt.ylabel('Time [s]',fontsize=18)
plt.xlabel('tanh input',fontsize=18)
plt.xticks(fontsize=18)
plt.xlim([x_min,x_max])
#plt.title('Agent_2_Lay2_out')
plt.grid(True)
plt.yticks([-40000, -20000, 0,  20000,  40000], ['0', '10', '20', '30', '40'],fontsize=18)


plt.subplot(3,2,6)
plt.plot(Agent_2_Lay2_out_history_reshape_vanilla,time,linewidth=1.0,color = 'C1',linestyle='-',label = 'Baseline')
plt.plot(Agent_2_Lay2_out_history_reshape_TS,time,linewidth=1.0,color = 'C0',label = 'Temporally smoothed',linestyle='-')
#plt.scatter(x,y,alpha=0.1)
plt.xlabel('tanh input',fontsize=18)
plt.ylabel('Time [s]',fontsize=18)
plt.xticks(fontsize=18)
plt.xlim([x_min,x_max])
#plt.title('Agent_2_Lay2_out')
plt.grid(True)
plt.yticks([-40000, -20000, 0,  20000,  40000], ['0', '10', '20', '30', '40'],fontsize=18)
plt.legend(loc='lower right',fontsize=16)

plt.tight_layout()
plt.savefig('Tanh_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Tanh_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)


fig7 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.subplot(3,2,1)
plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
#plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')
#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':10})
#plt.xlabel('tanh input',fontdict={'size':10})
plt.ylabel('tanh output',fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.title('Outer-loop actor',fontsize=19)
plt.grid(True)

plt.subplot(3,2,2)
plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
#plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
#plt.xlabel('tanh input',fontdict={'size':10})
plt.ylabel('tanh output',fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.title('Inner-loop actor',fontsize=19)
plt.grid(True)

plt.subplot(3,2,3)
#plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')
#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':10})
#plt.xlabel('tanh input',fontdict={'size':10})
plt.ylabel('tanh derivative',fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
#plt.title('Actor 1')
plt.grid(True)

plt.subplot(3,2,4)
#plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
#plt.xlabel('tanh input',fontdict={'size':10})
#plt.ylabel('tanh output',fontdict={'size':10})
plt.ylabel('tanh derivative',fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
x_min, x_max = plt.xlim()
#plt.title('Actor 2')
plt.grid(True)

plt.subplot(3,2,5)
plt.plot(Agent_1_Lay2_out_history_reshape_TS,time,linewidth=1.0,color = 'C0',linestyle='-',label = 'Temporally smoothed')
plt.plot(Agent_1_Lay2_out_history_reshape_filter,time,linewidth=1.0,color = 'C3',linestyle='--',label = 'Temporally smoothed and filtered')
#plt.scatter(x,y,alpha=0.1)
plt.ylabel('Time [s]',fontsize=19)
plt.xlabel('tanh input',fontsize=19)
plt.xticks(fontsize=19)
plt.xlim([x_min,x_max])
#plt.title('Agent_2_Lay2_out')
plt.grid(True)
plt.yticks([-40000, -20000, 0,  20000,  40000], ['0', '10', '20', '30', '40'],fontsize=19)


plt.subplot(3,2,6)
plt.plot(Agent_2_Lay2_out_history_reshape_TS,time,linewidth=1.0,color = 'C0',linestyle='-',label = 'Temporally smoothed')
plt.plot(Agent_2_Lay2_out_history_reshape_filter,time,linewidth=1.0,color = 'C3',linestyle='--',label = 'Temporally smoothed and filtered')
#plt.scatter(x,y,alpha=0.1)
plt.ylabel('Time [s]',fontsize=19)
plt.xlabel('tanh input',fontsize=19)
plt.xticks(fontsize=19)
plt.xlim([x_min,x_max])
#plt.title('Agent_2_Lay2_out')
plt.grid(True)
plt.yticks([-40000, -20000, 0,  20000,  40000], ['0', '10', '20', '30', '40'],fontsize=19)
plt.legend(loc='lower right',fontsize=12)

plt.tight_layout()
plt.savefig('Tanh_comparison_filter.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Tanh_comparison_filter.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)




fig8 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.subplot(3,2,1)
plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
plt.ylabel('tanh output',fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.title('Outer-loop actor',fontsize=19)
plt.grid(True)

plt.subplot(3,2,2)
plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
#plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
#plt.xlabel('tanh input',fontdict={'size':10})
plt.ylabel('tanh output',fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.title('Inner-loop actor',fontsize=19)
plt.grid(True)

plt.subplot(3,2,3)
#plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')
#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':10})
#plt.xlabel('tanh input',fontdict={'size':10})
plt.ylabel('tanh derivative',fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
#plt.title('Actor 1')
plt.grid(True)

plt.subplot(3,2,4)
#plt.plot(x_tanh,y_tanh,linewidth=1.0,color = 'C0',label = 'tanh')
plt.plot(x_tanh,z_tanh,linewidth=1.0,color = 'C0',label = 'dtanh')#plt.scatter(x,y,alpha=0.1)
#plt.xlabel('Time [s]',fontdict={'size':35})
#plt.ylabel('Agent_2_Lay2_out',fontdict={'size':35})
#plt.xlabel('tanh input',fontdict={'size':10})
#plt.ylabel('tanh output',fontdict={'size':10})
plt.ylabel('tanh derivative',fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
x_min, x_max = plt.xlim()
#plt.title('Actor 2')
plt.grid(True)

plt.subplot(3,2,5)
plt.plot(Agent_1_Lay2_out_history_reshape_vanilla,time,linewidth=0.01,color = 'C1',linestyle='-',label = 'IHDP')
plt.plot(Agent_1_Lay2_out_history_reshape_TS,time,linewidth=1.0,color = 'C0',linestyle='-',label = 'TS-IHDP')
plt.plot(Agent_1_Lay2_out_history_reshape_filter,time,linewidth=1.0,color = 'C3',linestyle='--',label = 'Command-filtered IHDP')
#plt.scatter(x,y,alpha=0.1)
plt.ylabel('Time [s]',fontsize=19)
plt.xlabel('tanh input',fontsize=19)
plt.xticks(fontsize=19)
plt.xlim([x_min,x_max])
#plt.title('Agent_2_Lay2_out')
plt.grid(True)
plt.yticks([-40000, -20000, 0,  20000,  40000], ['0', '10', '20', '30', '40'],fontsize=19)


plt.subplot(3,2,6)
plt.plot(Agent_2_Lay2_out_history_reshape_vanilla,time,linewidth=0.01,color = 'C1',linestyle='-',label = 'IHDP')
plt.plot(Agent_2_Lay2_out_history_reshape_TS,time,linewidth=1.0,color = 'C0',linestyle='-',label = 'TS-IHDP')
plt.plot(Agent_2_Lay2_out_history_reshape_filter,time,linewidth=1.0,color = 'C3',linestyle='--',label = 'Command-filtered IHDP')
#plt.scatter(x,y,alpha=0.1)
plt.ylabel('Time [s]',fontsize=19)
plt.xlabel('tanh input',fontsize=19)
plt.xticks(fontsize=19)
plt.xlim([x_min,x_max])
#plt.title('Agent_2_Lay2_out')
plt.grid(True)
plt.yticks([-40000, -20000, 0,  20000,  40000], ['0', '10', '20', '30', '40'],fontsize=19)
plt.legend(loc='lower right',fontsize=12)

plt.tight_layout()
plt.savefig('Tanh_comparison_smooth_filter.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Tanh_comparison_smooth_filter.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.show()