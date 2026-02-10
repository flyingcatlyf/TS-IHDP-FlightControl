import numpy as np
import matplotlib.pyplot as plt
import matplotlib

agent_1_grad_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Agent_1_grad__history')
agent_2_grad_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Agent_2_grad__history')

#agent_1_grad_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion3/Agent_1_grad__history')
#agent_2_grad_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion3/Agent_2_grad__history')

agent_1_grad_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Agent_1_grad_gain_history')
agent_2_grad_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Agent_2_grad_gain_history')

agent_1_grad_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter2/Agent_1_grad_gain_history')
agent_2_grad_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter2/Agent_2_grad_gain_history')

agent_1_grad_reshape_vanilla = agent_1_grad_vanilla[:40000,:]
agent_2_grad_reshape_vanilla = agent_2_grad_vanilla[:40000,:]

agent_1_grad_reshape_TS = agent_1_grad_TS[:40000,:]
agent_2_grad_reshape_TS = agent_2_grad_TS[:40000,:]

agent_1_grad_reshape_filter = agent_1_grad_filter[:40000,:]
agent_2_grad_reshape_filter = agent_2_grad_filter[:40000,:]

fig1 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.subplot(2,1,1)
plt.plot(agent_1_grad_reshape_vanilla[:,0],linewidth=0.3,color = 'C0',linestyle=':',label='Vanilla')
plt.plot(agent_1_grad_reshape_TS[:,0],linewidth=1.0,color = 'C2',linestyle='-',label='TS')
#plt.plot(reward_vanilla_agent2,linewidth=1.0,color = 'C0',linestyle='--',label='Vanilla')
#plt.xlabel('Time [s]',fontdict={'size':20})
#plt.ylabel(r'$\frac{\partial q_{\mathrm{ref (Actor)}}}{\partial e_{\alpha}}$',fontdict={'size':20})
plt.ylabel(r'$K_{1}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Actor 1',fontsize=20)
plt.grid(True)
plt.legend(loc='lower left',fontsize=20)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.subplot(2,1,2)
plt.plot(agent_2_grad_reshape_vanilla[:,0],linewidth=0.3,color = 'C0',linestyle=':',label='Vanilla')
plt.plot(agent_2_grad_reshape_TS[:,0],linewidth=1.0,color = 'C2',linestyle='-',label='TS')
#plt.plot(reward_vanilla_agent2,linewidth=1.0,color = 'C0',linestyle='--',label='Vanilla')
plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$K_{2}$',fontdict={'size':20})
#plt.ylabel(r'$\frac{\partial \delta}{\partial e_{q}}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Actor 2',fontsize=20)
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.tight_layout()
plt.savefig('Gain_comparison_TS.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Gain_comparison_TS.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

fig2 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.subplot(2,1,1)
plt.plot(agent_1_grad_reshape_filter[:,0],linewidth=1.0,color = 'C3',linestyle='-',label='TS+Filter')
plt.plot(agent_1_grad_reshape_TS[:,0],linewidth=1.0,color = 'C2',linestyle='--',label='TS')
#plt.plot(reward_vanilla_agent2,linewidth=1.0,color = 'C0',linestyle='--',label='Vanilla')
#plt.xlabel('Time [s]',fontdict={'size':20})
#plt.ylabel(r'$\frac{\partial q_{\mathrm{ref (Actor)}}}{\partial e_{\alpha}}$',fontdict={'size':20})
plt.ylabel(r'$K_{1}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Actor 1',fontsize=20)
plt.grid(True)
plt.legend(loc='lower left',fontsize=20)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.subplot(2,1,2)
plt.plot(agent_2_grad_reshape_filter[:,0],linewidth=1.0,color = 'C3',linestyle='-',label='TS+Filter')
plt.plot(agent_2_grad_reshape_TS[:,0],linewidth=1.0,color = 'C2',linestyle='--',label='TS')
#plt.plot(reward_vanilla_agent2,linewidth=1.0,color = 'C0',linestyle='--',label='Vanilla')
plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$K_{2}$',fontdict={'size':20})
#plt.ylabel(r'$\frac{\partial \delta}{\partial e_{q}}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Actor 2',fontsize=20)
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.tight_layout()
plt.savefig('Gain_comparison_Filter.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Gain_comparison_Filter.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.show()