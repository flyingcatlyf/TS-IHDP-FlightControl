import numpy as np
import matplotlib.pyplot as plt
import matplotlib


state_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter/state_history')
action_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter/action_history')
#q_ref_noise_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter/q_ref_noise_history')
q_ref_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter/q_ref_filtered_history')

state_smooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/state_history')
action_smooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/action_history')
#q_ref_noise_smooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/q_ref_noise_history')
q_ref_smooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/q_ref_filtered_history')

state_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/state_history')
action_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/action_history')
#q_ref_noise_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/q_ref_noise_history')
q_ref_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/q_ref_filtered_history')

alpha_ref_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/alpha_ref_history')
alpha_ref = np.reshape(alpha_ref_history,(-1,1))
alpha_ref_reshape = alpha_ref[:40000,:]

#vanilla
state_vanilla_reshape = state_vanilla[:40000,:]
action_vanilla_reshape = action_vanilla[:40000,]

#q_ref_noise_vanilla = np.reshape(q_ref_vanilla,(-1,1))
#q_ref_noise_vanilla_reshape = q_ref_noise_vanilla[:40000,:]
q_ref_vanilla = np.reshape(q_ref_vanilla,(-1,1))
q_ref_vanilla_reshape = q_ref_vanilla[:40000,:]

alpha_vanilla = np.reshape(state_vanilla_reshape[:,0],(-1,1))
q_vanilla = np.reshape(state_vanilla_reshape[:,1],(-1,1))

#action
delta_vanilla = np.reshape(action_vanilla_reshape[:,1],(-1,1))

#tracking error
e_alpha_vanilla = alpha_vanilla - alpha_ref_reshape
e_q_vanilla = q_vanilla - q_ref_vanilla_reshape

actionpow = 0.000005 * np.power(delta_vanilla,2)
eq_vanillapow=np.power(e_q_vanilla,2)
reward_vanilla_agent1 = np.power(e_alpha_vanilla,2) + 0.000005 * np.power(q_vanilla,2)
reward_vanilla_agent2 = np.power(e_q_vanilla,2) + 0.000005 * np.power(delta_vanilla,2)

reward_vanilla_agent1_3040s = reward_vanilla_agent1[30000:40000]


#smooth
state_smooth_reshape = state_smooth[:40000,:]
action_smooth_reshape = action_smooth[:40000,]

q_ref_smooth = np.reshape(q_ref_smooth,(-1,1))
q_ref_smooth_reshape = q_ref_smooth[:40000,:]

alpha_smooth = np.reshape(state_smooth_reshape[:,0],(-1,1))
q_smooth = np.reshape(state_smooth_reshape[:,1],(-1,1))

#action
delta_smooth = np.reshape(action_smooth_reshape[:,1],(-1,1))

#tracking error
e_alpha_smooth = alpha_smooth - alpha_ref_reshape
e_q_smooth = q_smooth - q_ref_smooth_reshape

reward_smooth_agent1 = np.power(e_alpha_smooth,2) + 0.000005 * np.power(q_smooth,2)
reward_smooth_agent2 = np.power(e_q_smooth,2) + 0.000005 * np.power(delta_smooth,2)

reward_smooth_agent1_3040s = reward_smooth_agent1[30000:40000]


#filter
state_filter_reshape = state_filter[:40000,:]
action_filter_reshape = action_filter[:40000,]

q_ref_filter = np.reshape(q_ref_filter,(-1,1))
q_ref_filter_reshape = q_ref_filter[:40000,:]

alpha_filter = np.reshape(state_filter_reshape[:,0],(-1,1))
q_filter = np.reshape(state_filter_reshape[:,1],(-1,1))

#action
delta_filter = np.reshape(action_filter_reshape[:,1],(-1,1))

#tracking error
e_alpha_filter = alpha_filter - alpha_ref_reshape
e_q_filter = q_filter - q_ref_filter_reshape

reward_filter_agent1 = np.power(e_alpha_filter,2) + 0.000005 * np.power(q_filter,2)
reward_filter_agent2 = np.power(e_q_filter,2) + 0.000005 * np.power(delta_filter,2)
reward_filter_agent1_3040s = reward_filter_agent1[30000:40000]


fig1 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(reward_filter_agent1,linewidth=1.0,color = 'C1',linestyle='-',label='TS + Filter')
plt.plot(reward_smooth_agent1,linewidth=1.0,color = 'C2',linestyle='-',label='TS')
plt.plot(reward_vanilla_agent1,linewidth=1.0,color = 'C0',linestyle='--',label='Vanilla')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel('reward',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

#reward 1
fig2 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

plt.subplot(2,1,1)
plt.plot(reward_filter_agent1,linewidth=1.0,color = 'C1',linestyle='-',label='TS + Filter')
plt.plot(reward_smooth_agent1,linewidth=1.0,color = 'C2',linestyle='-',label='TS')
plt.plot(reward_vanilla_agent1,linewidth=1.0,color = 'C0',linestyle='-',label='Vanilla')
plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$c_{1}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Higher-level agent',fontsize=20)
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20,)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.subplot(2,1,2)
plt.plot(reward_filter_agent1_3040s,linewidth=1.0,color = 'C1',linestyle='-',label='TS + Filter')
plt.plot(reward_smooth_agent1_3040s,linewidth=1.0,color = 'C2',linestyle='-',label='TS')
plt.plot(reward_vanilla_agent1_3040s,linewidth=1.0,color = 'C0',linestyle='-',label='Vanilla')
#plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$c_{1}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Higher-level agent',fontsize=20)
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20,ncol = 3)
#plt.xlim([30000,40000])
#plt.ylim([0,15])
plt.xticks([0,2500,5000,7500, 10000], ['30','32.5','35','37.5','40'])
plt.xlabel('Time [s]',fontdict={'size':20})


plt.tight_layout()
plt.savefig('Reward1_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Reward1_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

#reward 2
fig3 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

plt.subplot(2,1,1)
plt.plot(reward_filter_agent2,linewidth=1.0,color = 'C1',linestyle='-',label='TS + Filter')
plt.plot(reward_smooth_agent2,linewidth=1.0,color = 'C2',linestyle='-',label='TS')
#plt.plot(reward_vanilla_agent2,linewidth=1.0,color = 'C0',linestyle='--',label='Vanilla')
#plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$c_{2}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Lower-level agent',fontsize=20)
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.subplot(2,1,2)
#plt.plot(reward_filter_agent2,linewidth=1.0,color = 'C1',linestyle='-',label='TS + Filter')
#plt.plot(reward_smooth_agent2,linewidth=1.0,color = 'C2',linestyle='-',label='TS')
plt.plot(reward_vanilla_agent2,linewidth=1.0,color = 'C0',linestyle='-',label='Vanilla')
#plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$c_{2}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Lower-level agent',fontsize=20)
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(0.9,1.025),fontsize=20)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xlabel('Time [s]',fontdict={'size':20})

plt.tight_layout()
plt.savefig('Reward2_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Reward2_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)



#reward_smoothness
fig4 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

plt.subplot(2,1,1)
#plt.plot(reward_filter_agent1,linewidth=1.0,color = 'C1',linestyle='-',label='TS + Filter')
plt.plot(reward_smooth_agent1,linewidth=1.0,color = 'C2',linestyle='-',label='TS')
plt.plot(reward_vanilla_agent1,linewidth=1.0,color = 'C0',linestyle='--',label='Baseline')
#plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$c_{1}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Higher-level agent',fontsize=20)
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20,)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.subplot(2,1,2)
#plt.plot(reward_filter_agent1,linewidth=1.0,color = 'C1',linestyle='-',label='TS + Filter')
plt.plot(reward_smooth_agent2,linewidth=1.0,color = 'C2',linestyle='-',label='TS')
plt.plot(reward_vanilla_agent2,linewidth=1.0,color = 'C0',linestyle='--',label='Vanilla')
plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$c_{2}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Lower-level agent',fontsize=20)
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20,)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.tight_layout()
plt.savefig('Reward_smoothness_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Reward_smoothness_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)


#reward_filter
fig5 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

plt.subplot(2,1,1)
plt.plot(reward_filter_agent1,linewidth=1.0,color = 'C3',linestyle='-',label='TS + Filter')
plt.plot(reward_smooth_agent1,linewidth=1.0,color = 'C2',linestyle='--',label='TS')
#plt.plot(reward_vanilla_agent1,linewidth=1.0,color = 'C0',linestyle='--',label='Vanilla')
#plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$c_{1}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Higher-level agent',fontsize=20)
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20,)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.subplot(2,1,2)
plt.plot(reward_filter_agent1,linewidth=1.0,color = 'C3',linestyle='-',label='TS + Filter')
plt.plot(reward_smooth_agent2,linewidth=1.0,color = 'C2',linestyle='--',label='TS')
#plt.plot(reward_vanilla_agent2,linewidth=1.0,color = 'C0',linestyle='--',label='Vanilla')
plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$c_{2}$',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Lower-level agent',fontsize=20)
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20,)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

plt.tight_layout()
plt.savefig('Reward_filter_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Reward_filter_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)


plt.show()
