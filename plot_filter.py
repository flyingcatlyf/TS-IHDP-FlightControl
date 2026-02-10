import numpy as np
import matplotlib.pyplot as plt
import matplotlib


state_unfilter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/state_history')
action_unfilter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/action_history')
q_ref_noise_unfilter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/q_ref_noise_history')
q_ref_unfilter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/q_ref_filtered_history')
state_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/state_history')
action_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/action_history')
q_ref_noise_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/q_ref_noise_history')
q_ref_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/q_ref_filtered_history')

alpha_ref_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/alpha_ref_history')

#unfilter
state_unfilter_reshape = state_unfilter[:40000,:]
action_unfilter_reshape = action_unfilter[:40000,]

q_ref_noise_unfilter = np.reshape(q_ref_noise_unfilter,(-1,1))
q_ref_noise_unfilter_reshape = q_ref_noise_unfilter[:40000,:]
q_ref_unfilter = np.reshape(q_ref_unfilter,(-1,1))
q_ref_unfilter_reshape = q_ref_unfilter[:40000,:]

q_unfilter = np.reshape(state_unfilter_reshape[:,1],(-1,1))
q_unfilter_reshape = q_unfilter[:40000,:]

q_unfilter_reshape_1417s = q_unfilter_reshape[14000:17000,:]
q_unfilter_reshape_1923s = q_unfilter_reshape[19000:23000,:]
q_unfilter_reshape_2428s = q_unfilter_reshape[24000:28000,:]
q_unfilter_reshape_3340s = q_unfilter_reshape[33000:40000,:]

q_ref_noise_unfilter_reshape_1417s = q_ref_noise_unfilter_reshape[14000:17000,:]
q_ref_noise_unfilter_reshape_1923s = q_ref_noise_unfilter_reshape[19000:23000,:]
q_ref_noise_unfilter_reshape_2428s = q_ref_noise_unfilter_reshape[24000:28000,:]
q_ref_noise_unfilter_reshape_3340s = q_ref_noise_unfilter_reshape[33000:40000,:]

q_ref_unfilter_reshape_1417s = q_ref_unfilter_reshape[14000:17000,:]
q_ref_unfilter_reshape_1922s = q_ref_unfilter_reshape[19000:22000,:]
q_ref_unfilter_reshape_2428s = q_ref_unfilter_reshape[24000:28000,:]
q_ref_unfilter_reshape_3340s = q_ref_unfilter_reshape[33000:40000,:]


#action
action_unfilter = np.reshape(action_unfilter_reshape[:,1],(-1,1))
action_unfilter_reshape_1317s = action_unfilter[13000:17000,:]
action_unfilter_reshape_1011s = action_unfilter[10000:11000,:]
action_unfilter_reshape_2832s = action_unfilter[28000:32000,:]
action_unfilter_reshape_3238s = action_unfilter[32000:38000,:]

#arrary to list
q_ref_noise_unfilter_reshape_1417s_array = np.reshape(q_ref_noise_unfilter_reshape_1417s,(3000,))
q_ref_noise_unfilter_reshape_1923s_array = np.reshape(q_ref_noise_unfilter_reshape_1923s,(4000,))
q_ref_noise_unfilter_reshape_2428s_array = np.reshape(q_ref_noise_unfilter_reshape_2428s,(4000,))
q_ref_noise_unfilter_reshape_3340s_array = np.reshape(q_ref_noise_unfilter_reshape_3340s,(7000,))

action_unfilter_reshape_1317s_array = np.reshape(action_unfilter_reshape_1317s,(4000,))
action_unfilter_reshape_1011s_array = np.reshape(action_unfilter_reshape_1011s,(1000,))
action_unfilter_reshape_2832s_array = np.reshape(action_unfilter_reshape_2832s,(4000,))
action_unfilter_reshape_3238s_array = np.reshape(action_unfilter_reshape_3238s,(6000,))



#filter
state_filter_reshape = state_filter[:40000,:]
action_filter_reshape = action_filter[:40000,]
q_ref_noise_filter = np.reshape(q_ref_noise_filter,(-1,1))
q_ref_noise_filter_reshape = q_ref_noise_filter[:40000,:]
q_ref_filter = np.reshape(q_ref_filter,(-1,1))
q_ref_filter_reshape = q_ref_filter[:40000,:]

q_filter = np.reshape(state_filter_reshape[:,1],(-1,1))
q_filter_reshape = q_filter[:40000,:]

q_filter_reshape_1417s = q_filter_reshape[14000:17000,:]
q_filter_reshape_1923s = q_filter_reshape[19000:23000,:]
q_filter_reshape_2428s = q_filter_reshape[24000:28000,:]
q_filter_reshape_3340s = q_filter_reshape[33000:40000,:]

q_ref_noise_filter_reshape_1417s = q_ref_noise_filter_reshape[14000:17000,:]
q_ref_noise_filter_reshape_1923s = q_ref_noise_filter_reshape[19000:23000,:]
q_ref_noise_filter_reshape_2428s = q_ref_noise_filter_reshape[24000:28000,:]
q_ref_noise_filter_reshape_3340s = q_ref_noise_filter_reshape[33000:40000,:]

q_ref_filter_reshape_1417s = q_ref_filter_reshape[14000:17000,:]
q_ref_filter_reshape_1923s = q_ref_filter_reshape[19000:23000,:]
q_ref_filter_reshape_2428s = q_ref_filter_reshape[24000:28000,:]
q_ref_filter_reshape_3340s = q_ref_filter_reshape[33000:40000,:]

#action
action_filter = np.reshape(action_filter_reshape[:,1],(-1,1))
action_filter_reshape_1317s = action_filter[13000:17000,:]
action_filter_reshape_1011s = action_filter[10000:11000,:]
action_filter_reshape_2832s = action_filter[28000:32000,:]
action_filter_reshape_3238s = action_filter[32000:38000,:]

#arrary to list
q_ref_filter_reshape_1417s_array = np.reshape(q_ref_filter_reshape_1417s,(3000,))
q_ref_filter_reshape_1923s_array = np.reshape(q_ref_filter_reshape_1923s,(4000,))
q_ref_filter_reshape_2428s_array = np.reshape(q_ref_filter_reshape_2428s,(4000,))
q_ref_filter_reshape_3340s_array = np.reshape(q_ref_filter_reshape_3340s,(7000,))

action_filter_reshape_1317s_array = np.reshape(action_filter_reshape_1317s,(4000,))
action_filter_reshape_1011s_array = np.reshape(action_filter_reshape_1011s,(1000,))
action_filter_reshape_2832s_array = np.reshape(action_filter_reshape_2832s,(4000,))
action_filter_reshape_3238s_array = np.reshape(action_filter_reshape_3238s,(6000,))


alpha_ref_history = np.reshape(alpha_ref_history,(-1,1))
alpha_ref_history_reshape = alpha_ref_history[:40000,:]



time = np.array(range(0,40000,1))


fig1 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{ref}$')
plt.plot(state_unfilter_reshape[:,0],linewidth=1.0,color = 'C0',label=r'$\alpha$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig2 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{ref}$')
plt.plot(state_filter_reshape[:,0],linewidth=1.0,color = 'C0',label=r'$\alpha$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig3 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(q_ref_noise_unfilter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{actor}$')
#plt.plot(q_ref_unfilter_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{filter}$')
plt.plot(state_unfilter_reshape[:,1],linewidth=1.0,color = 'C0',label='q')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title('Pitch rate')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig4 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(q_ref_noise_filter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{actor}$')
plt.plot(q_ref_filter_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{filter}$')
plt.plot(state_filter_reshape[:,1],linewidth=1.0,color = 'C0',label='q')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title('Pitch rate')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig5 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
#plt.plot(action_history_reshape[:,0],linewidth=1.0,color = 'C0',label=r'$q$')
plt.plot(action_unfilter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title(r'$\delta history$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig6 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
#plt.plot(action_history_reshape[:,0],linewidth=1.0,color = 'C0',label=r'$q$')
plt.plot(action_filter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title(r'$\delta history$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)



#9_subplots_for_states
fig7 = plt.figure(figsize=(18.0,9.0))
plt.subplot(3,2,1)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
plt.plot(state_unfilter_reshape[:,0],linewidth=1.0,color = 'C0')
plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.title('Temporally smoothed',fontsize=19)
plt.legend(loc='upper right',fontsize=19)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.subplot(3,2,2)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
plt.plot(state_filter_reshape[:,0],linewidth=1.0,color = 'C0')
plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.title('Temporally smoothed and filtered',fontsize=19)
plt.legend(loc='upper right',fontsize=19)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.subplot(3,2,3)
plt.plot(q_ref_noise_unfilter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref}}$')
#plt.plot(q_ref_unfilter_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{ref(filter)}$')
plt.plot(state_unfilter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':19})
plt.grid(True)
#plt.title('DDPG (tanh-ReLU)')
plt.legend(loc='upper right',fontsize=19)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.subplot(3,2,4)
plt.plot(q_ref_noise_filter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref}}$')
plt.plot(q_ref_filter_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q^{\prime}_{\mathrm{ref}}$')
plt.plot(state_filter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':19})
plt.grid(True)
#plt.title('DDPG (tanh-ReLU)')
plt.legend(loc='upper right',fontsize=19)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)


plt.subplot(3,2,5)
plt.plot(action_unfilter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':19})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.subplot(3,2,6)
plt.plot(action_filter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':19})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=19)
plt.yticks(fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.tight_layout()
plt.savefig('Filter_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

fig7 = plt.figure(figsize=(18.0,9.0))
plt.subplot(3,2,1)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
plt.plot(state_unsmooth_reshape[:,0],linewidth=1.0,color = 'C0')
plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.title('Baseline', fontdict={'size':19})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.subplot(3,2,2)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
plt.plot(state_smooth_reshape[:,0],linewidth=1.0,color = 'C0')
plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.title('Temporally smoothed', fontdict={'size':19})
plt.legend(loc='upper right',fontsize=19)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)


plt.subplot(3,2,3)
plt.plot(q_ref_noise_unsmooth_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref}}$')
#plt.plot(q_ref_unfilter_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{ref(filter)}$')
plt.plot(state_unsmooth_reshape[:,1],linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':19})
plt.grid(True)
#plt.title('DDPG (tanh-ReLU)')
#plt.legend(loc='upper left',fontsize=10)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.subplot(3,2,4)
plt.plot(q_ref_noise_smooth_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref}}$')
#plt.plot(q_ref_smooth_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
plt.plot(state_smooth_reshape[:,1],linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':19})
plt.grid(True)
#plt.title('DDPG (tanh-ReLU)')
plt.legend(loc='upper right',fontsize=19)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.subplot(3,2,5)
plt.plot(action_unsmooth_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':19})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.subplot(3,2,6)
plt.plot(action_smooth_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':19})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.tight_layout()
plt.savefig('Smooth_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Smooth_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

#9_subplots_for_states
#fig7 = plt.figure(figsize=(18.0,9.0))
#plt.subplot(3,2,1)
#plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
#plt.plot(state_unfilter_reshape[:,0],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':15})
#plt.grid(True)
#plt.title('Unfiltered',fontdict={'size':15})
#plt.legend(loc='upper right',fontsize=15)
#plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=18)
#plt.yticks(fontsize=15)

#plt.subplot(3,2,2)
#plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
#plt.plot(state_filter_reshape[:,0],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':15})
#plt.grid(True)
#plt.title('Filtered',fontdict={'size':15})
#plt.legend(loc='upper right',fontsize=15)
#plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=18)
#plt.yticks(fontsize=15)

#plt.subplot(3,2,3)
#plt.plot(q_ref_noise_unfilter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref (actor)}}$')
#plt.plot(state_unfilter_reshape[:,1],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':15})
#plt.grid(True)
#plt.legend(loc='upper right',fontsize=15)
#plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
#plt.yticks(fontsize=15)

#plt.subplot(3,2,4)
#plt.plot(q_ref_noise_filter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref (actor)}}$')
#plt.plot(q_ref_filter_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
#plt.plot(state_filter_reshape[:,1],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':15})
#plt.grid(True)
#plt.legend(loc='upper right',fontsize=15,ncol=2)
#plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
#plt.yticks(fontsize=15)


#plt.subplot(3,2,5)
#plt.plot(action_unfilter_reshape[:,1],linewidth=1.0,color = 'C0')
#plt.xlabel('Time [s]',fontdict={'size':15})
#plt.ylabel(r'$\delta$ [deg]',fontdict={'size':15})
#plt.grid(True)
#plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
#plt.yticks(fontsize=15)


#plt.subplot(3,2,6)
#plt.plot(action_filter_reshape[:,1],linewidth=1.0,color = 'C0')
#plt.xlabel('Time [s]',fontdict={'size':15})
#plt.ylabel(r'$\delta$ [deg]',fontdict={'size':15})
#plt.grid(True)
#plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
#plt.yticks(fontsize=15)
#plt.savefig('Filter_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
#plt.savefig('Filter_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)


fig8 = plt.figure(figsize=(18.0,9.0))
plt.subplot(2,2,1)
plt.plot(q_ref_noise_unfilter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref(actor)}$')
plt.plot(q_unfilter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS',fontdict={'size':10})
plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 3000], ['14', '17'])

plt.subplot(2,2,2)
plt.plot(q_ref_noise_filter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref(actor)}$')
plt.plot(q_ref_filter_reshape_1417s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{ref(filter)}$')
plt.plot(q_filter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS + Filter',fontdict={'size':10})
plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 3000], ['14', '17'])

plt.subplot(2,2,3)
plt.plot(q_ref_noise_unfilter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
#plt.title('Unfiltered')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 7000], ['33', '40'])


plt.subplot(2,2,4)
plt.plot(q_ref_noise_filter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_ref_filter_reshape_3340s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{ref(filter)}$')
plt.plot(q_filter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
#plt.title('Filtered',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 7000], ['33', '40'])

plt.savefig('Filter_comparison_zoom.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison_zoom.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

fig9 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(q_ref_noise_unfilter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref(actor)}$')
plt.plot(q_unfilter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

fig10 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(q_ref_noise_filter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref(actor)}$')
plt.plot(q_filter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

fig11 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(q_ref_noise_unfilter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref(actor)}$')
plt.plot(q_unfilter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)
plt.xticks([0, 7000], ['33', '40'])

fig12 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(q_ref_noise_filter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref(actor)}$')
plt.plot(q_filter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)
plt.xticks([0, 7000], ['33', '40'])


#q comparison
fig13 = plt.figure(figsize=(18.0,9.0))
plt.subplot(2,4,1)
plt.plot(q_ref_noise_unfilter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 1000], ['5', '6'])

plt.subplot(2,4,2)
plt.plot(q_ref_noise_filter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_ref_filter_reshape_1417s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_filter_reshape_1417s,linewidth=1.0,color = 'C0')
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS + Filter',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 3000], ['14', '17'])

plt.subplot(2,4,3)
plt.plot(q_ref_noise_unfilter_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_1923s,linewidth=1.0,color = 'C0')
#plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 3000], ['19', '22'])

plt.subplot(2,4,4)
plt.plot(q_ref_noise_filter_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref(actor)}$')
plt.plot(q_ref_filter_reshape_1923s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{ref(filter)}$')
plt.plot(q_filter_reshape_1923s,linewidth=1.0,color = 'C0')
#plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS + Filter',fontdict={'size':10})
plt.legend(loc='lower right',fontsize=10)
plt.xticks([0, 4000], ['19', '23'])

plt.subplot(2,4,5)
plt.plot(q_ref_noise_unfilter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS',fontdict={'size':10})
plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 7000], ['33', '40'])

plt.subplot(2,4,6)
plt.plot(q_ref_noise_filter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{ref(actor)}$')
plt.plot(q_ref_filter_reshape_3340s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{ref(filter)}$')
plt.plot(q_filter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
#plt.title('Filtered',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 7000], ['33', '40'])

plt.subplot(2,4,7)
plt.plot(q_ref_noise_unfilter_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_2428s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
#plt.title('Unfiltered',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 4000], ['24', '28'])

plt.subplot(2,4,8)
plt.plot(q_ref_noise_filter_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_ref_filter_reshape_2428s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_filter_reshape_2428s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
#plt.title('Filtered',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 4000], ['24', '28'])

#plt.savefig('Filter_comparison_q.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
#plt.savefig('Filter_comparison_q.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)


#delta comparison
fig14 = plt.figure(figsize=(18.0,9.0))
plt.subplot(2,4,1)
plt.plot(action_unfilter_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.title('TS',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 4000], ['13', '17'])

plt.subplot(2,4,2)
plt.plot(action_filter_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS + Filter',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 4000], ['13', '17'])

plt.subplot(2,4,3)
plt.plot(action_unfilter_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
#plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 1000], ['10', '11'])

plt.subplot(2,4,4)
plt.plot(action_filter_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
#plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS + Filter',fontdict={'size':10})
#plt.legend(loc='lower right',fontsize=10)
plt.xticks([0, 1000], ['10', '11'])

plt.subplot(2,4,5)
plt.plot(action_unfilter_reshape_2832s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':10})
plt.grid(True)
plt.title('TS',fontdict={'size':10})
plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 4000], ['28', '32'])

plt.subplot(2,4,6)
plt.plot(action_filter_reshape_2832s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
#plt.title('Filtered',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 4000], ['28', '32'])

plt.subplot(2,4,7)
plt.plot(action_unfilter_reshape_3238s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
#plt.title('Unfiltered',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 6000], ['32', '38'])

plt.subplot(2,4,8)
plt.plot(action_filter_reshape_3238s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
#plt.title('Filtered',fontdict={'size':10})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 6000], ['32', '38'])

#plt.savefig('Filter_comparison_action.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
#plt.savefig('Filter_comparison_action.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)




#q frequency comparison
fig15 = plt.figure(figsize=(18.0,9.0))
plt.subplot(4,4,1)
plt.plot(q_ref_noise_unfilter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

plt.subplot(4,4,2)
plt.plot(q_ref_noise_filter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_ref_filter_reshape_1417s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_filter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.grid(True)
plt.xlabel('Time [s]',fontdict={'size':10})
plt.title('TS + Filter')
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

plt.subplot(4,4,3)
plt.plot(q_ref_noise_unfilter_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_1923s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS')
plt.xticks([0, 2000, 4000], ['19', '21', '23'])

plt.subplot(4,4,4)
plt.plot(q_ref_noise_filter_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref (actor)}}$')
plt.plot(q_ref_filter_reshape_1923s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
plt.plot(q_filter_reshape_1923s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.title('TS + Filter')
plt.grid(True)
plt.legend(loc='lower right',fontsize=10)
plt.xticks([0, 2000, 4000], ['19', '21', '23'])

plt.subplot(4,4,5)
plt.magnitude_spectrum(q_ref_noise_unfilter_reshape_1417s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.4)
plt.xlabel('Frequency')
plt.grid(True)

plt.subplot(4,4,6)
plt.magnitude_spectrum(q_ref_filter_reshape_1417s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency')
plt.ylabel('')
plt.ylim(0,0.4)
plt.grid(True)

plt.subplot(4,4,7)
plt.magnitude_spectrum(q_ref_noise_unfilter_reshape_1923s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency')
plt.ylabel('')
plt.ylim(0,0.4)
plt.grid(True)

plt.subplot(4,4,8)
plt.magnitude_spectrum(q_ref_filter_reshape_1923s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency')
plt.ylabel('')
plt.ylim(0,0.4)
plt.grid(True)

plt.subplot(4,4,9)
plt.plot(q_ref_noise_unfilter_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_2428s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 2000, 4000], ['24', '26', '28'])

plt.subplot(4,4,10)
plt.plot(q_ref_noise_filter_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_ref_filter_reshape_2428s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_filter_reshape_2428s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]')
plt.grid(True)
plt.xticks([0, 2000, 4000], ['24', '26', '28'])

plt.subplot(4,4,11)
plt.plot(q_ref_noise_unfilter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]')
plt.grid(True)
plt.xticks([0, 3500, 7000], ['33', '36.5', '40'])

plt.subplot(4,4,12)
plt.plot(q_ref_noise_filter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref(actor)}}$')
plt.plot(q_ref_filter_reshape_3340s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref(filter)}}$')
plt.plot(q_filter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]')
plt.grid(True)
plt.xticks([0, 3500, 7000], ['33', '36.5', '40'])


plt.subplot(4,4,13)
plt.magnitude_spectrum(q_ref_noise_unfilter_reshape_2428s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.1)
plt.xlabel('Frequency')
plt.grid(True)

plt.subplot(4,4,14)
plt.magnitude_spectrum(q_ref_filter_reshape_2428s_array, Fs=1/0.001, color='C3')
plt.ylabel('')
plt.ylim(0,0.1)
plt.xlabel('Frequency')
plt.grid(True)


plt.subplot(4,4,15)
plt.magnitude_spectrum(q_ref_noise_unfilter_reshape_3340s_array, Fs=1/0.001, color='C3')
plt.ylabel('')
plt.ylim(0,0.1)
plt.xlabel('Frequency')
plt.grid(True)

plt.subplot(4,4,16)
plt.magnitude_spectrum(q_ref_filter_reshape_3340s_array, Fs=1/0.001, color='C3')
plt.ylabel('')
plt.ylim(0,0.1)
plt.xlabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.savefig('Filter_comparison_q_FFT.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison_q_FFT.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)



#delta frequency comparison
fig16 = plt.figure(figsize=(18.0,9.0))
plt.subplot(4,4,1)
plt.plot(action_unfilter_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('TS')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':10})
plt.xticks([0, 500, 1000], ['10', '10.5', '11'])

plt.subplot(4,4,2)
plt.plot(action_filter_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('TS + Filter')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.xticks([0, 500, 1000], ['10', '10.5', '11'])

plt.subplot(4,4,3)
plt.plot(action_unfilter_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.title('TS')
plt.xticks([0, 2000, 4000], ['13', '15', '17'])

plt.subplot(4,4,4)
plt.plot(action_filter_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('TS + Filter')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.xticks([0, 2000, 4000], ['13', '15', '17'])

plt.subplot(4,4,5)
plt.magnitude_spectrum(action_unfilter_reshape_1011s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,0.4)
plt.grid(True)

plt.subplot(4,4,6)
plt.magnitude_spectrum(action_filter_reshape_1011s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency',fontdict={'size':10})
plt.ylabel('')
plt.ylim(0,0.4)
plt.grid(True)

plt.subplot(4,4,7)
plt.magnitude_spectrum(action_unfilter_reshape_1317s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.4)
plt.xlabel('Frequency',fontdict={'size':10})
plt.ylabel('')
plt.grid(True)

plt.subplot(4,4,8)
plt.magnitude_spectrum(action_filter_reshape_1317s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency',fontdict={'size':10})
plt.ylabel('')
plt.ylim(0,0.4)
plt.grid(True)

plt.subplot(4,4,9)
plt.plot(action_unfilter_reshape_2832s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':10})
plt.grid(True)
#plt.title('Unfiltered')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 2000, 4000], ['28', '30', '32'])

plt.subplot(4,4,10)
plt.plot(action_filter_reshape_2832s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 2000, 4000], ['28', '30', '32'])

plt.subplot(4,4,11)
plt.plot(action_unfilter_reshape_3238s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 3000, 6000], ['32', '35', '38'])

plt.subplot(4,4,12)
plt.plot(action_filter_reshape_3238s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 3000, 6000], ['32', '35', '38'])

plt.subplot(4,4,13)
plt.magnitude_spectrum(action_unfilter_reshape_2832s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.1)
plt.xlabel('Frequency',fontdict={'size':10})
plt.grid(True)

plt.subplot(4,4,14)
plt.magnitude_spectrum(action_filter_reshape_2832s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency',fontdict={'size':10})
plt.ylabel('')
plt.ylim(0,0.1)
plt.grid(True)

plt.subplot(4,4,15)
plt.magnitude_spectrum(action_unfilter_reshape_3238s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency',fontdict={'size':10})
plt.ylabel('')
plt.ylim(0,0.1)
plt.grid(True)

plt.subplot(4,4,16)
plt.magnitude_spectrum(action_filter_reshape_3238s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency',fontdict={'size':10})
plt.ylabel('')
plt.ylim(0,0.1)
plt.grid(True)

plt.tight_layout()
plt.savefig('Filter_comparison_action_FFT.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison_action_FFT.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.show()