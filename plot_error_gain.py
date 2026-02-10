import numpy as np
import matplotlib.pyplot as plt
import matplotlib

state_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/state_history')
alpha_ref_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/alpha_ref_history')
q_ref_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/q_ref_noise_history')
action_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/action_history')

state_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/state_history')
alpha_ref_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/alpha_ref_history')
q_ref_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/q_ref_noise_history')
action_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/action_history')

state_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter2/state_history')
alpha_ref_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter2/alpha_ref_history')
q_ref_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter2/q_ref_noise_history')
action_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter2/action_history')


#vanilla
state_vanilla_reshape = state_vanilla[:40000,:]
action_vanilla_reshape = action_vanilla[:40000,]
alpha_ref_vanilla = np.reshape(alpha_ref_vanilla,(-1,1))
alpha_ref_vanilla_reshape = alpha_ref_vanilla[:40000,:]
q_ref_vanilla = np.reshape(q_ref_vanilla,(-1,1))
q_ref_vanilla_reshape = q_ref_vanilla[:40000,:]

alpha_vanilla = state_vanilla_reshape[:,0]
q_vanilla = state_vanilla_reshape[:,1]
delta_vanilla = action_vanilla_reshape[:,1]

alpha_ref_vanilla_reshape = alpha_ref_vanilla_reshape.reshape(-1)
q_ref_vanilla_reshape = q_ref_vanilla_reshape.reshape(-1)

e_alpha_vanilla = alpha_vanilla - alpha_ref_vanilla_reshape
e_q_vanilla = q_vanilla - q_ref_vanilla_reshape

K1_vanilla = (q_ref_vanilla_reshape / e_alpha_vanilla)
K2_vanilla = (delta_vanilla / e_q_vanilla)

K1_vanilla = K1_vanilla[1:]
K2_vanilla = K2_vanilla[1:]

#TS
state_TS_reshape = state_TS[:40000,:]
action_TS_reshape = action_TS[:40000,]
alpha_ref_TS = np.reshape(alpha_ref_TS,(-1,1))
alpha_ref_TS_reshape = alpha_ref_TS[:40000,:]
q_ref_TS = np.reshape(q_ref_TS,(-1,1))
q_ref_TS_reshape = q_ref_TS[:40000,:]

alpha_TS = state_TS_reshape[:,0]
q_TS = state_TS_reshape[:,1]
delta_TS = action_TS_reshape[:,1]

alpha_ref_TS_reshape = alpha_ref_TS_reshape.reshape(-1)
q_ref_TS_reshape = q_ref_TS_reshape.reshape(-1)

e_alpha_TS = alpha_TS - alpha_ref_TS_reshape
e_q_TS = q_TS - q_ref_TS_reshape

K1_TS = (q_ref_TS_reshape / e_alpha_TS)
K2_TS = (delta_TS / e_q_TS)

K1_TS = K1_TS[1:]
K2_TS = K2_TS[1:]

#filter
state_filter_reshape = state_filter[:40000,:]
action_filter_reshape = action_filter[:40000,]
alpha_ref_filter = np.reshape(alpha_ref_filter,(-1,1))
alpha_ref_filter_reshape = alpha_ref_filter[:40000,:]
q_ref_filter = np.reshape(q_ref_filter,(-1,1))
q_ref_filter_reshape = q_ref_filter[:40000,:]

alpha_filter = state_filter_reshape[:,0]
q_filter = state_filter_reshape[:,1]
delta_filter = action_filter_reshape[:,1]

alpha_ref_filter_reshape = alpha_ref_filter_reshape.reshape(-1)
q_ref_filter_reshape = q_ref_filter_reshape.reshape(-1)

e_alpha_filter = alpha_filter - alpha_ref_filter_reshape
e_q_filter = q_filter - q_ref_filter_reshape

K1_filter = (q_ref_filter_reshape / e_alpha_filter)
K2_filter = (delta_filter / e_q_filter)

K1_filter = K1_filter[1:]
K2_filter = K2_filter[1:]

fig1 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(K1_vanilla,linewidth=1.0,color = 'C0',label=r'$K1$')
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
plt.plot(K2_vanilla,linewidth=1.0,color = 'C0',label=r'$K2$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\alpha Tracking$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig3 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(K1_TS,linewidth=1.0,color = 'C0',label=r'$K1$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\alpha Tracking$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig4 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(K2_TS,linewidth=1.0,color = 'C0',label=r'$K2$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\alpha Tracking$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig5 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(K1_filter,linewidth=1.0,color = 'C0',label=r'$K1$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\alpha Tracking$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig6 = plt.figure(figsize=(18.0,9.0))
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.plot(K2_filter,linewidth=1.0,color = 'C0',label=r'$K2$')
plt.xlabel('Time [s]',fontdict={'size':35})
plt.ylabel(r'$\alpha Tracking$ [deg]',fontdict={'size':35})
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title(r'$\alpha tracking$')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

plt.show()


