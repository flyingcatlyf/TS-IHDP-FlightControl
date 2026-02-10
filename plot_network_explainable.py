import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#Vanilla
Critic_1_Lay1_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Critic_1_Lay1_history')
Critic_1_Lay2_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Critic_1_Lay2_history')
Critic_2_Lay1_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Critic_2_Lay1_history')
Critic_2_Lay2_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Critic_2_Lay2_history')
Actor_1_Lay1_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Actor_1_Lay1_history')
Actor_1_Lay2_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Actor_1_Lay2_history')
Actor_2_Lay1_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Actor_2_Lay1_history')
Actor_2_Lay2_history_vanilla = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter2/Actor_2_Lay2_history')

#TS
Critic_1_Lay1_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Critic_1_Lay1_history')
Critic_1_Lay2_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Critic_1_Lay2_history')
Critic_2_Lay1_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Critic_2_Lay1_history')
Critic_2_Lay2_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Critic_2_Lay2_history')
Actor_1_Lay1_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Actor_1_Lay1_history')
Actor_1_Lay2_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Actor_1_Lay2_history')
Actor_2_Lay1_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Actor_2_Lay1_history')
Actor_2_Lay2_history_TS = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion2/Actor_2_Lay2_history')

#vanilla
Critic_1_Lay1_history_reshape_vanilla = Critic_1_Lay1_history_vanilla[:40000,:]
Critic_1_Lay2_history_reshape_vanilla = Critic_1_Lay2_history_vanilla[:40000,:]
Critic_2_Lay1_history_reshape_vanilla = Critic_2_Lay1_history_vanilla[:40000,:]
Critic_2_Lay2_history_reshape_vanilla = Critic_2_Lay1_history_vanilla[:40000,:]
Actor_1_Lay1_history_reshape_vanilla = Actor_1_Lay1_history_vanilla[:40000,:]
Actor_1_Lay2_history_reshape_vanilla = Actor_1_Lay2_history_vanilla[:40000,:]
Actor_2_Lay1_history_reshape_vanilla = Actor_2_Lay1_history_vanilla[:40000,:]
Actor_2_Lay2_history_reshape_vanilla = Actor_2_Lay1_history_vanilla[:40000,:]

#TS
Critic_1_Lay1_history_reshape_TS = Critic_1_Lay1_history_TS[:40000,:]
Critic_1_Lay2_history_reshape_TS = Critic_1_Lay2_history_TS[:40000,:]
Critic_2_Lay1_history_reshape_TS = Critic_2_Lay1_history_TS[:40000,:]
Critic_2_Lay2_history_reshape_TS = Critic_2_Lay1_history_TS[:40000,:]
Actor_1_Lay1_history_reshape_TS = Actor_1_Lay1_history_TS[:40000,:]
Actor_1_Lay2_history_reshape_TS = Actor_1_Lay2_history_TS[:40000,:]
Actor_2_Lay1_history_reshape_TS = Actor_2_Lay1_history_TS[:40000,:]
Actor_2_Lay2_history_reshape_TS = Actor_2_Lay1_history_TS[:40000,:]


fig1 = plt.figure(figsize=(18.0,9.0))
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,0],linewidth=1.0,color = 'C0',label = '1')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,1],linewidth=1.0,color = 'C1',label = '2')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,2],linewidth=1.0,color = 'C2',label = '3')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,3],linewidth=1.0,color = 'C0',label = '4')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,4],linewidth=1.0,color = 'C1',label = '5')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,5],linewidth=1.0,color = 'C2',label = '6')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,6],linewidth=1.0,color = 'C0',label = '7')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,7],linewidth=1.0,color = 'C1',label = '8')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,8],linewidth=1.0,color = 'C2',label = '9')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,9],linewidth=1.0,color = 'C0',label = '10')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,10],linewidth=1.0,color = 'C1',label = '11')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,11],linewidth=1.0,color = 'C2',label = '12')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,12],linewidth=1.0,color = 'C0',label = '13')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,13],linewidth=1.0,color = 'C1',label = '14')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,14],linewidth=1.0,color = 'C2',label = '15')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,15],linewidth=1.0,color = 'C0',label = '16')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,16],linewidth=1.0,color = 'C1',label = '17')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,17],linewidth=1.0,color = 'C2',label = '18')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,18],linewidth=1.0,color = 'C0',label = '19')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,19],linewidth=1.0,color = 'C1',label = '20')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,20],linewidth=1.0,color = 'C2',label = '21')
#plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,0],linewidth=1.0,color = 'C0',label = '22')
#plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,1],linewidth=1.0,color = 'C0',label = '23')
#plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,2],linewidth=1.0,color = 'C0',label = '24')
#plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,3],linewidth=1.0,color = 'C0',label = '25')
#plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,4],linewidth=1.0,color = 'C0',label = '26')
#plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,5],linewidth=1.0,color = 'C0',label = '27')
#plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,6],linewidth=1.0,color = 'C0',label = '28')
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=14)
plt.ylabel(r'$\vartheta_{1}$',fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)

fig2 = plt.figure(figsize=(18.0,9.0))
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,0],linewidth=1.0,color = 'C0',label = '1')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,1],linewidth=1.0,color = 'C1',label = '2')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,2],linewidth=1.0,color = 'C2',label = '3')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,3],linewidth=1.0,color = 'C0',label = '4')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,4],linewidth=1.0,color = 'C1',label = '5')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,5],linewidth=1.0,color = 'C2',label = '6')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,6],linewidth=1.0,color = 'C0',label = '7')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,7],linewidth=1.0,color = 'C1',label = '8')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,8],linewidth=1.0,color = 'C2',label = '9')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,9],linewidth=1.0,color = 'C0',label = '10')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,10],linewidth=1.0,color = 'C1',label = '11')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,11],linewidth=1.0,color = 'C2',label = '12')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,12],linewidth=1.0,color = 'C0',label = '13')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,13],linewidth=1.0,color = 'C1',label = '14')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,14],linewidth=1.0,color = 'C2',label = '15')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,15],linewidth=1.0,color = 'C0',label = '16')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,16],linewidth=1.0,color = 'C1',label = '17')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,17],linewidth=1.0,color = 'C2',label = '18')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,18],linewidth=1.0,color = 'C0',label = '19')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,19],linewidth=1.0,color = 'C1',label = '20')
#plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,20],linewidth=1.0,color = 'C2',label = '21')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,0],linewidth=3.0,label = '22')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,1],linewidth=3.0,label = '23')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,2],linewidth=3.0,label = '24')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,3],linewidth=1.0,label = '25')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,4],linewidth=1.0,label = '26')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,5],linewidth=1.0,label = '27')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,6],linewidth=1.0,label = '28')
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=14)
plt.ylabel(r'$\vartheta_{1}$',fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)



fig3 = plt.figure(figsize=(18.0,9.0))
plt.subplot(3,2,1)
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,0],linewidth=1.0,label = '1')
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)

plt.subplot(3,2,2)
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,1],linewidth=1.0,label = '2')
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)

plt.subplot(3,2,3)
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,2],linewidth=1.0,label = '3')
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)

plt.subplot(3,2,4)
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,3],linewidth=3.0,label = '4')
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)

plt.subplot(3,2,5)
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,4],linewidth=1.0,label = '5')
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)

plt.subplot(3,2,6)
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,5],linewidth=1.0,label = '6')
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)

plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=14)
plt.ylabel(r'$\vartheta_{1}$',fontsize=14)
plt.yticks(fontsize=14)

fig4 = plt.figure(figsize=(18.0,9.0))
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,6],linewidth=1.0,label = '7')
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)

fig5 = plt.figure(figsize=(18.0,9.0))
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,0],linewidth=1.0,label = '1')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,3],linewidth=1.0,label = '4')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,6],linewidth=1.0,label = '7')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,9],linewidth=1.0,label = '10')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,12],linewidth=1.0,label = '13')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,15],linewidth=1.0,label = '16')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,18],linewidth=1.0,label = '19')
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=20)

plt.show()





plt.subplot(3,7,2)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,1],linewidth=1.0,color = 'C1',label = '2')

fig4 = plt.figure(figsize=(18.0,9.0))
plt.subplot(3,7,1)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,0],linewidth=1.0,color = 'C0',label = '1')

plt.subplot(3,7,2)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,1],linewidth=1.0,color = 'C1',label = '2')

plt.subplot(3,7,3)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,2],linewidth=1.0,color = 'C2',label = '3')
plt.subplot(3,7,4)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,3],linewidth=1.0,color = 'C0',label = '4')
plt.subplot(3,7,5)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,4],linewidth=1.0,color = 'C1',label = '5')
plt.subplot(3,7,6)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,5],linewidth=1.0,color = 'C2',label = '6')
plt.subplot(3,7,7)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,6],linewidth=1.0,color = 'C0',label = '7')
plt.subplot(3,7,8)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,7],linewidth=1.0,color = 'C1',label = '8')
plt.subplot(3,7,9)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,8],linewidth=1.0,color = 'C2',label = '9')
plt.subplot(3,7,10)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,9],linewidth=1.0,color = 'C0',label = '10')
plt.subplot(3,7,11)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,10],linewidth=1.0,color = 'C1',label = '11')
plt.subplot(3,7,12)
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,11],linewidth=1.0,color = 'C2',label = '12')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,12],linewidth=1.0,color = 'C0',label = '13')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,13],linewidth=1.0,color = 'C1',label = '14')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,14],linewidth=1.0,color = 'C2',label = '15')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,15],linewidth=1.0,color = 'C0',label = '16')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,16],linewidth=1.0,color = 'C1',label = '17')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,17],linewidth=1.0,color = 'C2',label = '18')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,18],linewidth=1.0,color = 'C0',label = '19')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,19],linewidth=1.0,color = 'C1',label = '20')
plt.plot(Actor_1_Lay1_history_reshape_vanilla[:,20],linewidth=1.0,color = 'C2',label = '21')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,0],linewidth=1.0,color = 'C0',label = '22')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,1],linewidth=1.0,color = 'C0',label = '23')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,2],linewidth=1.0,color = 'C0',label = '24')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,3],linewidth=1.0,color = 'C0',label = '25')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,4],linewidth=1.0,color = 'C0',label = '26')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,5],linewidth=1.0,color = 'C0',label = '27')
plt.plot(Actor_1_Lay2_history_reshape_vanilla[:,6],linewidth=1.0,color = 'C0',label = '28')
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=14)
plt.ylabel(r'$\vartheta_{1}$',fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=35)


plt.show()
