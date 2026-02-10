import numpy as np
import matplotlib.pyplot as plt
import matplotlib


state_unsmooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter/state_history')
action_unsmooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter/action_history')
q_ref_noise_unsmooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter/q_ref_noise_history')
q_ref_unsmooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/nosmooth_nofilter/q_ref_filtered_history')
state_smooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/state_history')
action_smooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/action_history')
q_ref_noise_smooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/q_ref_noise_history')
q_ref_smooth = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/q_ref_filtered_history')

alpha_ref_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/alpha_ref_history')

#unsmooth
state_unsmooth_reshape = state_unsmooth[:40000,:]
action_unsmooth_reshape = action_unsmooth[:40000,]

q_ref_noise_unsmooth = np.reshape(q_ref_noise_unsmooth,(-1,1))
q_ref_noise_unsmooth_reshape = q_ref_noise_unsmooth[:40000,:]
q_ref_unsmooth = np.reshape(q_ref_unsmooth,(-1,1))
q_ref_unsmooth_reshape = q_ref_unsmooth[:40000,:]

q_unsmooth = np.reshape(state_unsmooth_reshape[:,1],(-1,1))
q_unsmooth_reshape = q_unsmooth[:40000,:]

q_unsmooth_reshape_1417s = q_unsmooth_reshape[14000:17000,:]
q_unsmooth_reshape_1923s = q_unsmooth_reshape[19000:23000,:]
q_unsmooth_reshape_2428s = q_unsmooth_reshape[24000:28000,:]
q_unsmooth_reshape_3340s = q_unsmooth_reshape[33000:40000,:]
q_unsmooth_reshape = q_unsmooth_reshape[0:40000,:]


q_ref_noise_unsmooth_reshape_1417s = q_ref_noise_unsmooth_reshape[14000:17000,:]
q_ref_noise_unsmooth_reshape_1923s = q_ref_noise_unsmooth_reshape[19000:23000,:]
q_ref_noise_unsmooth_reshape_2428s = q_ref_noise_unsmooth_reshape[24000:28000,:]
q_ref_noise_unsmooth_reshape_3340s = q_ref_noise_unsmooth_reshape[33000:40000,:]
q_ref_noise_unsmooth_reshape = q_ref_noise_unsmooth_reshape[0:40000,:]

q_ref_unsmooth_reshape_1417s = q_ref_unsmooth_reshape[14000:17000,:]
q_ref_unsmooth_reshape_1922s = q_ref_unsmooth_reshape[19000:22000,:]
q_ref_unsmooth_reshape_2428s = q_ref_unsmooth_reshape[24000:28000,:]
q_ref_unsmooth_reshape_3340s = q_ref_unsmooth_reshape[33000:40000,:]
q_ref_unsmooth_reshape = q_ref_unsmooth_reshape[0:40000,:]

#action
action_unsmooth = np.reshape(action_unsmooth_reshape[:,1],(-1,1))
action_unsmooth_reshape_1317s = action_unsmooth[13000:17000,:]
action_unsmooth_reshape_1011s = action_unsmooth[10000:11000,:]
action_unsmooth_reshape_2832s = action_unsmooth[28000:32000,:]
action_unsmooth_reshape_3238s = action_unsmooth[32000:38000,:]
action_unsmooth_reshape = action_unsmooth[0:40000,:]


#arrary to list
q_ref_noise_unsmooth_reshape_1417s_array = np.reshape(q_ref_noise_unsmooth_reshape_1417s,(3000,))
q_ref_noise_unsmooth_reshape_1923s_array = np.reshape(q_ref_noise_unsmooth_reshape_1923s,(4000,))
q_ref_noise_unsmooth_reshape_2428s_array = np.reshape(q_ref_noise_unsmooth_reshape_2428s,(4000,))
q_ref_noise_unsmooth_reshape_3340s_array = np.reshape(q_ref_noise_unsmooth_reshape_3340s,(7000,))
q_ref_noise_unsmooth_reshape_array = np.reshape(q_ref_noise_unsmooth_reshape,(40000,))

action_unsmooth_reshape_1317s_array = np.reshape(action_unsmooth_reshape_1317s,(4000,))
action_unsmooth_reshape_1011s_array = np.reshape(action_unsmooth_reshape_1011s,(1000,))
action_unsmooth_reshape_2832s_array = np.reshape(action_unsmooth_reshape_2832s,(4000,))
action_unsmooth_reshape_3238s_array = np.reshape(action_unsmooth_reshape_3238s,(6000,))
action_unsmooth_reshape_array = np.reshape(action_unsmooth_reshape,(40000,))

#delta q/action
q_ref_noise_unsmooth_reshape_with0 = np.reshape(np.insert(q_ref_noise_unsmooth_reshape,0,[0]),[40001,1])[:40000,:]
delta_q_ref_noise_unsmooth_reshape = np.abs(q_ref_noise_unsmooth_reshape - q_ref_noise_unsmooth_reshape_with0)

action_unsmooth_with0 = np.reshape(np.insert(action_unsmooth,0,[0]),[40001,1])[:40000,:]
delta_action_unsmooth = np.abs(action_unsmooth - action_unsmooth_with0)

#FFT
#Transformation
q_ref_noise_unsmooth_reshape_1417s_array_FFT = np.fft.fft(q_ref_noise_unsmooth_reshape_1417s_array)
q_ref_noise_unsmooth_reshape_1923s_array_FFT = np.fft.fft(q_ref_noise_unsmooth_reshape_1923s_array)
q_ref_noise_unsmooth_reshape_2428s_array_FFT = np.fft.fft(q_ref_noise_unsmooth_reshape_2428s_array)
q_ref_noise_unsmooth_reshape_3340s_array_FFT = np.fft.fft(q_ref_noise_unsmooth_reshape_3340s_array)
q_ref_noise_unsmooth_reshape_array_FFT = np.fft.fft(q_ref_noise_unsmooth_reshape_array)


action_unsmooth_reshape_1317s_array_FFT = np.fft.fft(action_unsmooth_reshape_1317s_array)
action_unsmooth_reshape_1011s_array_FFT = np.fft.fft(action_unsmooth_reshape_1011s_array)
action_unsmooth_reshape_2832s_array_FFT = np.fft.fft(action_unsmooth_reshape_2832s_array)
action_unsmooth_reshape_3238s_array_FFT = np.fft.fft(action_unsmooth_reshape_3238s_array)
action_unsmooth_reshape_array_FFT = np.fft.fft(action_unsmooth_reshape_array)

#sample number
n_q_1417s_unsmooth = q_ref_noise_unsmooth_reshape_1417s_array_FFT.size
n_q_1923s_unsmooth = q_ref_noise_unsmooth_reshape_1923s_array_FFT.size
n_q_2428s_unsmooth = q_ref_noise_unsmooth_reshape_2428s_array_FFT.size
n_q_3340s_unsmooth = q_ref_noise_unsmooth_reshape_3340s_array_FFT.size
n_q_unsmooth = q_ref_noise_unsmooth_reshape_array_FFT.size

n_action_1317s_unsmooth = action_unsmooth_reshape_1317s_array_FFT.size
n_action_1011s_unsmooth = action_unsmooth_reshape_1011s_array_FFT.size
n_action_2832s_unsmooth = action_unsmooth_reshape_2832s_array_FFT.size
n_action_3238s_unsmooth = action_unsmooth_reshape_3238s_array_FFT.size
n_action_unsmooth = action_unsmooth_reshape_array_FFT.size

#frequency
freq_q_1417s_unsmooth = np.fft.fftfreq(n_q_1417s_unsmooth, 0.001)
freq_q_1923s_unsmooth = np.fft.fftfreq(n_q_1923s_unsmooth, 0.001)
freq_q_2428s_unsmooth = np.fft.fftfreq(n_q_2428s_unsmooth, 0.001)
freq_q_3340s_unsmooth = np.fft.fftfreq(n_q_3340s_unsmooth, 0.001)
freq_q_unsmooth = np.fft.fftfreq(n_q_unsmooth, 0.001)

freq_action_1317s_unsmooth = np.fft.fftfreq(n_action_1317s_unsmooth, 0.001)
freq_action_1011s_unsmooth = np.fft.fftfreq(n_action_1011s_unsmooth, 0.001)
freq_action_2832s_unsmooth = np.fft.fftfreq(n_action_2832s_unsmooth, 0.001)
freq_action_3238s_unsmooth = np.fft.fftfreq(n_action_3238s_unsmooth, 0.001)
freq_action_unsmooth = np.fft.fftfreq(n_action_unsmooth, 0.001)

#half of data
n_q_1417s_unsmooth_half = n_q_1417s_unsmooth//2
n_q_1923s_unsmooth_half = n_q_1923s_unsmooth//2
n_q_2428s_unsmooth_half = n_q_2428s_unsmooth//2
n_q_3340s_unsmooth_half = n_q_3340s_unsmooth//2
n_q_unsmooth_half = n_q_unsmooth//2

n_action_1317s_unsmooth_half = n_action_1317s_unsmooth//2
n_action_1011s_unsmooth_half = n_action_1011s_unsmooth//2
n_action_2832s_unsmooth_half = n_action_2832s_unsmooth//2
n_action_3238s_unsmooth_half = n_action_3238s_unsmooth//2
n_action_unsmooth_half = n_action_unsmooth//2

freq_q_1417s_unsmooth_half = freq_q_1417s_unsmooth[:n_q_1417s_unsmooth_half]
freq_q_1923s_unsmooth_half = freq_q_1923s_unsmooth[:n_q_1923s_unsmooth_half]
freq_q_2428s_unsmooth_half = freq_q_2428s_unsmooth[:n_q_2428s_unsmooth_half]
freq_q_3340s_unsmooth_half = freq_q_3340s_unsmooth[:n_q_3340s_unsmooth_half]
freq_q_unsmooth_half = freq_q_unsmooth[:n_q_unsmooth_half]

freq_action_1317s_unsmooth_half = freq_action_1317s_unsmooth[:n_action_1317s_unsmooth_half]
freq_action_1011s_unsmooth_half = freq_action_1011s_unsmooth[:n_action_1011s_unsmooth_half]
freq_action_2832s_unsmooth_half = freq_action_2832s_unsmooth[:n_action_2832s_unsmooth_half]
freq_action_3238s_unsmooth_half = freq_action_3238s_unsmooth[:n_action_3238s_unsmooth_half]
freq_action_unsmooth_half = freq_action_unsmooth[:n_action_unsmooth_half]

q_ref_noise_unsmooth_reshape_1417s_array_FFT_half = q_ref_noise_unsmooth_reshape_1417s_array_FFT[:n_q_1417s_unsmooth_half]
q_ref_noise_unsmooth_reshape_1923s_array_FFT_half = q_ref_noise_unsmooth_reshape_1923s_array_FFT[:n_q_1923s_unsmooth_half]
q_ref_noise_unsmooth_reshape_2428s_array_FFT_half = q_ref_noise_unsmooth_reshape_2428s_array_FFT[:n_q_2428s_unsmooth_half]
q_ref_noise_unsmooth_reshape_3340s_array_FFT_half = q_ref_noise_unsmooth_reshape_3340s_array_FFT[:n_q_3340s_unsmooth_half]
q_ref_noise_unsmooth_reshape_array_FFT_half = q_ref_noise_unsmooth_reshape_array_FFT[:n_q_unsmooth_half]

action_ref_noise_unsmooth_reshape_1317s_array_FFT_half = action_unsmooth_reshape_1317s_array_FFT[:n_action_1317s_unsmooth_half]
action_ref_noise_unsmooth_reshape_1011s_array_FFT_half = action_unsmooth_reshape_1011s_array_FFT[:n_action_1011s_unsmooth_half]
action_ref_noise_unsmooth_reshape_2832s_array_FFT_half = action_unsmooth_reshape_2832s_array_FFT[:n_action_2832s_unsmooth_half]
action_ref_noise_unsmooth_reshape_3238s_array_FFT_half = action_unsmooth_reshape_3238s_array_FFT[:n_action_3238s_unsmooth_half]
action_ref_noise_unsmooth_reshape_array_FFT_half = action_unsmooth_reshape_array_FFT[:n_action_unsmooth_half]

#amplitude modification
q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified = np.concatenate(([q_ref_noise_unsmooth_reshape_1417s_array_FFT_half[0]/n_q_1417s_unsmooth], q_ref_noise_unsmooth_reshape_1417s_array_FFT_half[1:-1]*2/n_q_1417s_unsmooth, [q_ref_noise_unsmooth_reshape_1417s_array_FFT_half[-1]/n_q_1417s_unsmooth]))
q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified_abs = np.abs(q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified)

q_ref_noise_unsmooth_reshape_1923s_array_FFT_half_modified = np.concatenate(([q_ref_noise_unsmooth_reshape_1923s_array_FFT_half[0]/n_q_1923s_unsmooth], q_ref_noise_unsmooth_reshape_1923s_array_FFT_half[1:-1]*2/n_q_1923s_unsmooth, [q_ref_noise_unsmooth_reshape_1923s_array_FFT_half[-1]/n_q_1923s_unsmooth]))
q_ref_noise_unsmooth_reshape_1923s_array_FFT_half_modified_abs = np.abs(q_ref_noise_unsmooth_reshape_1923s_array_FFT_half_modified)

q_ref_noise_unsmooth_reshape_2428s_array_FFT_half_modified = np.concatenate(([q_ref_noise_unsmooth_reshape_2428s_array_FFT_half[0]/n_q_2428s_unsmooth], q_ref_noise_unsmooth_reshape_2428s_array_FFT_half[1:-1]*2/n_q_2428s_unsmooth, [q_ref_noise_unsmooth_reshape_2428s_array_FFT_half[-1]/n_q_2428s_unsmooth]))
q_ref_noise_unsmooth_reshape_2428s_array_FFT_half_modified_abs = np.abs(q_ref_noise_unsmooth_reshape_2428s_array_FFT_half_modified)

q_ref_noise_unsmooth_reshape_3340s_array_FFT_half_modified = np.concatenate(([q_ref_noise_unsmooth_reshape_3340s_array_FFT_half[0]/n_q_3340s_unsmooth], q_ref_noise_unsmooth_reshape_3340s_array_FFT_half[1:-1]*2/n_q_3340s_unsmooth, [q_ref_noise_unsmooth_reshape_3340s_array_FFT_half[-1]/n_q_3340s_unsmooth]))
q_ref_noise_unsmooth_reshape_3340s_array_FFT_half_modified_abs = np.abs(q_ref_noise_unsmooth_reshape_3340s_array_FFT_half_modified)

q_ref_noise_unsmooth_reshape_array_FFT_half_modified = np.concatenate(([q_ref_noise_unsmooth_reshape_array_FFT_half[0]/n_q_unsmooth], q_ref_noise_unsmooth_reshape_array_FFT_half[1:-1]*2/n_q_unsmooth, [q_ref_noise_unsmooth_reshape_array_FFT_half[-1]/n_q_unsmooth]))
q_ref_noise_unsmooth_reshape_array_FFT_half_modified_abs = np.abs(q_ref_noise_unsmooth_reshape_array_FFT_half_modified)

action_ref_noise_unsmooth_reshape_1317s_array_FFT_half_modified = np.concatenate(([action_ref_noise_unsmooth_reshape_1317s_array_FFT_half[0]/n_action_1317s_unsmooth], action_ref_noise_unsmooth_reshape_1317s_array_FFT_half[1:-1]*2/n_action_1317s_unsmooth, [action_ref_noise_unsmooth_reshape_1317s_array_FFT_half[-1]/n_action_1317s_unsmooth]))
action_ref_noise_unsmooth_reshape_1317s_array_FFT_half_modified_abs = np.abs(action_ref_noise_unsmooth_reshape_1317s_array_FFT_half_modified)

action_ref_noise_unsmooth_reshape_1011s_array_FFT_half_modified = np.concatenate(([action_ref_noise_unsmooth_reshape_1011s_array_FFT_half[0]/n_action_1011s_unsmooth], action_ref_noise_unsmooth_reshape_1011s_array_FFT_half[1:-1]*2/n_action_1011s_unsmooth, [action_ref_noise_unsmooth_reshape_1011s_array_FFT_half[-1]/n_action_1011s_unsmooth]))
action_ref_noise_unsmooth_reshape_1011s_array_FFT_half_modified_abs = np.abs(action_ref_noise_unsmooth_reshape_1011s_array_FFT_half_modified)

action_ref_noise_unsmooth_reshape_2832s_array_FFT_half_modified = np.concatenate(([action_ref_noise_unsmooth_reshape_2832s_array_FFT_half[0]/n_action_2832s_unsmooth], action_ref_noise_unsmooth_reshape_2832s_array_FFT_half[1:-1]*2/n_action_2832s_unsmooth, [action_ref_noise_unsmooth_reshape_2832s_array_FFT_half[-1]/n_action_2832s_unsmooth]))
action_ref_noise_unsmooth_reshape_2832s_array_FFT_half_modified_abs = np.abs(action_ref_noise_unsmooth_reshape_2832s_array_FFT_half_modified)

action_ref_noise_unsmooth_reshape_3238s_array_FFT_half_modified = np.concatenate(([action_ref_noise_unsmooth_reshape_3238s_array_FFT_half[0]/n_action_3238s_unsmooth], action_ref_noise_unsmooth_reshape_3238s_array_FFT_half[1:-1]*2/n_action_3238s_unsmooth, [action_ref_noise_unsmooth_reshape_3238s_array_FFT_half[-1]/n_action_3238s_unsmooth]))
action_ref_noise_unsmooth_reshape_3238s_array_FFT_half_modified_abs = np.abs(action_ref_noise_unsmooth_reshape_3238s_array_FFT_half_modified)


action_ref_noise_unsmooth_reshape_array_FFT_half_modified = np.concatenate(([action_ref_noise_unsmooth_reshape_array_FFT_half[0]/n_action_unsmooth], action_ref_noise_unsmooth_reshape_array_FFT_half[1:-1]*2/n_action_unsmooth, [action_ref_noise_unsmooth_reshape_array_FFT_half[-1]/n_action_unsmooth]))
action_ref_noise_unsmooth_reshape_array_FFT_half_modified_abs = np.abs(action_ref_noise_unsmooth_reshape_array_FFT_half_modified)

#smooth
state_smooth_reshape = state_smooth[:40000,:]
action_smooth_reshape = action_smooth[:40000,]
q_ref_noise_smooth = np.reshape(q_ref_noise_smooth,(-1,1))
q_ref_noise_smooth_reshape = q_ref_noise_smooth[:40000,:]
q_ref_smooth = np.reshape(q_ref_smooth,(-1,1))
q_ref_smooth_reshape = q_ref_smooth[:40000,:]

q_smooth = np.reshape(state_smooth_reshape[:,1],(-1,1))
q_smooth_reshape = q_smooth[:40000,:]

q_smooth_reshape_1417s = q_smooth_reshape[14000:17000,:]
q_smooth_reshape_1923s = q_smooth_reshape[19000:23000,:]
q_smooth_reshape_2428s = q_smooth_reshape[24000:28000,:]
q_smooth_reshape_3340s = q_smooth_reshape[33000:40000,:]
q_smooth_reshape = q_smooth_reshape[0:40000,:]

q_ref_noise_smooth_reshape_1417s = q_ref_noise_smooth_reshape[14000:17000,:]
q_ref_noise_smooth_reshape_1923s = q_ref_noise_smooth_reshape[19000:23000,:]
q_ref_noise_smooth_reshape_2428s = q_ref_noise_smooth_reshape[24000:28000,:]
q_ref_noise_smooth_reshape_3340s = q_ref_noise_smooth_reshape[33000:40000,:]
q_ref_noise_smooth_reshape = q_ref_noise_smooth_reshape[0:40000,:]

q_ref_smooth_reshape_1417s = q_ref_smooth_reshape[14000:17000,:]
q_ref_smooth_reshape_1923s = q_ref_smooth_reshape[19000:23000,:]
q_ref_smooth_reshape_2428s = q_ref_smooth_reshape[24000:28000,:]
q_ref_smooth_reshape_3340s = q_ref_smooth_reshape[33000:40000,:]
q_ref_smooth_reshape = q_ref_smooth_reshape[0:40000,:]

#action
action_smooth = np.reshape(action_smooth_reshape[:,1],(-1,1))
action_smooth_reshape_1317s = action_smooth[13000:17000,:]
action_smooth_reshape_1011s = action_smooth[10000:11000,:]
action_smooth_reshape_2832s = action_smooth[28000:32000,:]
action_smooth_reshape_3238s = action_smooth[32000:38000,:]
action_smooth_reshape = action_smooth[0:40000,:]

#arrary to list
q_ref_smooth_reshape_1417s_array = np.reshape(q_ref_smooth_reshape_1417s,(3000,))
q_ref_smooth_reshape_1923s_array = np.reshape(q_ref_smooth_reshape_1923s,(4000,))
q_ref_smooth_reshape_2428s_array = np.reshape(q_ref_smooth_reshape_2428s,(4000,))
q_ref_smooth_reshape_3340s_array = np.reshape(q_ref_smooth_reshape_3340s,(7000,))
q_ref_smooth_reshape_array = np.reshape(q_ref_smooth_reshape,(40000,))

action_smooth_reshape_1317s_array = np.reshape(action_smooth_reshape_1317s,(4000,))
action_smooth_reshape_1011s_array = np.reshape(action_smooth_reshape_1011s,(1000,))
action_smooth_reshape_2832s_array = np.reshape(action_smooth_reshape_2832s,(4000,))
action_smooth_reshape_3238s_array = np.reshape(action_smooth_reshape_3238s,(6000,))
action_smooth_reshape_array = np.reshape(action_smooth_reshape,(40000,))

#delta q/action
q_ref_noise_smooth_reshape_with0 = np.reshape(np.insert(q_ref_noise_smooth_reshape,0,[0]),[40001,1])[:40000,:]
delta_q_ref_noise_smooth_reshape = np.abs(q_ref_noise_smooth_reshape - q_ref_noise_smooth_reshape_with0)

action_smooth_with0 = np.reshape(np.insert(action_smooth,0,[0]),[40001,1])[:40000,:]
delta_action_smooth = np.abs(action_smooth - action_smooth_with0)


alpha_ref_history = np.reshape(alpha_ref_history,(-1,1))
alpha_ref_history_reshape = alpha_ref_history[:40000,:]


#FFT
#Transformation
q_ref_smooth_reshape_1417s_array_FFT = np.fft.fft(q_ref_smooth_reshape_1417s_array)
q_ref_smooth_reshape_1923s_array_FFT = np.fft.fft(q_ref_smooth_reshape_1923s_array)
q_ref_smooth_reshape_2428s_array_FFT = np.fft.fft(q_ref_smooth_reshape_2428s_array)
q_ref_smooth_reshape_3340s_array_FFT = np.fft.fft(q_ref_smooth_reshape_3340s_array)
q_ref_smooth_reshape_array_FFT = np.fft.fft(q_ref_smooth_reshape_array)

action_smooth_reshape_1317s_array_FFT = np.fft.fft(action_smooth_reshape_1317s_array)
action_smooth_reshape_1011s_array_FFT = np.fft.fft(action_smooth_reshape_1011s_array)
action_smooth_reshape_2832s_array_FFT = np.fft.fft(action_smooth_reshape_2832s_array)
action_smooth_reshape_3238s_array_FFT = np.fft.fft(action_smooth_reshape_3238s_array)
action_smooth_reshape_array_FFT = np.fft.fft(action_smooth_reshape_array)

#sample number
n_q_1417s_smooth = q_ref_smooth_reshape_1417s_array_FFT.size
n_q_1923s_smooth = q_ref_smooth_reshape_1923s_array_FFT.size
n_q_2428s_smooth = q_ref_smooth_reshape_2428s_array_FFT.size
n_q_3340s_smooth = q_ref_smooth_reshape_3340s_array_FFT.size
n_q_smooth = q_ref_smooth_reshape_array_FFT.size

n_action_1317s_smooth = action_smooth_reshape_1317s_array_FFT.size
n_action_1011s_smooth = action_smooth_reshape_1011s_array_FFT.size
n_action_2832s_smooth = action_smooth_reshape_2832s_array_FFT.size
n_action_3238s_smooth = action_smooth_reshape_3238s_array_FFT.size
n_action_smooth = action_smooth_reshape_array_FFT.size

#frequency
freq_q_1417s_smooth = np.fft.fftfreq(n_q_1417s_smooth, 0.001)
freq_q_1923s_smooth = np.fft.fftfreq(n_q_1923s_smooth, 0.001)
freq_q_2428s_smooth = np.fft.fftfreq(n_q_2428s_smooth, 0.001)
freq_q_3340s_smooth = np.fft.fftfreq(n_q_3340s_smooth, 0.001)
freq_q_smooth = np.fft.fftfreq(n_q_smooth, 0.001)

freq_action_1317s_smooth = np.fft.fftfreq(n_action_1317s_smooth, 0.001)
freq_action_1011s_smooth = np.fft.fftfreq(n_action_1011s_smooth, 0.001)
freq_action_2832s_smooth = np.fft.fftfreq(n_action_2832s_smooth, 0.001)
freq_action_3238s_smooth = np.fft.fftfreq(n_action_3238s_smooth, 0.001)
freq_action_smooth = np.fft.fftfreq(n_action_smooth, 0.001)

#half of data
n_q_1417s_smooth_half = n_q_1417s_smooth//2
n_q_1923s_smooth_half = n_q_1923s_smooth//2
n_q_2428s_smooth_half = n_q_2428s_smooth//2
n_q_3340s_smooth_half = n_q_3340s_smooth//2
n_q_smooth_half = n_q_smooth//2

n_action_1317s_smooth_half = n_action_1317s_smooth//2
n_action_1011s_smooth_half = n_action_1011s_smooth//2
n_action_2832s_smooth_half = n_action_2832s_smooth//2
n_action_3238s_smooth_half = n_action_3238s_smooth//2
n_action_smooth_half = n_action_smooth//2

freq_q_1417s_smooth_half = freq_q_1417s_smooth[:n_q_1417s_smooth_half]
freq_q_1923s_smooth_half = freq_q_1923s_smooth[:n_q_1923s_smooth_half]
freq_q_2428s_smooth_half = freq_q_2428s_smooth[:n_q_2428s_smooth_half]
freq_q_3340s_smooth_half = freq_q_3340s_smooth[:n_q_3340s_smooth_half]
freq_q_smooth_half = freq_q_smooth[:n_q_smooth_half]

freq_action_1317s_smooth_half = freq_action_1317s_smooth[:n_action_1317s_smooth_half]
freq_action_1011s_smooth_half = freq_action_1011s_smooth[:n_action_1011s_smooth_half]
freq_action_2832s_smooth_half = freq_action_2832s_smooth[:n_action_2832s_smooth_half]
freq_action_3238s_smooth_half = freq_action_3238s_smooth[:n_action_3238s_smooth_half]
freq_action_smooth_half = freq_action_smooth[:n_action_smooth_half]

q_ref_smooth_reshape_1417s_array_FFT_half = q_ref_smooth_reshape_1417s_array_FFT[:n_q_1417s_smooth_half]
q_ref_smooth_reshape_1923s_array_FFT_half = q_ref_smooth_reshape_1923s_array_FFT[:n_q_1923s_smooth_half]
q_ref_smooth_reshape_2428s_array_FFT_half = q_ref_smooth_reshape_2428s_array_FFT[:n_q_2428s_smooth_half]
q_ref_smooth_reshape_3340s_array_FFT_half = q_ref_smooth_reshape_3340s_array_FFT[:n_q_3340s_smooth_half]
q_ref_smooth_reshape_array_FFT_half = q_ref_smooth_reshape_array_FFT[:n_q_smooth_half]

action_ref_smooth_reshape_1317s_array_FFT_half = action_smooth_reshape_1317s_array_FFT[:n_action_1317s_smooth_half]
action_ref_smooth_reshape_1011s_array_FFT_half = action_smooth_reshape_1011s_array_FFT[:n_action_1011s_smooth_half]
action_ref_smooth_reshape_2832s_array_FFT_half = action_smooth_reshape_2832s_array_FFT[:n_action_2832s_smooth_half]
action_ref_smooth_reshape_3238s_array_FFT_half = action_smooth_reshape_3238s_array_FFT[:n_action_3238s_smooth_half]
action_ref_smooth_reshape_array_FFT_half = action_smooth_reshape_array_FFT[:n_action_smooth_half]

#amplitude modification
q_ref_smooth_reshape_1417s_array_FFT_half_modified = np.concatenate(([q_ref_smooth_reshape_1417s_array_FFT_half[0]/n_q_1417s_smooth], q_ref_smooth_reshape_1417s_array_FFT_half[1:-1]*2/n_q_1417s_smooth, [q_ref_smooth_reshape_1417s_array_FFT_half[-1]/n_q_1417s_smooth]))
q_ref_smooth_reshape_1417s_array_FFT_half_modified_abs = np.abs(q_ref_smooth_reshape_1417s_array_FFT_half_modified)

q_ref_smooth_reshape_1923s_array_FFT_half_modified = np.concatenate(([q_ref_smooth_reshape_1923s_array_FFT_half[0]/n_q_1923s_smooth], q_ref_smooth_reshape_1923s_array_FFT_half[1:-1]*2/n_q_1923s_smooth, [q_ref_smooth_reshape_1923s_array_FFT_half[-1]/n_q_1923s_smooth]))
q_ref_smooth_reshape_1923s_array_FFT_half_modified_abs = np.abs(q_ref_smooth_reshape_1923s_array_FFT_half_modified)

q_ref_smooth_reshape_2428s_array_FFT_half_modified = np.concatenate(([q_ref_smooth_reshape_2428s_array_FFT_half[0]/n_q_2428s_smooth], q_ref_smooth_reshape_2428s_array_FFT_half[1:-1]*2/n_q_2428s_smooth, [q_ref_smooth_reshape_2428s_array_FFT_half[-1]/n_q_2428s_smooth]))
q_ref_smooth_reshape_2428s_array_FFT_half_modified_abs = np.abs(q_ref_smooth_reshape_2428s_array_FFT_half_modified)

q_ref_smooth_reshape_3340s_array_FFT_half_modified = np.concatenate(([q_ref_smooth_reshape_3340s_array_FFT_half[0]/n_q_3340s_smooth], q_ref_smooth_reshape_3340s_array_FFT_half[1:-1]*2/n_q_3340s_smooth, [q_ref_smooth_reshape_3340s_array_FFT_half[-1]/n_q_3340s_smooth]))
q_ref_smooth_reshape_3340s_array_FFT_half_modified_abs = np.abs(q_ref_smooth_reshape_3340s_array_FFT_half_modified)

q_ref_smooth_reshape_array_FFT_half_modified = np.concatenate(([q_ref_smooth_reshape_array_FFT_half[0]/n_q_smooth], q_ref_smooth_reshape_array_FFT_half[1:-1]*2/n_q_smooth, [q_ref_smooth_reshape_array_FFT_half[-1]/n_q_smooth]))
q_ref_smooth_reshape_array_FFT_half_modified_abs = np.abs(q_ref_smooth_reshape_array_FFT_half_modified)

action_ref_smooth_reshape_1317s_array_FFT_half_modified = np.concatenate(([action_ref_smooth_reshape_1317s_array_FFT_half[0]/n_action_1317s_smooth], action_ref_smooth_reshape_1317s_array_FFT_half[1:-1]*2/n_action_1317s_smooth, [action_ref_smooth_reshape_1317s_array_FFT_half[-1]/n_action_1317s_smooth]))
action_ref_smooth_reshape_1317s_array_FFT_half_modified_abs = np.abs(action_ref_smooth_reshape_1317s_array_FFT_half_modified)

action_ref_smooth_reshape_1011s_array_FFT_half_modified = np.concatenate(([action_ref_smooth_reshape_1011s_array_FFT_half[0]/n_action_1011s_smooth], action_ref_smooth_reshape_1011s_array_FFT_half[1:-1]*2/n_action_1011s_smooth, [action_ref_smooth_reshape_1011s_array_FFT_half[-1]/n_action_1011s_smooth]))
action_ref_smooth_reshape_1011s_array_FFT_half_modified_abs = np.abs(action_ref_smooth_reshape_1011s_array_FFT_half_modified)

action_ref_smooth_reshape_2832s_array_FFT_half_modified = np.concatenate(([action_ref_smooth_reshape_2832s_array_FFT_half[0]/n_action_2832s_smooth], action_ref_smooth_reshape_2832s_array_FFT_half[1:-1]*2/n_action_2832s_smooth, [action_ref_smooth_reshape_2832s_array_FFT_half[-1]/n_action_2832s_smooth]))
action_ref_smooth_reshape_2832s_array_FFT_half_modified_abs = np.abs(action_ref_smooth_reshape_2832s_array_FFT_half_modified)

action_ref_smooth_reshape_3238s_array_FFT_half_modified = np.concatenate(([action_ref_smooth_reshape_3238s_array_FFT_half[0]/n_action_3238s_smooth], action_ref_smooth_reshape_3238s_array_FFT_half[1:-1]*2/n_action_3238s_smooth, [action_ref_smooth_reshape_3238s_array_FFT_half[-1]/n_action_3238s_smooth]))
action_ref_smooth_reshape_3238s_array_FFT_half_modified_abs = np.abs(action_ref_smooth_reshape_3238s_array_FFT_half_modified)

action_ref_smooth_reshape_array_FFT_half_modified = np.concatenate(([action_ref_smooth_reshape_array_FFT_half[0]/n_action_smooth], action_ref_smooth_reshape_array_FFT_half[1:-1]*2/n_action_smooth, [action_ref_smooth_reshape_array_FFT_half[-1]/n_action_smooth]))
action_ref_smooth_reshape_array_FFT_half_modified_abs = np.abs(action_ref_smooth_reshape_array_FFT_half_modified)

time = np.array(range(0,40000,1))


#Filter data
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

alpha_ref_history = np.reshape(alpha_ref_history,(-1,1))
alpha_ref_history_reshape = alpha_ref_history[:40000,:]

#unsmooth
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

#delta q/action
q_ref_noise_unfilter_reshape_with0 = np.reshape(np.insert(q_ref_noise_unfilter_reshape,0,[0]),[40001,1])[:40000,:]
delta_q_ref_noise_unfilter_reshape = np.abs(q_ref_noise_unfilter_reshape - q_ref_noise_unfilter_reshape_with0)

action_unfilter_with0 = np.reshape(np.insert(action_unfilter,0,[0]),[40001,1])[:40000,:]
delta_action_unfilter = np.abs(action_unfilter - action_unfilter_with0)

#FFT
#Transformation
q_ref_noise_unfilter_reshape_1417s_array_FFT = np.fft.fft(q_ref_noise_unfilter_reshape_1417s_array)
q_ref_noise_unfilter_reshape_1923s_array_FFT = np.fft.fft(q_ref_noise_unfilter_reshape_1923s_array)
q_ref_noise_unfilter_reshape_2428s_array_FFT = np.fft.fft(q_ref_noise_unfilter_reshape_2428s_array)
q_ref_noise_unfilter_reshape_3340s_array_FFT = np.fft.fft(q_ref_noise_unfilter_reshape_3340s_array)

action_unfilter_reshape_1317s_array_FFT = np.fft.fft(action_unfilter_reshape_1317s_array)
action_unfilter_reshape_1011s_array_FFT = np.fft.fft(action_unfilter_reshape_1011s_array)
action_unfilter_reshape_2832s_array_FFT = np.fft.fft(action_unfilter_reshape_2832s_array)
action_unfilter_reshape_3238s_array_FFT = np.fft.fft(action_unfilter_reshape_3238s_array)

#sample number
n_q_1417s_unfilter = q_ref_noise_unfilter_reshape_1417s_array_FFT.size
n_q_1923s_unfilter = q_ref_noise_unfilter_reshape_1923s_array_FFT.size
n_q_2428s_unfilter = q_ref_noise_unfilter_reshape_2428s_array_FFT.size
n_q_3340s_unfilter = q_ref_noise_unfilter_reshape_3340s_array_FFT.size

n_action_1317s_unfilter = action_unfilter_reshape_1317s_array_FFT.size
n_action_1011s_unfilter = action_unfilter_reshape_1011s_array_FFT.size
n_action_2832s_unfilter = action_unfilter_reshape_2832s_array_FFT.size
n_action_3238s_unfilter = action_unfilter_reshape_3238s_array_FFT.size

#frequency
freq_q_1417s_unfilter = np.fft.fftfreq(n_q_1417s_unfilter, 0.001)
freq_q_1923s_unfilter = np.fft.fftfreq(n_q_1923s_unfilter, 0.001)
freq_q_2428s_unfilter = np.fft.fftfreq(n_q_2428s_unfilter, 0.001)
freq_q_3340s_unfilter = np.fft.fftfreq(n_q_3340s_unfilter, 0.001)

freq_action_1317s_unfilter = np.fft.fftfreq(n_action_1317s_unfilter, 0.001)
freq_action_1011s_unfilter = np.fft.fftfreq(n_action_1011s_unfilter, 0.001)
freq_action_2832s_unfilter = np.fft.fftfreq(n_action_2832s_unfilter, 0.001)
freq_action_3238s_unfilter = np.fft.fftfreq(n_action_3238s_unfilter, 0.001)

#half of data
n_q_1417s_unfilter_half = n_q_1417s_unfilter//2
n_q_1923s_unfilter_half = n_q_1923s_unfilter//2
n_q_2428s_unfilter_half = n_q_2428s_unfilter//2
n_q_3340s_unfilter_half = n_q_3340s_unfilter//2

n_action_1317s_unfilter_half = n_action_1317s_unfilter//2
n_action_1011s_unfilter_half = n_action_1011s_unfilter//2
n_action_2832s_unfilter_half = n_action_2832s_unfilter//2
n_action_3238s_unfilter_half = n_action_3238s_unfilter//2

freq_q_1417s_unfilter_half = freq_q_1417s_unfilter[:n_q_1417s_unfilter_half]
freq_q_1923s_unfilter_half = freq_q_1923s_unfilter[:n_q_1923s_unfilter_half]
freq_q_2428s_unfilter_half = freq_q_2428s_unfilter[:n_q_2428s_unfilter_half]
freq_q_3340s_unfilter_half = freq_q_3340s_unfilter[:n_q_3340s_unfilter_half]

freq_action_1317s_unfilter_half = freq_action_1317s_unfilter[:n_action_1317s_unfilter_half]
freq_action_1011s_unfilter_half = freq_action_1011s_unfilter[:n_action_1011s_unfilter_half]
freq_action_2832s_unfilter_half = freq_action_2832s_unfilter[:n_action_2832s_unfilter_half]
freq_action_3238s_unfilter_half = freq_action_3238s_unfilter[:n_action_3238s_unfilter_half]

q_ref_noise_unfilter_reshape_1417s_array_FFT_half = q_ref_noise_unfilter_reshape_1417s_array_FFT[:n_q_1417s_unfilter_half]
q_ref_noise_unfilter_reshape_1923s_array_FFT_half = q_ref_noise_unfilter_reshape_1923s_array_FFT[:n_q_1923s_unfilter_half]
q_ref_noise_unfilter_reshape_2428s_array_FFT_half = q_ref_noise_unfilter_reshape_2428s_array_FFT[:n_q_2428s_unfilter_half]
q_ref_noise_unfilter_reshape_3340s_array_FFT_half = q_ref_noise_unfilter_reshape_3340s_array_FFT[:n_q_3340s_unfilter_half]

action_ref_noise_unfilter_reshape_1317s_array_FFT_half = action_unfilter_reshape_1317s_array_FFT[:n_action_1317s_unfilter_half]
action_ref_noise_unfilter_reshape_1011s_array_FFT_half = action_unfilter_reshape_1011s_array_FFT[:n_action_1011s_unfilter_half]
action_ref_noise_unfilter_reshape_2832s_array_FFT_half = action_unfilter_reshape_2832s_array_FFT[:n_action_2832s_unfilter_half]
action_ref_noise_unfilter_reshape_3238s_array_FFT_half = action_unfilter_reshape_3238s_array_FFT[:n_action_3238s_unfilter_half]

#amplitude modification
q_ref_noise_unfilter_reshape_1417s_array_FFT_half_modified = np.concatenate(([q_ref_noise_unfilter_reshape_1417s_array_FFT_half[0]/n_q_1417s_unfilter], q_ref_noise_unfilter_reshape_1417s_array_FFT_half[1:-1]*2/n_q_1417s_unfilter, [q_ref_noise_unfilter_reshape_1417s_array_FFT_half[-1]/n_q_1417s_unfilter]))
q_ref_noise_unfilter_reshape_1417s_array_FFT_half_modified_abs = np.abs(q_ref_noise_unfilter_reshape_1417s_array_FFT_half_modified)

q_ref_noise_unfilter_reshape_1923s_array_FFT_half_modified = np.concatenate(([q_ref_noise_unfilter_reshape_1923s_array_FFT_half[0]/n_q_1923s_unfilter], q_ref_noise_unfilter_reshape_1923s_array_FFT_half[1:-1]*2/n_q_1923s_unfilter, [q_ref_noise_unfilter_reshape_1923s_array_FFT_half[-1]/n_q_1923s_unfilter]))
q_ref_noise_unfilter_reshape_1923s_array_FFT_half_modified_abs = np.abs(q_ref_noise_unfilter_reshape_1923s_array_FFT_half_modified)

q_ref_noise_unfilter_reshape_2428s_array_FFT_half_modified = np.concatenate(([q_ref_noise_unfilter_reshape_2428s_array_FFT_half[0]/n_q_2428s_unfilter], q_ref_noise_unfilter_reshape_2428s_array_FFT_half[1:-1]*2/n_q_2428s_unfilter, [q_ref_noise_unfilter_reshape_2428s_array_FFT_half[-1]/n_q_2428s_unfilter]))
q_ref_noise_unfilter_reshape_2428s_array_FFT_half_modified_abs = np.abs(q_ref_noise_unfilter_reshape_2428s_array_FFT_half_modified)

q_ref_noise_unfilter_reshape_3340s_array_FFT_half_modified = np.concatenate(([q_ref_noise_unfilter_reshape_3340s_array_FFT_half[0]/n_q_3340s_unfilter], q_ref_noise_unfilter_reshape_3340s_array_FFT_half[1:-1]*2/n_q_3340s_unfilter, [q_ref_noise_unfilter_reshape_3340s_array_FFT_half[-1]/n_q_3340s_unfilter]))
q_ref_noise_unfilter_reshape_3340s_array_FFT_half_modified_abs = np.abs(q_ref_noise_unfilter_reshape_3340s_array_FFT_half_modified)

action_ref_noise_unfilter_reshape_1317s_array_FFT_half_modified = np.concatenate(([action_ref_noise_unfilter_reshape_1317s_array_FFT_half[0]/n_action_1317s_unfilter], action_ref_noise_unfilter_reshape_1317s_array_FFT_half[1:-1]*2/n_action_1317s_unfilter, [action_ref_noise_unfilter_reshape_1317s_array_FFT_half[-1]/n_action_1317s_unfilter]))
action_ref_noise_unfilter_reshape_1317s_array_FFT_half_modified_abs = np.abs(action_ref_noise_unfilter_reshape_1317s_array_FFT_half_modified)

action_ref_noise_unfilter_reshape_1011s_array_FFT_half_modified = np.concatenate(([action_ref_noise_unfilter_reshape_1011s_array_FFT_half[0]/n_action_1011s_unfilter], action_ref_noise_unfilter_reshape_1011s_array_FFT_half[1:-1]*2/n_action_1011s_unfilter, [action_ref_noise_unfilter_reshape_1011s_array_FFT_half[-1]/n_action_1011s_unfilter]))
action_ref_noise_unfilter_reshape_1011s_array_FFT_half_modified_abs = np.abs(action_ref_noise_unfilter_reshape_1011s_array_FFT_half_modified)

action_ref_noise_unfilter_reshape_2832s_array_FFT_half_modified = np.concatenate(([action_ref_noise_unfilter_reshape_2832s_array_FFT_half[0]/n_action_2832s_unfilter], action_ref_noise_unfilter_reshape_2832s_array_FFT_half[1:-1]*2/n_action_2832s_unfilter, [action_ref_noise_unfilter_reshape_2832s_array_FFT_half[-1]/n_action_2832s_unfilter]))
action_ref_noise_unfilter_reshape_2832s_array_FFT_half_modified_abs = np.abs(action_ref_noise_unfilter_reshape_2832s_array_FFT_half_modified)

action_ref_noise_unfilter_reshape_3238s_array_FFT_half_modified = np.concatenate(([action_ref_noise_unfilter_reshape_3238s_array_FFT_half[0]/n_action_3238s_unfilter], action_ref_noise_unfilter_reshape_3238s_array_FFT_half[1:-1]*2/n_action_3238s_unfilter, [action_ref_noise_unfilter_reshape_3238s_array_FFT_half[-1]/n_action_3238s_unfilter]))
action_ref_noise_unfilter_reshape_3238s_array_FFT_half_modified_abs = np.abs(action_ref_noise_unfilter_reshape_3238s_array_FFT_half_modified)

#smooth
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
q_filter_reshape = q_filter_reshape[0:40000,:]

q_ref_noise_filter_reshape_1417s = q_ref_noise_filter_reshape[14000:17000,:]
q_ref_noise_filter_reshape_1923s = q_ref_noise_filter_reshape[19000:23000,:]
q_ref_noise_filter_reshape_2428s = q_ref_noise_filter_reshape[24000:28000,:]
q_ref_noise_filter_reshape_3340s = q_ref_noise_filter_reshape[33000:40000,:]
q_ref_noise_filter_reshape = q_ref_noise_filter_reshape[0:40000,:]

q_ref_filter_reshape_1417s = q_ref_filter_reshape[14000:17000,:]
q_ref_filter_reshape_1923s = q_ref_filter_reshape[19000:23000,:]
q_ref_filter_reshape_2428s = q_ref_filter_reshape[24000:28000,:]
q_ref_filter_reshape_3340s = q_ref_filter_reshape[33000:40000,:]
q_ref_filter_reshape = q_ref_filter_reshape

#action
action_filter = np.reshape(action_filter_reshape[:,1],(-1,1))
action_filter_reshape_1317s = action_filter[13000:17000,:]
action_filter_reshape_1011s = action_filter[10000:11000,:]
action_filter_reshape_2832s = action_filter[28000:32000,:]
action_filter_reshape_3238s = action_filter[32000:38000,:]
action_filter_reshape = action_filter[0:40000,:]

#arrary to list
q_ref_filter_reshape_1417s_array = np.reshape(q_ref_filter_reshape_1417s,(3000,))
q_ref_filter_reshape_1923s_array = np.reshape(q_ref_filter_reshape_1923s,(4000,))
q_ref_filter_reshape_2428s_array = np.reshape(q_ref_filter_reshape_2428s,(4000,))
q_ref_filter_reshape_3340s_array = np.reshape(q_ref_filter_reshape_3340s,(7000,))
q_ref_filter_reshape_array = np.reshape(q_ref_filter_reshape,(40000,))

action_filter_reshape_1317s_array = np.reshape(action_filter_reshape_1317s,(4000,))
action_filter_reshape_1011s_array = np.reshape(action_filter_reshape_1011s,(1000,))
action_filter_reshape_2832s_array = np.reshape(action_filter_reshape_2832s,(4000,))
action_filter_reshape_3238s_array = np.reshape(action_filter_reshape_3238s,(6000,))
action_filter_reshape_array = np.reshape(action_filter_reshape,(40000,))

#delta q/action
q_ref_noise_filter_reshape_with0 = np.reshape(np.insert(q_ref_noise_filter_reshape,0,[0]),[40001,1])[:40000,:]
delta_q_ref_noise_filter_reshape = np.abs(q_ref_noise_filter_reshape - q_ref_noise_filter_reshape_with0)

action_filter_with0 = np.reshape(np.insert(action_filter,0,[0]),[40001,1])[:40000,:]
delta_action_filter = np.abs(action_filter - action_filter_with0)


#alpha_ref_history = np.reshape(alpha_ref_history,(-1,1))
#alpha_ref_history_reshape = alpha_ref_history[:40000,:]


#FFT
#Transformation
q_ref_filter_reshape_1417s_array_FFT = np.fft.fft(q_ref_filter_reshape_1417s_array)
q_ref_filter_reshape_1923s_array_FFT = np.fft.fft(q_ref_filter_reshape_1923s_array)
q_ref_filter_reshape_2428s_array_FFT = np.fft.fft(q_ref_filter_reshape_2428s_array)
q_ref_filter_reshape_3340s_array_FFT = np.fft.fft(q_ref_filter_reshape_3340s_array)
q_ref_filter_reshape_array_FFT = np.fft.fft(q_ref_filter_reshape_array)


action_filter_reshape_1317s_array_FFT = np.fft.fft(action_filter_reshape_1317s_array)
action_filter_reshape_1011s_array_FFT = np.fft.fft(action_filter_reshape_1011s_array)
action_filter_reshape_2832s_array_FFT = np.fft.fft(action_filter_reshape_2832s_array)
action_filter_reshape_3238s_array_FFT = np.fft.fft(action_filter_reshape_3238s_array)
action_filter_reshape_array_FFT = np.fft.fft(action_filter_reshape_array)

#sample number
n_q_1417s_filter = q_ref_filter_reshape_1417s_array_FFT.size
n_q_1923s_filter = q_ref_filter_reshape_1923s_array_FFT.size
n_q_2428s_filter = q_ref_filter_reshape_2428s_array_FFT.size
n_q_3340s_filter = q_ref_filter_reshape_3340s_array_FFT.size
n_q_filter = q_ref_filter_reshape_array_FFT.size

n_action_1317s_filter = action_filter_reshape_1317s_array_FFT.size
n_action_1011s_filter = action_filter_reshape_1011s_array_FFT.size
n_action_2832s_filter = action_filter_reshape_2832s_array_FFT.size
n_action_3238s_filter = action_filter_reshape_3238s_array_FFT.size
n_action_filter = action_filter_reshape_array_FFT.size

#frequency
freq_q_1417s_filter = np.fft.fftfreq(n_q_1417s_filter, 0.001)
freq_q_1923s_filter = np.fft.fftfreq(n_q_1923s_filter, 0.001)
freq_q_2428s_filter = np.fft.fftfreq(n_q_2428s_filter, 0.001)
freq_q_3340s_filter = np.fft.fftfreq(n_q_3340s_filter, 0.001)
freq_q_filter = np.fft.fftfreq(n_q_filter, 0.001)

freq_action_1317s_filter = np.fft.fftfreq(n_action_1317s_filter, 0.001)
freq_action_1011s_filter = np.fft.fftfreq(n_action_1011s_filter, 0.001)
freq_action_2832s_filter = np.fft.fftfreq(n_action_2832s_filter, 0.001)
freq_action_3238s_filter = np.fft.fftfreq(n_action_3238s_filter, 0.001)
freq_action_filter = np.fft.fftfreq(n_action_filter, 0.001)

#half of data
n_q_1417s_filter_half = n_q_1417s_filter//2
n_q_1923s_filter_half = n_q_1923s_filter//2
n_q_2428s_filter_half = n_q_2428s_filter//2
n_q_3340s_filter_half = n_q_3340s_filter//2
n_q_filter_half = n_q_filter//2

n_action_1317s_filter_half = n_action_1317s_filter//2
n_action_1011s_filter_half = n_action_1011s_filter//2
n_action_2832s_filter_half = n_action_2832s_filter//2
n_action_3238s_filter_half = n_action_3238s_filter//2
n_action_filter_half = n_action_filter//2

freq_q_1417s_filter_half = freq_q_1417s_filter[:n_q_1417s_filter_half]
freq_q_1923s_filter_half = freq_q_1923s_filter[:n_q_1923s_filter_half]
freq_q_2428s_filter_half = freq_q_2428s_filter[:n_q_2428s_filter_half]
freq_q_3340s_filter_half = freq_q_3340s_filter[:n_q_3340s_filter_half]
freq_q_filter_half = freq_q_filter[:n_q_filter_half]

freq_action_1317s_filter_half = freq_action_1317s_filter[:n_action_1317s_filter_half]
freq_action_1011s_filter_half = freq_action_1011s_filter[:n_action_1011s_filter_half]
freq_action_2832s_filter_half = freq_action_2832s_filter[:n_action_2832s_filter_half]
freq_action_3238s_filter_half = freq_action_3238s_filter[:n_action_3238s_filter_half]
freq_action_filter_half = freq_action_filter[:n_action_filter_half]

q_ref_filter_reshape_1417s_array_FFT_half = q_ref_filter_reshape_1417s_array_FFT[:n_q_1417s_filter_half]
q_ref_filter_reshape_1923s_array_FFT_half = q_ref_filter_reshape_1923s_array_FFT[:n_q_1923s_filter_half]
q_ref_filter_reshape_2428s_array_FFT_half = q_ref_filter_reshape_2428s_array_FFT[:n_q_2428s_filter_half]
q_ref_filter_reshape_3340s_array_FFT_half = q_ref_filter_reshape_3340s_array_FFT[:n_q_3340s_filter_half]
q_ref_filter_reshape_array_FFT_half = q_ref_filter_reshape_array_FFT[:n_q_filter_half]

action_ref_filter_reshape_1317s_array_FFT_half = action_filter_reshape_1317s_array_FFT[:n_action_1317s_filter_half]
action_ref_filter_reshape_1011s_array_FFT_half = action_filter_reshape_1011s_array_FFT[:n_action_1011s_filter_half]
action_ref_filter_reshape_2832s_array_FFT_half = action_filter_reshape_2832s_array_FFT[:n_action_2832s_filter_half]
action_ref_filter_reshape_3238s_array_FFT_half = action_filter_reshape_3238s_array_FFT[:n_action_3238s_filter_half]
action_ref_filter_reshape_array_FFT_half = action_filter_reshape_array_FFT[:n_action_filter_half]

#amplitude modification
q_ref_filter_reshape_1417s_array_FFT_half_modified = np.concatenate(([q_ref_filter_reshape_1417s_array_FFT_half[0]/n_q_1417s_filter], q_ref_filter_reshape_1417s_array_FFT_half[1:-1]*2/n_q_1417s_filter, [q_ref_filter_reshape_1417s_array_FFT_half[-1]/n_q_1417s_filter]))
q_ref_filter_reshape_1417s_array_FFT_half_modified_abs = np.abs(q_ref_filter_reshape_1417s_array_FFT_half_modified)

q_ref_filter_reshape_1923s_array_FFT_half_modified = np.concatenate(([q_ref_filter_reshape_1923s_array_FFT_half[0]/n_q_1923s_filter], q_ref_filter_reshape_1923s_array_FFT_half[1:-1]*2/n_q_1923s_filter, [q_ref_filter_reshape_1923s_array_FFT_half[-1]/n_q_1923s_filter]))
q_ref_filter_reshape_1923s_array_FFT_half_modified_abs = np.abs(q_ref_filter_reshape_1923s_array_FFT_half_modified)

q_ref_filter_reshape_2428s_array_FFT_half_modified = np.concatenate(([q_ref_filter_reshape_2428s_array_FFT_half[0]/n_q_2428s_filter], q_ref_filter_reshape_2428s_array_FFT_half[1:-1]*2/n_q_2428s_filter, [q_ref_filter_reshape_2428s_array_FFT_half[-1]/n_q_2428s_filter]))
q_ref_filter_reshape_2428s_array_FFT_half_modified_abs = np.abs(q_ref_filter_reshape_2428s_array_FFT_half_modified)

q_ref_filter_reshape_3340s_array_FFT_half_modified = np.concatenate(([q_ref_filter_reshape_3340s_array_FFT_half[0]/n_q_3340s_filter], q_ref_filter_reshape_3340s_array_FFT_half[1:-1]*2/n_q_3340s_filter, [q_ref_filter_reshape_3340s_array_FFT_half[-1]/n_q_3340s_filter]))
q_ref_filter_reshape_3340s_array_FFT_half_modified_abs = np.abs(q_ref_filter_reshape_3340s_array_FFT_half_modified)

q_ref_filter_reshape_array_FFT_half_modified = np.concatenate(([q_ref_filter_reshape_array_FFT_half[0]/n_q_filter], q_ref_filter_reshape_array_FFT_half[1:-1]*2/n_q_filter, [q_ref_filter_reshape_array_FFT_half[-1]/n_q_filter]))
q_ref_filter_reshape_array_FFT_half_modified_abs = np.abs(q_ref_filter_reshape_array_FFT_half_modified)

action_ref_filter_reshape_1317s_array_FFT_half_modified = np.concatenate(([action_ref_filter_reshape_1317s_array_FFT_half[0]/n_action_1317s_filter], action_ref_filter_reshape_1317s_array_FFT_half[1:-1]*2/n_action_1317s_filter, [action_ref_filter_reshape_1317s_array_FFT_half[-1]/n_action_1317s_filter]))
action_ref_filter_reshape_1317s_array_FFT_half_modified_abs = np.abs(action_ref_filter_reshape_1317s_array_FFT_half_modified)

action_ref_filter_reshape_1011s_array_FFT_half_modified = np.concatenate(([action_ref_filter_reshape_1011s_array_FFT_half[0]/n_action_1011s_filter], action_ref_filter_reshape_1011s_array_FFT_half[1:-1]*2/n_action_1011s_filter, [action_ref_filter_reshape_1011s_array_FFT_half[-1]/n_action_1011s_filter]))
action_ref_filter_reshape_1011s_array_FFT_half_modified_abs = np.abs(action_ref_filter_reshape_1011s_array_FFT_half_modified)

action_ref_filter_reshape_2832s_array_FFT_half_modified = np.concatenate(([action_ref_filter_reshape_2832s_array_FFT_half[0]/n_action_2832s_filter], action_ref_filter_reshape_2832s_array_FFT_half[1:-1]*2/n_action_2832s_filter, [action_ref_filter_reshape_2832s_array_FFT_half[-1]/n_action_2832s_filter]))
action_ref_filter_reshape_2832s_array_FFT_half_modified_abs = np.abs(action_ref_filter_reshape_2832s_array_FFT_half_modified)

action_ref_filter_reshape_3238s_array_FFT_half_modified = np.concatenate(([action_ref_filter_reshape_3238s_array_FFT_half[0]/n_action_3238s_filter], action_ref_filter_reshape_3238s_array_FFT_half[1:-1]*2/n_action_3238s_filter, [action_ref_filter_reshape_3238s_array_FFT_half[-1]/n_action_3238s_filter]))
action_ref_filter_reshape_3238s_array_FFT_half_modified_abs = np.abs(action_ref_filter_reshape_3238s_array_FFT_half_modified)

action_ref_filter_reshape_array_FFT_half_modified = np.concatenate(([action_ref_filter_reshape_array_FFT_half[0]/n_action_filter], action_ref_filter_reshape_array_FFT_half[1:-1]*2/n_action_filter, [action_ref_filter_reshape_array_FFT_half[-1]/n_action_filter]))
action_ref_filter_reshape_array_FFT_half_modified_abs = np.abs(action_ref_filter_reshape_array_FFT_half_modified)

time = np.array(range(0,40000,1))



#np.fft.fft
#q frequency comparison (np.fft.fft)
fig1 = plt.figure(figsize=(18,9))
plt.subplot(4,4,1)
plt.plot(q_ref_noise_unsmooth_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unsmooth_reshape_1417s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('Baseline')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

plt.subplot(4,4,2)
plt.plot(q_ref_noise_smooth_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
#plt.plot(q_ref_smooth_reshape_1417s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_smooth_reshape_1417s,linewidth=1.0,color = 'C0')
plt.grid(True)
plt.xlabel('Time [s]',fontdict={'size':10})
plt.title('TS')
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

plt.subplot(4,4,3)
plt.plot(q_ref_noise_unsmooth_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unsmooth_reshape_1923s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.title('Baseline')
plt.xticks([0, 2000, 4000], ['19', '21', '23'])

plt.subplot(4,4,4)
plt.plot(q_ref_noise_smooth_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref (actor)}}$')
#plt.plot(q_ref_smooth_reshape_1923s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
plt.plot(q_smooth_reshape_1923s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.title('TS')
plt.grid(True)
plt.legend(loc='lower right',fontsize=10)
plt.xticks([0, 2000, 4000], ['19', '21', '23'])

plt.subplot(4,4,5)
plt.plot(freq_q_1417s_unsmooth_half[0:n_q_1417s_unsmooth_half//5], q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.plot(freq_1417s_half, q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified_abs,linewidth=1.0,color = 'C3',)
plt.ylim(0,1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4,4,6)
plt.plot(freq_q_1417s_smooth_half[0:n_q_1417s_smooth_half//5], q_ref_smooth_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_smooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_1417s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]')
plt.ylim(0,1)
plt.grid(True)

s = freq_q_1923s_unsmooth_half[0:n_q_1923s_unsmooth_half]

plt.subplot(4,4,7)
plt.plot(freq_q_1923s_unsmooth_half[0:n_q_1923s_unsmooth_half//5], q_ref_noise_unsmooth_reshape_1923s_array_FFT_half_modified_abs[0:n_q_1923s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_noise_unsmooth_reshape_1923s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]')
plt.ylim(0,1)
plt.grid(True)

plt.subplot(4,4,8)
plt.plot(freq_q_1923s_smooth_half[0:n_q_1923s_smooth_half//5], q_ref_smooth_reshape_1923s_array_FFT_half_modified_abs[0:n_q_1923s_smooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_1923s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]')
plt.ylabel('')
plt.ylim(0,1)
plt.grid(True)

plt.subplot(4,4,9)
plt.plot(q_ref_noise_unsmooth_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unsmooth_reshape_2428s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 2000, 4000], ['24', '26', '28'])

plt.subplot(4,4,10)
plt.plot(q_ref_noise_smooth_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
#plt.plot(q_ref_smooth_reshape_2428s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_smooth_reshape_2428s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]')
plt.grid(True)
plt.xticks([0, 2000, 4000], ['24', '26', '28'])

plt.subplot(4,4,11)
plt.plot(q_ref_noise_unsmooth_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unsmooth_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]')
plt.grid(True)
plt.xticks([0, 3500, 7000], ['33', '36.5', '40'])

plt.subplot(4,4,12)
plt.plot(q_ref_noise_smooth_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref(actor)}}$')
#plt.plot(q_ref_smooth_reshape_3340s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref(filter)}}$')
plt.plot(q_smooth_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]')
plt.grid(True)
plt.xticks([0, 3500, 7000], ['33', '36.5', '40'])


plt.subplot(4,4,13)
plt.plot(freq_q_2428s_unsmooth_half[0:n_q_2428s_unsmooth_half//5], q_ref_noise_unsmooth_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_noise_unsmooth_reshape_2428s_array, Fs=1/0.001, color='C3')
plt.ylim(0,10)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4,4,14)
plt.plot(freq_q_2428s_smooth_half[0:n_q_2428s_unsmooth_half//5], q_ref_smooth_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_2428s_array, Fs=1/0.001, color='C3')
plt.ylabel('')
plt.ylim(0,10)
plt.xlabel('Frequency [Hz]')
plt.grid(True)


plt.subplot(4,4,15)
plt.plot(freq_q_3340s_unsmooth_half[0:n_q_3340s_unsmooth_half//5], q_ref_noise_unsmooth_reshape_3340s_array_FFT_half_modified_abs[0:n_q_3340s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_noise_unsmooth_reshape_3340s_array, Fs=1/0.001, color='C3')
plt.ylabel('')
plt.ylim(0,10)
plt.xlabel('Frequency [Hz]')
plt.grid(True)

plt.subplot(4,4,16)
plt.plot(freq_q_3340s_smooth_half[0:n_q_3340s_unsmooth_half//5], q_ref_smooth_reshape_3340s_array_FFT_half_modified_abs[0:n_q_3340s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_3340s_array, Fs=1/0.001, color='C3')
plt.ylim(0,10)
plt.xlabel('Frequency [Hz]')
plt.grid(True)

plt.tight_layout()
plt.savefig('Smooth_comparison_q_FFT_fft.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Smooth_comparison_q_FFT_fft.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)




#delta frequency comparison (np.fft.fft)
fig2 = plt.figure(figsize=(18,9))
plt.subplot(4,4,1)
plt.plot(action_unsmooth_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('Baseline')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':10})
plt.xticks([0, 500, 1000], ['10', '10.5', '11'])

plt.subplot(4,4,2)
plt.plot(action_smooth_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('TS')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.xticks([0, 500, 1000], ['10', '10.5', '11'])

plt.subplot(4,4,3)
plt.plot(action_unsmooth_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.title('Baseline')
plt.xticks([0, 2000, 4000], ['13', '15', '17'])

plt.subplot(4,4,4)
plt.plot(action_smooth_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('TS')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.xticks([0, 2000, 4000], ['13', '15', '17'])

plt.subplot(4,4,5)
plt.plot(freq_action_1011s_unsmooth_half[0:n_action_1011s_smooth_half//5], action_ref_noise_unsmooth_reshape_1011s_array_FFT_half_modified_abs[0:n_action_1011s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_1011s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
plt.ylabel('Amplitude')
plt.ylim(0,3)
plt.grid(True)

plt.subplot(4,4,6)
plt.plot(freq_action_1011s_smooth_half[0:n_action_1011s_smooth_half//5], action_ref_smooth_reshape_1011s_array_FFT_half_modified_abs[0:n_action_1011s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_1011s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,3)
plt.grid(True)

plt.subplot(4,4,7)
plt.plot(freq_action_1317s_unsmooth_half[0:n_action_1317s_unsmooth_half//5], action_ref_noise_unsmooth_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_1317s_array, Fs=1/0.001, color='C3')
plt.ylim(0,3)
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
plt.ylabel('')
plt.grid(True)

plt.subplot(4,4,8)
plt.plot(freq_action_1317s_smooth_half[0:n_action_1317s_smooth_half//5], action_ref_smooth_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_smooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_1317s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,3)
plt.grid(True)

plt.subplot(4,4,9)
plt.plot(action_unsmooth_reshape_2832s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':10})
plt.grid(True)
#plt.title('Unfiltered')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 2000, 4000], ['28', '30', '32'])

plt.subplot(4,4,10)
plt.plot(action_smooth_reshape_2832s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 2000, 4000], ['28', '30', '32'])

plt.subplot(4,4,11)
plt.plot(action_unsmooth_reshape_3238s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 3000, 6000], ['32', '35', '38'])

plt.subplot(4,4,12)
plt.plot(action_smooth_reshape_3238s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 3000, 6000], ['32', '35', '38'])

plt.subplot(4,4,13)
plt.plot(freq_action_2832s_unsmooth_half[0:n_action_2832s_unsmooth_half//5], action_ref_noise_unsmooth_reshape_2832s_array_FFT_half_modified_abs[0:n_action_2832s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_2832s_array, Fs=1/0.001, color='C3')
plt.ylim(0,8)
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4,4,14)
plt.plot(freq_action_2832s_smooth_half[0:n_action_2832s_smooth_half//5], action_ref_smooth_reshape_2832s_array_FFT_half_modified_abs[0:n_action_2832s_smooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_2832s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,8)
plt.grid(True)

plt.subplot(4,4,15)
plt.plot(freq_action_3238s_unsmooth_half[0:n_action_3238s_unsmooth_half//5], action_ref_noise_unsmooth_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_unsmooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_3238s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,8)
plt.grid(True)

plt.subplot(4,4,16)
plt.plot(freq_action_3238s_smooth_half[0:n_action_3238s_smooth_half//5], action_ref_smooth_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_smooth_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_3238s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,8)
plt.grid(True)

plt.tight_layout()
plt.savefig('Smooth_comparison_action_FFT_fft.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Smooth_comparison_action_FFT_fft.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)


#np.fft.fft
#q frequency comparison (np.fft.fft)
fig3 = plt.figure(figsize=(18.0,9.0))
plt.subplot(4,4,1)
plt.plot(q_ref_noise_unfilter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('Vanilla')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

plt.subplot(4,4,2)
plt.plot(q_ref_noise_filter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
#plt.plot(q_ref_smooth_reshape_1417s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_filter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.grid(True)
plt.xlabel('Time [s]',fontdict={'size':10})
plt.title('TS')
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

plt.subplot(4,4,3)
plt.plot(q_ref_noise_unfilter_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_1923s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.title('Vanilla')
plt.xticks([0, 2000, 4000], ['19', '21', '23'])

plt.subplot(4,4,4)
plt.plot(q_ref_noise_filter_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref (actor)}}$')
#plt.plot(q_ref_smooth_reshape_1923s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
plt.plot(q_filter_reshape_1923s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.title('TS')
plt.grid(True)
plt.legend(loc='lower right',fontsize=10)
plt.xticks([0, 2000, 4000], ['19', '21', '23'])

plt.subplot(4,4,5)
plt.plot(freq_q_1417s_unfilter_half[0:n_q_1417s_unfilter_half//5], q_ref_noise_unfilter_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.plot(freq_1417s_half, q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified_abs,linewidth=1.0,color = 'C3',)
plt.ylim(0,0.6)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4,4,6)
plt.plot(freq_q_1417s_filter_half[0:n_q_1417s_filter_half//5], q_ref_filter_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_1417s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]')
plt.ylim(0,0.6)
plt.grid(True)

s = freq_q_1923s_unfilter_half[0:n_q_1923s_unfilter_half]

plt.subplot(4,4,7)
plt.plot(freq_q_1923s_unfilter_half[0:n_q_1923s_unfilter_half//5], q_ref_noise_unfilter_reshape_1923s_array_FFT_half_modified_abs[0:n_q_1923s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_noise_unsmooth_reshape_1923s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]')
plt.ylim(0,0.6)
plt.grid(True)

plt.subplot(4,4,8)
plt.plot(freq_q_1923s_filter_half[0:n_q_1923s_filter_half//5], q_ref_filter_reshape_1923s_array_FFT_half_modified_abs[0:n_q_1923s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_1923s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]')
plt.ylabel('')
plt.ylim(0,0.6)
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
#plt.plot(q_ref_smooth_reshape_2428s,linewidth=1.0,color = 'C2',linestyle='--')
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
#plt.plot(q_ref_smooth_reshape_3340s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref(filter)}}$')
plt.plot(q_filter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]')
plt.grid(True)
plt.xticks([0, 3500, 7000], ['33', '36.5', '40'])


plt.subplot(4,4,13)
plt.plot(freq_q_2428s_unfilter_half[0:n_q_2428s_unfilter_half//5], q_ref_noise_unfilter_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_noise_unsmooth_reshape_2428s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.2)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4,4,14)
plt.plot(freq_q_2428s_filter_half[0:n_q_2428s_unfilter_half//5], q_ref_filter_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_2428s_array, Fs=1/0.001, color='C3')
plt.ylabel('')
plt.ylim(0,0.2)
plt.xlabel('Frequency [Hz]')
plt.grid(True)


plt.subplot(4,4,15)
plt.plot(freq_q_3340s_unfilter_half[0:n_q_3340s_unfilter_half//5], q_ref_noise_unfilter_reshape_3340s_array_FFT_half_modified_abs[0:n_q_3340s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_noise_unsmooth_reshape_3340s_array, Fs=1/0.001, color='C3')
plt.ylabel('')
plt.ylim(0,0.2)
plt.xlabel('Frequency [Hz]')
plt.grid(True)

plt.subplot(4,4,16)
plt.plot(freq_q_3340s_filter_half[0:n_q_3340s_unfilter_half//5], q_ref_filter_reshape_3340s_array_FFT_half_modified_abs[0:n_q_3340s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_3340s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.2)
plt.xlabel('Frequency [Hz]')
plt.grid(True)

plt.tight_layout()
plt.savefig('Filter_comparison_q_FFT_fft.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison_q_FFT_fft.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)




#delta frequency comparison (np.fft.fft)
fig4 = plt.figure(figsize=(18.0,9.0))
plt.subplot(4,4,1)
plt.plot(action_unfilter_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('Vanilla')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':10})
plt.xticks([0, 500, 1000], ['10', '10.5', '11'])

plt.subplot(4,4,2)
plt.plot(action_filter_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('TS')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.xticks([0, 500, 1000], ['10', '10.5', '11'])

plt.subplot(4,4,3)
plt.plot(action_unfilter_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.title('Vanilla')
plt.xticks([0, 2000, 4000], ['13', '15', '17'])

plt.subplot(4,4,4)
plt.plot(action_filter_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('TS')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.xticks([0, 2000, 4000], ['13', '15', '17'])

plt.subplot(4,4,5)
plt.plot(freq_action_1011s_unfilter_half[0:n_action_1011s_filter_half//5], action_ref_noise_unfilter_reshape_1011s_array_FFT_half_modified_abs[0:n_action_1011s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_1011s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
plt.ylabel('Amplitude')
plt.ylim(0,0.6)
plt.grid(True)

plt.subplot(4,4,6)
plt.plot(freq_action_1011s_filter_half[0:n_action_1011s_filter_half//5], action_ref_filter_reshape_1011s_array_FFT_half_modified_abs[0:n_action_1011s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_1011s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,0.6)
plt.grid(True)

plt.subplot(4,4,7)
plt.plot(freq_action_1317s_unfilter_half[0:n_action_1317s_unfilter_half//5], action_ref_noise_unfilter_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_1317s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.6)
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
plt.ylabel('')
plt.grid(True)

plt.subplot(4,4,8)
plt.plot(freq_action_1317s_filter_half[0:n_action_1317s_filter_half//5], action_ref_filter_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_1317s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,0.6)
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
plt.plot(freq_action_2832s_unfilter_half[0:n_action_2832s_unfilter_half//5], action_ref_noise_unfilter_reshape_2832s_array_FFT_half_modified_abs[0:n_action_2832s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_2832s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.3)
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4,4,14)
plt.plot(freq_action_2832s_filter_half[0:n_action_2832s_filter_half//5], action_ref_filter_reshape_2832s_array_FFT_half_modified_abs[0:n_action_2832s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_2832s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,0.3)
plt.grid(True)

plt.subplot(4,4,15)
plt.plot(freq_action_3238s_unfilter_half[0:n_action_3238s_unfilter_half//5], action_ref_noise_unfilter_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_3238s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,0.3)
plt.grid(True)

plt.subplot(4,4,16)
plt.plot(freq_action_3238s_filter_half[0:n_action_3238s_filter_half//5], action_ref_filter_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_3238s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontdict={'size':10})
#plt.ylabel('')
plt.ylim(0,0.3)
plt.grid(True)

plt.tight_layout()
plt.savefig('Filter_comparison_action_FFT_fft.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison_action_FFT_fft.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)



#np.fft.fft
#q frequency comparison (np.fft.fft)
fig5 = plt.figure(figsize=(18,9))
plt.subplot(4,4,1)
plt.plot(q_ref_noise_unsmooth_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unsmooth_reshape_1417s,linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.title('IHDP')
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

plt.subplot(4,4,2)
plt.plot(q_ref_noise_smooth_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
#plt.plot(q_ref_smooth_reshape_1417s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_smooth_reshape_1417s,linewidth=1.0,color = 'C0')
plt.grid(True)
plt.title('TS-IHDP')
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

plt.subplot(4,4,3)
plt.plot(q_ref_noise_filter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--')
#plt.plot(q_ref_smooth_reshape_1417s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_filter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.grid(True)
plt.title('Command-filtered TS-IHDP')
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'])

plt.subplot(4,4,4)
plt.plot(freq_q_1417s_unsmooth_half[0:n_q_1417s_unsmooth_half//5], q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_unsmooth_half//5],linewidth=1.0,color = 'C0',label = 'IHDP')
plt.plot(freq_q_1417s_smooth_half[0:n_q_1417s_smooth_half//5], q_ref_smooth_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_smooth_half//5],linewidth=1.0,color = 'C1',label = 'TS-IHDP')
plt.plot(freq_q_1417s_filter_half[0:n_q_1417s_filter_half//5], q_ref_filter_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_filter_half//5],linewidth=1.0,color = 'C3',label = 'command-filtered TS-IHDP')
plt.fill_between(freq_q_1417s_unsmooth_half[0:n_q_1417s_unsmooth_half//5], q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_unsmooth_half//5], 0, color = 'C0', alpha=0.2)
plt.fill_between(freq_q_1417s_smooth_half[0:n_q_1417s_smooth_half//5], q_ref_smooth_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_smooth_half//5], 0,color = 'C1', alpha=0.2)
plt.fill_between(freq_q_1417s_filter_half[0:n_q_1417s_filter_half//5], q_ref_filter_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_filter_half//5], 0, color = 'C3',alpha=0.2)
plt.ylim(0,1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=10)


plt.subplot(4,4,5)
plt.plot(q_ref_noise_unsmooth_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unsmooth_reshape_2428s,linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 2000, 4000], ['24', '26', '28'])

plt.subplot(4,4,6)
plt.plot(q_ref_noise_smooth_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
#plt.plot(q_ref_smooth_reshape_2428s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_smooth_reshape_2428s,linewidth=1.0,color = 'C0')
plt.grid(True)
plt.xticks([0, 2000, 4000], ['24', '26', '28'])


s = freq_q_1923s_unsmooth_half[0:n_q_1923s_unsmooth_half]

plt.subplot(4,4,7)
plt.plot(q_ref_noise_filter_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
#plt.plot(q_ref_smooth_reshape_2428s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_filter_reshape_2428s,linewidth=1.0,color = 'C0')
plt.grid(True)
plt.xticks([0, 2000, 4000], ['24', '26', '28'])

plt.subplot(4,4,8)
plt.plot(freq_q_2428s_unsmooth_half[0:n_q_2428s_unsmooth_half//5], q_ref_noise_unsmooth_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_unsmooth_half//5],linewidth=1.0,color = 'C0')
plt.plot(freq_q_2428s_smooth_half[0:n_q_2428s_smooth_half//5], q_ref_smooth_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_smooth_half//5],linewidth=1.0,color = 'C1')
plt.plot(freq_q_2428s_filter_half[0:n_q_2428s_unfilter_half//5], q_ref_filter_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_filter_half//5],linewidth=1.0,color = 'C3')
plt.fill_between(freq_q_2428s_unsmooth_half[0:n_q_2428s_unsmooth_half//5], q_ref_noise_unsmooth_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_unsmooth_half//5], 0, color = 'C0', alpha=0.2)
plt.fill_between(freq_q_2428s_smooth_half[0:n_q_2428s_smooth_half//5], q_ref_smooth_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_smooth_half//5], 0,color = 'C1', alpha=0.2)
plt.fill_between(freq_q_2428s_filter_half[0:n_q_2428s_filter_half//5], q_ref_filter_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_filter_half//5], 0, color = 'C3',alpha=0.2)
plt.ylim(0,1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=10)

plt.subplot(4,4,9)
plt.plot(action_unsmooth_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
#plt.title('Baseline')
plt.xticks([0, 2000, 4000], ['13', '15', '17'])
plt.ylabel(r'$\delta$ [deg/s]',fontdict={'size':10})

plt.subplot(4,4,10)
plt.plot(action_smooth_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
#plt.title('TS')
plt.xticks([0, 2000, 4000], ['13', '15', '17'])

plt.subplot(4,4,11)
plt.plot(action_filter_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
#plt.title('TS')
plt.xticks([0, 2000, 4000], ['13', '15', '17'])
#plt.subplot(4,4,11)
#plt.plot(q_ref_noise_unsmooth_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--')
#plt.plot(q_unsmooth_reshape_3340s,linewidth=1.0,color = 'C0')
###plt.xlabel('Time [s]')
#plt.grid(True)
#plt.xticks([0, 3500, 7000], ['33', '36.5', '40'])

plt.subplot(4,4,12)
plt.plot(freq_action_1317s_unsmooth_half[0:n_action_1317s_unsmooth_half//5], action_ref_noise_unsmooth_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_unsmooth_half//5],linewidth=1.0,color = 'C0')
plt.plot(freq_action_1317s_smooth_half[0:n_action_1317s_smooth_half//5], action_ref_smooth_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_smooth_half//5],linewidth=1.0,color = 'C1')
plt.plot(freq_action_1317s_filter_half[0:n_action_1317s_unfilter_half//5], action_ref_filter_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_filter_half//5],linewidth=1.0,color = 'C3')
plt.fill_between(freq_action_1317s_unsmooth_half[0:n_action_1317s_unsmooth_half//5], action_ref_noise_unsmooth_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_unsmooth_half//5], 0, color = 'C0', alpha=0.2)
plt.fill_between(freq_action_1317s_smooth_half[0:n_action_1317s_smooth_half//5], action_ref_smooth_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_smooth_half//5], 0,color = 'C1', alpha=0.2)
plt.fill_between(freq_action_1317s_filter_half[0:n_action_1317s_unfilter_half//5], action_ref_filter_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_filter_half//5], 0, color = 'C3',alpha=0.2)
plt.ylim(0,1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=10)

plt.subplot(4,4,13)
plt.plot(action_unsmooth_reshape_3238s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 3000, 6000], ['32', '35', '38'])
plt.ylabel(r'$\delta$ [deg/s]',fontdict={'size':10})

plt.subplot(4,4,14)
plt.plot(action_smooth_reshape_3238s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 3000, 6000], ['32', '35', '38'])


plt.subplot(4,4,15)
plt.plot(action_filter_reshape_3238s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontdict={'size':10})
plt.grid(True)
plt.xticks([0, 3000, 6000], ['32', '35', '38'])

plt.subplot(4,4,16)
plt.plot(freq_action_3238s_unsmooth_half[0:n_action_3238s_unsmooth_half//5], action_ref_noise_unsmooth_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_unsmooth_half//5],linewidth=1.0,color = 'C0')
plt.plot(freq_action_3238s_smooth_half[0:n_action_3238s_smooth_half//5], action_ref_smooth_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_smooth_half//5],linewidth=1.0,color = 'C1')
plt.plot(freq_action_1317s_filter_half[0:n_action_3238s_unfilter_half//5], action_ref_filter_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_filter_half//5],linewidth=1.0,color = 'C3')
plt.fill_between(freq_action_3238s_unsmooth_half[0:n_action_3238s_unsmooth_half//5], action_ref_noise_unsmooth_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_unsmooth_half//5], 0, color = 'C0', alpha=0.2)
plt.fill_between(freq_action_3238s_smooth_half[0:n_action_3238s_smooth_half//5], action_ref_smooth_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_smooth_half//5], 0,color = 'C1', alpha=0.2)
plt.fill_between(freq_action_3238s_filter_half[0:n_action_3238s_unfilter_half//5], action_ref_filter_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_filter_half//5], 0, color = 'C3',alpha=0.2)
plt.ylim(0,1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=10)
plt.tight_layout()
plt.savefig('Smooth_comparison_q_FFT_fft.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Smooth_comparison_q_FFT_fft.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

#fig6 = plt.figure(figsize=(18,9))
#plt.plot(freq_q_unsmooth_half[0:n_q_unsmooth_half//5], q_ref_noise_unsmooth_reshape_array_FFT_half_modified_abs[0:n_q_unsmooth_half//5],linewidth=1.0,color = 'C3')
##plt.magnitude_spectrum(q_ref_smooth_reshape_2428s_array, Fs=1/0.001, color='C3')
#plt.ylabel('')
#plt.ylim(0,1.0)
#plt.xlabel('Frequency [Hz]')
#plt.grid(True)


fig6 = plt.figure(figsize=(18,9))
plt.subplot(2,1,1)
plt.plot(freq_q_unsmooth_half[0:n_q_unsmooth_half//5], q_ref_noise_unsmooth_reshape_array_FFT_half_modified_abs[0:n_q_unsmooth_half//5],linewidth=0.01,color = 'C0',label='IHDP')
plt.plot(freq_q_smooth_half[0:n_q_smooth_half//5], q_ref_smooth_reshape_array_FFT_half_modified_abs[0:n_q_smooth_half//5],linewidth=1.0,color = 'C1',label='TS-IHDP')
plt.plot(freq_q_filter_half[0:n_q_filter_half//5], q_ref_filter_reshape_array_FFT_half_modified_abs[0:n_q_filter_half//5],linewidth=1.0,color = 'C3',label='command-filtered TS-IHDP')
plt.fill_between(freq_q_unsmooth_half[0:n_q_unsmooth_half//5], q_ref_noise_unsmooth_reshape_array_FFT_half_modified_abs[0:n_q_unsmooth_half//5], 0, color = 'C0', alpha=0.2)
plt.fill_between(freq_q_smooth_half[0:n_q_smooth_half//5], q_ref_smooth_reshape_array_FFT_half_modified_abs[0:n_q_smooth_half//5], 0,color = 'C1', alpha=0.2)
plt.fill_between(freq_q_filter_half[0:n_q_filter_half//5], q_ref_filter_reshape_array_FFT_half_modified_abs[0:n_q_filter_half//5], 0, color = 'C3',alpha=0.2)
plt.ylim(0,0.6)
plt.xlabel('Frequency [Hz]',fontdict={'size':20})
plt.ylabel('Amplitude',fontdict={'size':20})
plt.grid(True)
plt.title('pitch rate reference',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=18)

#fig8 = plt.figure(figsize=(18,9))
plt.subplot(2,1,2)
plt.plot(freq_action_unsmooth_half[0:n_action_unsmooth_half//5], action_ref_noise_unsmooth_reshape_array_FFT_half_modified_abs[0:n_action_unsmooth_half//5],linewidth=0.01,color = 'C0')
plt.plot(freq_action_smooth_half[0:n_action_smooth_half//5], action_ref_smooth_reshape_array_FFT_half_modified_abs[0:n_action_smooth_half//5],linewidth=1.0,color = 'C1')
plt.plot(freq_action_filter_half[0:n_action_filter_half//5], action_ref_filter_reshape_array_FFT_half_modified_abs[0:n_action_filter_half//5],linewidth=1.0,color = 'C3')
plt.fill_between(freq_action_unsmooth_half[0:n_action_unsmooth_half//5], action_ref_noise_unsmooth_reshape_array_FFT_half_modified_abs[0:n_action_unsmooth_half//5], 0, color = 'C0', alpha=0.2)
plt.fill_between(freq_action_smooth_half[0:n_action_smooth_half//5], action_ref_smooth_reshape_array_FFT_half_modified_abs[0:n_action_smooth_half//5], 0,color = 'C1', alpha=0.2)
plt.fill_between(freq_action_filter_half[0:n_action_filter_half//5], action_ref_filter_reshape_array_FFT_half_modified_abs[0:n_action_filter_half//5], 0, color = 'C3',alpha=0.2)
plt.ylim(0,0.6)
plt.xlabel('Frequency [Hz]',fontdict={'size':20})
plt.ylabel('Amplitude',fontdict={'size':20})
plt.grid(True)
#plt.legend(loc='upper right',bbox_to_anchor=(1,1.025),fontsize=10)
plt.title('control surface deflection',fontdict={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.tight_layout()
plt.savefig('Smooth_filter_comparison_action_FFT_fft.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Smooth_filter_comparison_action_FFT_fft.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.show()