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

q_ref_noise_unsmooth_reshape_1417s = q_ref_noise_unsmooth_reshape[14000:17000,:]
q_ref_noise_unsmooth_reshape_1923s = q_ref_noise_unsmooth_reshape[19000:23000,:]
q_ref_noise_unsmooth_reshape_2428s = q_ref_noise_unsmooth_reshape[24000:28000,:]
q_ref_noise_unsmooth_reshape_3340s = q_ref_noise_unsmooth_reshape[33000:40000,:]

q_ref_unsmooth_reshape_1417s = q_ref_unsmooth_reshape[14000:17000,:]
q_ref_unsmooth_reshape_1922s = q_ref_unsmooth_reshape[19000:22000,:]
q_ref_unsmooth_reshape_2428s = q_ref_unsmooth_reshape[24000:28000,:]
q_ref_unsmooth_reshape_3340s = q_ref_unsmooth_reshape[33000:40000,:]

#action
action_unsmooth = np.reshape(action_unsmooth_reshape[:,1],(-1,1))
action_unsmooth_reshape_1317s = action_unsmooth[13000:17000,:]
action_unsmooth_reshape_1011s = action_unsmooth[10000:11000,:]
action_unsmooth_reshape_2832s = action_unsmooth[28000:32000,:]
action_unsmooth_reshape_3238s = action_unsmooth[32000:38000,:]


#arrary to list
q_ref_noise_unsmooth_reshape_1417s_array = np.reshape(q_ref_noise_unsmooth_reshape_1417s,(3000,))
q_ref_noise_unsmooth_reshape_1923s_array = np.reshape(q_ref_noise_unsmooth_reshape_1923s,(4000,))
q_ref_noise_unsmooth_reshape_2428s_array = np.reshape(q_ref_noise_unsmooth_reshape_2428s,(4000,))
q_ref_noise_unsmooth_reshape_3340s_array = np.reshape(q_ref_noise_unsmooth_reshape_3340s,(7000,))

action_unsmooth_reshape_1317s_array = np.reshape(action_unsmooth_reshape_1317s,(4000,))
action_unsmooth_reshape_1011s_array = np.reshape(action_unsmooth_reshape_1011s,(1000,))
action_unsmooth_reshape_2832s_array = np.reshape(action_unsmooth_reshape_2832s,(4000,))
action_unsmooth_reshape_3238s_array = np.reshape(action_unsmooth_reshape_3238s,(6000,))

#delta q/action
q_ref_noise_unsmooth_reshape_with0 = np.reshape(np.insert(q_ref_noise_unsmooth_reshape,0,[0]),[40001,1])[:40000,:]
delta_q_ref_noise_unsmooth_reshape = np.abs(q_ref_noise_unsmooth_reshape - q_ref_noise_unsmooth_reshape_with0)

action_unsmooth_with0 = np.reshape(np.insert(action_unsmooth,0,[0]),[40001,1])[:40000,:]
delta_action_unsmooth = np.abs(action_unsmooth - action_unsmooth_with0)


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

q_ref_noise_smooth_reshape_1417s = q_ref_noise_smooth_reshape[14000:17000,:]
q_ref_noise_smooth_reshape_1923s = q_ref_noise_smooth_reshape[19000:23000,:]
q_ref_noise_smooth_reshape_2428s = q_ref_noise_smooth_reshape[24000:28000,:]
q_ref_noise_smooth_reshape_3340s = q_ref_noise_smooth_reshape[33000:40000,:]

q_ref_smooth_reshape_1417s = q_ref_smooth_reshape[14000:17000,:]
q_ref_smooth_reshape_1923s = q_ref_smooth_reshape[19000:23000,:]
q_ref_smooth_reshape_2428s = q_ref_smooth_reshape[24000:28000,:]
q_ref_smooth_reshape_3340s = q_ref_smooth_reshape[33000:40000,:]

#action
action_smooth = np.reshape(action_smooth_reshape[:,1],(-1,1))
action_smooth_reshape_1317s = action_smooth[13000:17000,:]
action_smooth_reshape_1011s = action_smooth[10000:11000,:]
action_smooth_reshape_2832s = action_smooth[28000:32000,:]
action_smooth_reshape_3238s = action_smooth[32000:38000,:]

#arrary to list
q_ref_smooth_reshape_1417s_array = np.reshape(q_ref_smooth_reshape_1417s,(3000,))
q_ref_smooth_reshape_1923s_array = np.reshape(q_ref_smooth_reshape_1923s,(4000,))
q_ref_smooth_reshape_2428s_array = np.reshape(q_ref_smooth_reshape_2428s,(4000,))
q_ref_smooth_reshape_3340s_array = np.reshape(q_ref_smooth_reshape_3340s,(7000,))

action_smooth_reshape_1317s_array = np.reshape(action_smooth_reshape_1317s,(4000,))
action_smooth_reshape_1011s_array = np.reshape(action_smooth_reshape_1011s,(1000,))
action_smooth_reshape_2832s_array = np.reshape(action_smooth_reshape_2832s,(4000,))
action_smooth_reshape_3238s_array = np.reshape(action_smooth_reshape_3238s,(6000,))

#delta q/action
q_ref_noise_smooth_reshape_with0 = np.reshape(np.insert(q_ref_noise_smooth_reshape,0,[0]),[40001,1])[:40000,:]
delta_q_ref_noise_smooth_reshape = np.abs(q_ref_noise_smooth_reshape - q_ref_noise_smooth_reshape_with0)

action_smooth_with0 = np.reshape(np.insert(action_smooth,0,[0]),[40001,1])[:40000,:]
delta_action_smooth = np.abs(action_smooth - action_smooth_with0)


alpha_ref_history = np.reshape(alpha_ref_history,(-1,1))
alpha_ref_history_reshape = alpha_ref_history[:40000,:]



time = np.array(range(0,40000,1))

# Filter data
state_unfilter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/state_history')
action_unfilter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/action_history')
q_ref_noise_unfilter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/q_ref_noise_history')
q_ref_unfilter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/q_ref_filtered_history')
state_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/state_history')
action_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/action_history')
q_ref_noise_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/q_ref_noise_history')
q_ref_filter = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/filter/filter_w=20/q_ref_filtered_history')

alpha_ref_history = np.loadtxt('/home/yifei/PycharmProjects/SafeIHDP_controlloss18_smallgain_Vx/origion/alpha_ref_history')

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

#delta q/action
q_ref_noise_filter_reshape_with0 = np.reshape(np.insert(q_ref_noise_filter_reshape,0,[0]),[40001,1])[:40000,:]
delta_q_ref_noise_filter_reshape = np.abs(q_ref_noise_filter_reshape - q_ref_noise_filter_reshape_with0)

action_filter_with0 = np.reshape(np.insert(action_filter,0,[0]),[40001,1])[:40000,:]
delta_action_filter = np.abs(action_filter - action_filter_with0)


alpha_ref_history = np.reshape(alpha_ref_history,(-1,1))
alpha_ref_history_reshape = alpha_ref_history[:40000,:]


#FFT
#Transformation
q_ref_filter_reshape_1417s_array_FFT = np.fft.fft(q_ref_filter_reshape_1417s_array)
q_ref_filter_reshape_1923s_array_FFT = np.fft.fft(q_ref_filter_reshape_1923s_array)
q_ref_filter_reshape_2428s_array_FFT = np.fft.fft(q_ref_filter_reshape_2428s_array)
q_ref_filter_reshape_3340s_array_FFT = np.fft.fft(q_ref_filter_reshape_3340s_array)

action_filter_reshape_1317s_array_FFT = np.fft.fft(action_filter_reshape_1317s_array)
action_filter_reshape_1011s_array_FFT = np.fft.fft(action_filter_reshape_1011s_array)
action_filter_reshape_2832s_array_FFT = np.fft.fft(action_filter_reshape_2832s_array)
action_filter_reshape_3238s_array_FFT = np.fft.fft(action_filter_reshape_3238s_array)

#sample number
n_q_1417s_filter = q_ref_filter_reshape_1417s_array_FFT.size
n_q_1923s_filter = q_ref_filter_reshape_1923s_array_FFT.size
n_q_2428s_filter = q_ref_filter_reshape_2428s_array_FFT.size
n_q_3340s_filter = q_ref_filter_reshape_3340s_array_FFT.size

n_action_1317s_filter = action_filter_reshape_1317s_array_FFT.size
n_action_1011s_filter = action_filter_reshape_1011s_array_FFT.size
n_action_2832s_filter = action_filter_reshape_2832s_array_FFT.size
n_action_3238s_filter = action_filter_reshape_3238s_array_FFT.size

#frequency
freq_q_1417s_filter = np.fft.fftfreq(n_q_1417s_filter, 0.001)
freq_q_1923s_filter = np.fft.fftfreq(n_q_1923s_filter, 0.001)
freq_q_2428s_filter = np.fft.fftfreq(n_q_2428s_filter, 0.001)
freq_q_3340s_filter = np.fft.fftfreq(n_q_3340s_filter, 0.001)

freq_action_1317s_filter = np.fft.fftfreq(n_action_1317s_filter, 0.001)
freq_action_1011s_filter = np.fft.fftfreq(n_action_1011s_filter, 0.001)
freq_action_2832s_filter = np.fft.fftfreq(n_action_2832s_filter, 0.001)
freq_action_3238s_filter = np.fft.fftfreq(n_action_3238s_filter, 0.001)

#half of data
n_q_1417s_filter_half = n_q_1417s_filter//2
n_q_1923s_filter_half = n_q_1923s_filter//2
n_q_2428s_filter_half = n_q_2428s_filter//2
n_q_3340s_filter_half = n_q_3340s_filter//2

n_action_1317s_filter_half = n_action_1317s_filter//2
n_action_1011s_filter_half = n_action_1011s_filter//2
n_action_2832s_filter_half = n_action_2832s_filter//2
n_action_3238s_filter_half = n_action_3238s_filter//2

freq_q_1417s_filter_half = freq_q_1417s_filter[:n_q_1417s_filter_half]
freq_q_1923s_filter_half = freq_q_1923s_filter[:n_q_1923s_filter_half]
freq_q_2428s_filter_half = freq_q_2428s_filter[:n_q_2428s_filter_half]
freq_q_3340s_filter_half = freq_q_3340s_filter[:n_q_3340s_filter_half]

freq_action_1317s_filter_half = freq_action_1317s_filter[:n_action_1317s_filter_half]
freq_action_1011s_filter_half = freq_action_1011s_filter[:n_action_1011s_filter_half]
freq_action_2832s_filter_half = freq_action_2832s_filter[:n_action_2832s_filter_half]
freq_action_3238s_filter_half = freq_action_3238s_filter[:n_action_3238s_filter_half]

q_ref_filter_reshape_1417s_array_FFT_half = q_ref_filter_reshape_1417s_array_FFT[:n_q_1417s_filter_half]
q_ref_filter_reshape_1923s_array_FFT_half = q_ref_filter_reshape_1923s_array_FFT[:n_q_1923s_filter_half]
q_ref_filter_reshape_2428s_array_FFT_half = q_ref_filter_reshape_2428s_array_FFT[:n_q_2428s_filter_half]
q_ref_filter_reshape_3340s_array_FFT_half = q_ref_filter_reshape_3340s_array_FFT[:n_q_3340s_filter_half]

action_ref_filter_reshape_1317s_array_FFT_half = action_filter_reshape_1317s_array_FFT[:n_action_1317s_filter_half]
action_ref_filter_reshape_1011s_array_FFT_half = action_filter_reshape_1011s_array_FFT[:n_action_1011s_filter_half]
action_ref_filter_reshape_2832s_array_FFT_half = action_filter_reshape_2832s_array_FFT[:n_action_2832s_filter_half]
action_ref_filter_reshape_3238s_array_FFT_half = action_filter_reshape_3238s_array_FFT[:n_action_3238s_filter_half]

#amplitude modification
q_ref_filter_reshape_1417s_array_FFT_half_modified = np.concatenate(([q_ref_filter_reshape_1417s_array_FFT_half[0]/n_q_1417s_filter], q_ref_filter_reshape_1417s_array_FFT_half[1:-1]*2/n_q_1417s_filter, [q_ref_filter_reshape_1417s_array_FFT_half[-1]/n_q_1417s_filter]))
q_ref_filter_reshape_1417s_array_FFT_half_modified_abs = np.abs(q_ref_filter_reshape_1417s_array_FFT_half_modified)

q_ref_filter_reshape_1923s_array_FFT_half_modified = np.concatenate(([q_ref_filter_reshape_1923s_array_FFT_half[0]/n_q_1923s_filter], q_ref_filter_reshape_1923s_array_FFT_half[1:-1]*2/n_q_1923s_filter, [q_ref_filter_reshape_1923s_array_FFT_half[-1]/n_q_1923s_filter]))
q_ref_filter_reshape_1923s_array_FFT_half_modified_abs = np.abs(q_ref_filter_reshape_1923s_array_FFT_half_modified)

q_ref_filter_reshape_2428s_array_FFT_half_modified = np.concatenate(([q_ref_filter_reshape_2428s_array_FFT_half[0]/n_q_2428s_filter], q_ref_filter_reshape_2428s_array_FFT_half[1:-1]*2/n_q_2428s_filter, [q_ref_filter_reshape_2428s_array_FFT_half[-1]/n_q_2428s_filter]))
q_ref_filter_reshape_2428s_array_FFT_half_modified_abs = np.abs(q_ref_filter_reshape_2428s_array_FFT_half_modified)

q_ref_filter_reshape_3340s_array_FFT_half_modified = np.concatenate(([q_ref_filter_reshape_3340s_array_FFT_half[0]/n_q_3340s_filter], q_ref_filter_reshape_3340s_array_FFT_half[1:-1]*2/n_q_3340s_filter, [q_ref_filter_reshape_3340s_array_FFT_half[-1]/n_q_3340s_filter]))
q_ref_filter_reshape_3340s_array_FFT_half_modified_abs = np.abs(q_ref_filter_reshape_3340s_array_FFT_half_modified)

action_ref_filter_reshape_1317s_array_FFT_half_modified = np.concatenate(([action_ref_filter_reshape_1317s_array_FFT_half[0]/n_action_1317s_filter], action_ref_filter_reshape_1317s_array_FFT_half[1:-1]*2/n_action_1317s_filter, [action_ref_filter_reshape_1317s_array_FFT_half[-1]/n_action_1317s_filter]))
action_ref_filter_reshape_1317s_array_FFT_half_modified_abs = np.abs(action_ref_filter_reshape_1317s_array_FFT_half_modified)

action_ref_filter_reshape_1011s_array_FFT_half_modified = np.concatenate(([action_ref_filter_reshape_1011s_array_FFT_half[0]/n_action_1011s_filter], action_ref_filter_reshape_1011s_array_FFT_half[1:-1]*2/n_action_1011s_filter, [action_ref_filter_reshape_1011s_array_FFT_half[-1]/n_action_1011s_filter]))
action_ref_filter_reshape_1011s_array_FFT_half_modified_abs = np.abs(action_ref_filter_reshape_1011s_array_FFT_half_modified)

action_ref_filter_reshape_2832s_array_FFT_half_modified = np.concatenate(([action_ref_filter_reshape_2832s_array_FFT_half[0]/n_action_2832s_filter], action_ref_filter_reshape_2832s_array_FFT_half[1:-1]*2/n_action_2832s_filter, [action_ref_filter_reshape_2832s_array_FFT_half[-1]/n_action_2832s_filter]))
action_ref_filter_reshape_2832s_array_FFT_half_modified_abs = np.abs(action_ref_filter_reshape_2832s_array_FFT_half_modified)

action_ref_filter_reshape_3238s_array_FFT_half_modified = np.concatenate(([action_ref_filter_reshape_3238s_array_FFT_half[0]/n_action_3238s_filter], action_ref_filter_reshape_3238s_array_FFT_half[1:-1]*2/n_action_3238s_filter, [action_ref_filter_reshape_3238s_array_FFT_half[-1]/n_action_3238s_filter]))
action_ref_filter_reshape_3238s_array_FFT_half_modified_abs = np.abs(action_ref_filter_reshape_3238s_array_FFT_half_modified)



time = np.array(range(0,40000,1))


#9_subplots_for_states
fig1 = plt.figure(figsize=(18.0,7.0))
plt.subplot(3,3,1)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
plt.plot(state_unsmooth_reshape[:,0],linewidth=1.0,color = 'C0')
plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.title('IHDP', fontdict={'size':19})
#plt.legend(loc='upper right',fontsize=10)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,3,2)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(state_smooth_reshape[:,0],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.title('TS-IHDP', fontdict={'size':19})
#plt.legend(loc='upper right',fontsize=19)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,3,3)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
plt.plot(state_filter_reshape[:,0],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$\alpha$ [deg]',fontsize=15)
plt.grid(True)
plt.title('Command-filtered TS-IHDP', fontsize=19)
plt.legend(loc='upper right',fontsize=18)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)


plt.subplot(3,3,4)
plt.plot(q_ref_noise_unsmooth_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref}}$')
#plt.plot(q_ref_unfilter_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{ref(filter)}$')
plt.plot(state_unsmooth_reshape[:,1],linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':19})
plt.grid(True)
#plt.title('DDPG (tanh-ReLU)')
#plt.legend(loc='upper left',fontsize=10)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,3,5)
plt.plot(q_ref_noise_smooth_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref}}$')
#plt.plot(q_ref_smooth_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
plt.plot(state_smooth_reshape[:,1],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$q$ [deg/s]',fontdict={'size':19})
plt.grid(True)
#plt.title('DDPG (tanh-ReLU)')
plt.legend(loc='upper right',fontsize=18)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,3,6)
plt.plot(q_ref_noise_filter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q^{\prime}_{\mathrm{ref}}$')
#plt.plot(q_ref_smooth_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
plt.plot(state_filter_reshape[:,1],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$q$ [deg/s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize=18)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,3,7)
plt.plot(action_unsmooth_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,3,8)
plt.plot(action_smooth_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontdict={'size':19})
#plt.ylabel(r'$\delta$ [deg]',fontdict={'size':19})
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,3,9)
plt.plot(action_filter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=19)
#plt.ylabel(r'$\delta$ [deg]',fontsize=15)
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.savefig('Smooth_filter_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Smooth_filter_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)


fig2 = plt.figure(figsize=(18,9))
plt.subplot(3,2,1)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
plt.plot(state_unfilter_reshape[:,0],linewidth=1.0,color = 'C0')
plt.ylabel(r'$\alpha$ [deg]',fontsize=15)
plt.grid(True)
plt.title('TS-IHDP', fontsize=15)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,2,2)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
plt.plot(state_filter_reshape[:,0],linewidth=1.0,color = 'C0')
plt.ylabel(r'$\alpha$ [deg]',fontsize=15)
plt.grid(True)
plt.title('Command-filtered TS-IHDP', fontsize=15)
plt.legend(loc='upper right',fontsize=15)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,2,3)
plt.plot(q_ref_noise_unfilter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref}}$')
#plt.plot(q_ref_unfilter_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{ref(filter)}$')
plt.plot(state_unfilter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontsize=15)
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper right',fontsize=15)

plt.subplot(3,2,4)
plt.plot(q_ref_noise_filter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q^{\prime}_{\mathrm{ref}}$')
#plt.plot(q_ref_smooth_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
plt.plot(state_filter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.ylabel(r'$q$ [deg/s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize=15)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(3,2,5)
plt.plot(action_unfilter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=15)
plt.ylabel(r'$\delta$ [deg]',fontsize=15)
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)


plt.subplot(3,2,6)
plt.plot(action_filter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=15)
plt.ylabel(r'$\delta$ [deg]',fontsize=15)
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('Filter_comparison.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)



fig3 = plt.figure(figsize=(10.0,7.0))

plt.subplot(3,1,1)
plt.plot(alpha_ref_history_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$\alpha_{\mathrm{ref}}$')
plt.plot(state_filter_reshape[:,0],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$\alpha$ [deg]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize=18)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel(r'$\alpha$ [deg]',fontdict={'size':15})


plt.subplot(3,1,2)
plt.plot(q_ref_noise_filter_reshape,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q^{\prime}_{\mathrm{ref}}$')
#plt.plot(q_ref_smooth_reshape,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
plt.plot(state_filter_reshape[:,1],linewidth=1.0,color = 'C0')
#plt.ylabel(r'$q$ [deg/s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize=18)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel(r'$q$ [deg/s]',fontdict={'size':15})

plt.subplot(3,1,3)
plt.plot(action_filter_reshape[:,1],linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=19)
#plt.ylabel(r'$\delta$ [deg]',fontsize=15)
plt.grid(True)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel(r'$\delta$ [deg]',fontdict={'size':15})


plt.tight_layout()
plt.savefig('Smooth_filter_comparison_short.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Smooth_filter_comparison_short.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)


plt.show()