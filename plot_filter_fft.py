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


fig1 = plt.figure(figsize=(18.0,9.0))
plt.plot(freq_q_1417s_unfilter_half, q_ref_noise_unfilter_reshape_1417s_array_FFT_half_modified_abs,label='np.fft.fft')
plt.magnitude_spectrum(q_ref_noise_unfilter_reshape_1417s_array, Fs=1/0.001, color='C3',label='plt.magnitude_spectrum')
plt.title('twp python functions', fontdict={'size':10})
plt.grid(True)
plt.legend(loc='upper right',fontsize=10)

#subplots_for_states
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


# plt.magnitude_spectrum
#q frequency comparison (plt.magnitude_spectrum)
fig5 = plt.figure(figsize=(18.0,9.0))
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
plt.magnitude_spectrum(q_ref_noise_unfilter_reshape_1417s_array, Fs=1/0.001, color='C3')
#plt.plot(freq_1417s_half[0:300], q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified_abs[0:300],linewidth=1.0,color = 'C3')
#plt.plot(freq_1417s_half, q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified_abs,linewidth=1.0,color = 'C3',)
plt.ylim(0,0.4)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
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
fig6 = plt.figure(figsize=(18.0,9.0))
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


#delta q
fig7 = plt.figure(figsize=(18,9))
plt.subplot(2,1,1)
plt.stem(delta_q_ref_noise_unfilter_reshape, basefmt='None', markerfmt='',linefmt='-C1',label = 'Vanilla')
plt.stem(delta_q_ref_noise_filter_reshape, basefmt='None', markerfmt='',linefmt='-C0',label = 'TS')
#plt.plot(delta_q_ref_noise_unsmooth_reshape,linewidth=1.0,color = 'C1',linestyle='-',label='Vanilla')
#plt.plot(delta_q_ref_noise_smooth_reshape,linewidth=1.0,color = 'C0',linestyle='-',label='Temporal smooth')
plt.grid(True)
#plt.title('Higher-level agent',fontdict={'size':20})


plt.legend(loc='upper left',fontsize=20)
#plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$\vert \Delta q_{\mathrm{ref}} \vert$ [deg/s]',fontdict={'size':20})
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize = 20)
plt.yticks(fontsize = 20)

plt.subplot(2,1,2)
plt.stem(delta_action_unfilter, basefmt='None', markerfmt='',linefmt='-C1',label='Vanilla')
plt.stem(delta_action_filter, basefmt='None', markerfmt='',linefmt='-C0',label='TS')
plt.grid(True)
#plt.title('Lower-level agent',fontdict={'size':20})
plt.xlabel('Time [s]',fontdict={'size':20})
plt.ylabel(r'$\vert \Delta \delta \vert$ [deg]',fontdict={'size':20})
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'],fontsize = 20)
plt.yticks(fontsize = 20)

plt.tight_layout()
plt.savefig('Filter_comparison_incrementactions.pdf',bbox_inches='tight', pad_inches=0, dpi=300)
plt.savefig('Filter_comparison_incrementactions.eps',bbox_inches='tight', pad_inches=0, dpi=300)


#np.fft.fft size of figure = (4x2)
#q frequency comparison (np.fft.fft)
fig8 = plt.figure(figsize=(14,12))
plt.subplot(4,2,1)
plt.plot(q_ref_noise_unfilter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref}}$')
plt.plot(q_unfilter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=16)
plt.ylabel(r'$q$ [deg/s]',fontsize=16)
plt.grid(True)
plt.title('TS-IHDP',fontsize=16)
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'],fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right',fontsize=16)

plt.subplot(4,2,2)
plt.plot(q_ref_noise_filter_reshape_1417s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q^{\prime}_{\mathrm{ref}}$')
#plt.plot(q_ref_smooth_reshape_1417s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_filter_reshape_1417s,linewidth=1.0,color = 'C0')
plt.grid(True)
plt.xlabel('Time [s]',fontsize=16)
plt.title('Command-filtered TS-IHDP',fontsize=16)
plt.xticks([0, 1500, 3000], ['14', '15.5', '17'],fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right',fontsize=16)


plt.subplot(4,2,3)
plt.plot(freq_q_1417s_unfilter_half[0:n_q_1417s_unfilter_half//5], q_ref_noise_unfilter_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.plot(freq_1417s_half, q_ref_noise_unsmooth_reshape_1417s_array_FFT_half_modified_abs,linewidth=1.0,color = 'C3',)
plt.ylim(0,0.6)
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,4)
plt.plot(freq_q_1417s_filter_half[0:n_q_1417s_filter_half//5], q_ref_filter_reshape_1417s_array_FFT_half_modified_abs[0:n_q_1417s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_1417s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylim(0,0.6)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,5)
plt.plot(q_ref_noise_unfilter_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_1923s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=16)
plt.ylabel(r'$q$ [deg/s]',fontsize=16)
plt.grid(True)
plt.xticks([0, 2000, 4000], ['19', '21', '23'],fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,6)
plt.plot(q_ref_noise_filter_reshape_1923s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref (actor)}}$')
#plt.plot(q_ref_smooth_reshape_1923s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref (filter)}}$')
plt.plot(q_filter_reshape_1923s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=16)
plt.grid(True)
#plt.legend(loc='lower right',fontsize=15)
plt.xticks([0, 2000, 4000], ['19', '21', '23'],fontsize=16)
plt.yticks(fontsize=16)

s = freq_q_1923s_unfilter_half[0:n_q_1923s_unfilter_half]

plt.subplot(4,2,7)
plt.plot(freq_q_1923s_unfilter_half[0:n_q_1923s_unfilter_half//5], q_ref_noise_unfilter_reshape_1923s_array_FFT_half_modified_abs[0:n_q_1923s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_noise_unsmooth_reshape_1923s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.ylim(0,0.6)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,8)
plt.plot(freq_q_1923s_filter_half[0:n_q_1923s_filter_half//5], q_ref_filter_reshape_1923s_array_FFT_half_modified_abs[0:n_q_1923s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_1923s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('')
plt.ylim(0,0.6)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig('Filter_comparison_q_fft_part_1.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison_q_fft_part_1.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

fig9 = plt.figure(figsize=(14,12))
plt.subplot(4,2,1)
plt.plot(q_ref_noise_unfilter_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_2428s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=16)
plt.ylabel(r'$q$ [deg/s]',fontsize=16)
plt.grid(True)
plt.xticks([0, 2000, 4000], ['24', '26', '28'],fontsize=16)
plt.yticks(fontsize=16)
plt.title('TS-IHDP',fontsize=16)


plt.subplot(4,2,2)
plt.plot(q_ref_noise_filter_reshape_2428s,linewidth=1.0,color = 'C1',linestyle='--')
#plt.plot(q_ref_smooth_reshape_2428s,linewidth=1.0,color = 'C2',linestyle='--')
plt.plot(q_filter_reshape_2428s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=16)
plt.grid(True)
plt.xticks([0, 2000, 4000], ['24', '26', '28'],fontsize=16)
plt.yticks(fontsize=16)
plt.title('Command-filtered TS-IHDP',fontsize=16)


plt.subplot(4,2,3)
plt.plot(freq_q_2428s_unfilter_half[0:n_q_2428s_unfilter_half//5], q_ref_noise_unfilter_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_noise_unsmooth_reshape_2428s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.2)
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.subplot(4,2,4)
plt.plot(freq_q_2428s_filter_half[0:n_q_2428s_unfilter_half//5], q_ref_filter_reshape_2428s_array_FFT_half_modified_abs[0:n_q_2428s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_2428s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.2)
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.subplot(4,2,5)
plt.plot(q_ref_noise_unfilter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--')
plt.plot(q_unfilter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=16)
plt.ylabel(r'$q$ [deg/s]',fontsize=16)
plt.grid(True)
plt.xticks([0, 3500, 7000], ['33', '36.5', '40'],fontsize=16)
plt.yticks(fontsize=16)


plt.subplot(4,2,6)
plt.plot(q_ref_noise_filter_reshape_3340s,linewidth=1.0,color = 'C1',linestyle='--',label=r'$q_{\mathrm{ref(actor)}}$')
#plt.plot(q_ref_smooth_reshape_3340s,linewidth=1.0,color = 'C2',linestyle='--',label=r'$q_{\mathrm{ref(filter)}}$')
plt.plot(q_filter_reshape_3340s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=16)
plt.grid(True)
plt.xticks([0, 3500, 7000], ['33', '36.5', '40'],fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,7)
plt.plot(freq_q_3340s_unfilter_half[0:n_q_3340s_unfilter_half//5], q_ref_noise_unfilter_reshape_3340s_array_FFT_half_modified_abs[0:n_q_3340s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_noise_unsmooth_reshape_3340s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.2)
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.subplot(4,2,8)
plt.plot(freq_q_3340s_filter_half[0:n_q_3340s_unfilter_half//5], q_ref_filter_reshape_3340s_array_FFT_half_modified_abs[0:n_q_3340s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(q_ref_smooth_reshape_3340s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.2)
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.tight_layout()
plt.savefig('Filter_comparison_q_fft_part_2.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison_q_fft_part_2.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)



#delta frequency comparison (np.fft.fft) figure size=(4,2)
fig10 = plt.figure(figsize=(14,12))
plt.subplot(4,2,1)
plt.plot(action_unfilter_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('TS-IHDP',fontsize=16)
plt.xlabel('Time [s]',fontsize=16)
plt.ylabel(r'$\delta$ [deg]',fontsize=16)
plt.xticks([0, 500, 1000], ['10', '10.5', '11'],fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,2)
plt.plot(action_filter_reshape_1011s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.title('Command-filtered TS-IHDP',fontsize=16)
plt.xlabel('Time [s]',fontsize=16)
plt.xticks([0, 500, 1000], ['10', '10.5', '11'],fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,3)
plt.plot(freq_action_1011s_unfilter_half[0:n_action_1011s_filter_half//5], action_ref_noise_unfilter_reshape_1011s_array_FFT_half_modified_abs[0:n_action_1011s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_1011s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.ylim(0,0.6)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.subplot(4,2,4)
plt.plot(freq_action_1011s_filter_half[0:n_action_1011s_filter_half//5], action_ref_filter_reshape_1011s_array_FFT_half_modified_abs[0:n_action_1011s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_1011s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylim(0,0.6)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,5)
plt.plot(action_unfilter_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontsize=16)
plt.ylabel(r'$\delta$ [deg]',fontsize=16)
plt.grid(True)
plt.xticks([0, 2000, 4000], ['13', '15', '17'],fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,6)
plt.plot(action_filter_reshape_1317s,linewidth=1.0,color = 'C0',linestyle='-')
plt.grid(True)
plt.xlabel('Time [s]',fontsize=16)
plt.xticks([0, 2000, 4000], ['13', '15', '17'],fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,7)
plt.plot(freq_action_1317s_unfilter_half[0:n_action_1317s_unfilter_half//5], action_ref_noise_unfilter_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_1317s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.6)
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,8)
plt.plot(freq_action_1317s_filter_half[0:n_action_1317s_filter_half//5], action_ref_filter_reshape_1317s_array_FFT_half_modified_abs[0:n_action_1317s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_1317s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylim(0,0.6)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig('Filter_comparison_action_fft_part_1.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison_action_fft_part_1.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)


fig11 = plt.figure(figsize=(14,12))
plt.subplot(4,2,1)
plt.plot(action_unfilter_reshape_2832s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontsize=16)
plt.ylabel(r'$\delta$ [deg]',fontsize=16)
plt.grid(True)
plt.title('TS',fontsize=16)
plt.xticks([0, 2000, 4000], ['28', '30', '32'],fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,2)
plt.plot(action_filter_reshape_2832s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontsize=16)
plt.grid(True)
plt.xticks([0, 2000, 4000], ['28', '30', '32'],fontsize=16)
plt.yticks(fontsize=16)
plt.title('TS+Filter',fontsize=16)

plt.subplot(4,2,3)
plt.plot(freq_action_2832s_unfilter_half[0:n_action_2832s_unfilter_half//5], action_ref_noise_unfilter_reshape_2832s_array_FFT_half_modified_abs[0:n_action_2832s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_2832s_array, Fs=1/0.001, color='C3')
plt.ylim(0,0.3)
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,4)
plt.plot(freq_action_2832s_filter_half[0:n_action_2832s_filter_half//5], action_ref_filter_reshape_2832s_array_FFT_half_modified_abs[0:n_action_2832s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_2832s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylim(0,0.3)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,5)
plt.plot(action_unfilter_reshape_3238s,linewidth=1.0,color = 'C0')
plt.xlabel('Time [s]',fontsize=16)
plt.ylabel(r'$\delta$ [deg]',fontsize=16)
plt.grid(True)
plt.xticks([0, 3000, 6000], ['32', '35', '38'],fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,6)
plt.plot(action_filter_reshape_3238s,linewidth=1.0,color = 'C0',linestyle='-')
plt.xlabel('Time [s]',fontsize=16)
plt.grid(True)
plt.xticks([0, 3000, 6000], ['32', '35', '38'],fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,7)
plt.plot(freq_action_3238s_unfilter_half[0:n_action_3238s_unfilter_half//5], action_ref_noise_unfilter_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_unfilter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_unsmooth_reshape_3238s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('Amplitude',fontsize=16)
plt.ylim(0,0.3)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.subplot(4,2,8)
plt.plot(freq_action_3238s_filter_half[0:n_action_3238s_filter_half//5], action_ref_filter_reshape_3238s_array_FFT_half_modified_abs[0:n_action_3238s_filter_half//5],linewidth=1.0,color = 'C3')
#plt.magnitude_spectrum(action_smooth_reshape_3238s_array, Fs=1/0.001, color='C3')
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylim(0,0.3)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig('Filter_comparison_action_fft_part2.pdf',bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig('Filter_comparison_action_fft_part2.eps',bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.show()