import pywt
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, issq_cwt, Wavelet

parameter = "wavelet_center_freq" 

# Range for Morse Beta (Proxy for Bandwidth)
b_start, b_end, b_step = 0.5, 20, 0.5
# Range for Morlet Mu (Proxy for Center Frequency)
c_start, c_end, c_step = 2.0, 15.0, 0.5 

if parameter == "wavelet_bandwidth":
    test_range = np.arange(b_start, b_end, b_step)
    xlabel = "Wavelet Bandwidth (Morse Beta)"
elif parameter == "wavelet_center_freq":
    test_range = np.arange(c_start, c_end, c_step)
    xlabel = "Wavelet Center Frequency (Morlet Mu)"

wavelet = pywt.ContinuousWavelet('morl')  # Mexican Hat wavelet

psi, x = wavelet.wavefun(level=10)
std_dev = []


for val in test_range:
    # Configure the ssqueezepy Wavelet Object dynamically
    if parameter == "wavelet_bandwidth":
        W_obj = pywt.ContiniuousWavelet(('gmw', {'beta': val, 'gamma': 3}))
        wavelet_label = f"Morse(β={val})"
        psi, x = W_obj.wavefun(level=10)
        std_dev.append(np.std(psi))
    else:
        W_obj = pywt.ContinuousWavelet(('morlet', {'mu': val}))
        wavelet_label = f"Morlet(μ={val})"
        psi, x = W_obj.wavefun(level=10)
        std_dev.append(np.std(psi))


plt.plot(test_range, std_dev)




print(f"Standard Deviation of 'mexh' wavelet: {std_dev:.4f}")