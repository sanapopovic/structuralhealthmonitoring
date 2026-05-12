from transforms.wavelet_processing import wavelet_scalogram

import numpy as np
import scipy as sp
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import preprocess
import transforms.stft_processing as stft_processing
import matplotlib.pyplot as plt
from transforms.wavelet_processing import wavelet_scalogram


# #All files should be uploaded as csv
# data = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")

# t = data["Propagation time (micsec)"] #time axis in microseconds for this experiment
# y = data["Sum Propagated signal (nm)"] #measured signal, in nanometres

# time, freq, amp = wavelet_scalogram(t, y, wavelet = 'cgau2', n_scales=100, name="wavelet_scalogram") #Runs the Continuous Wavelet Transform (CWT) using a complex Morlet wavelet

#makes a sine wave

#parameters
freq = 1.33e5
wlen = 1


t = np.linspace(0,500e-6, 2500)
signalSine = np.sin(2*np.pi * freq * t)

total_samples = len(t)
window_length = total_samples // 4        # half the total length
pad = (total_samples - window_length) // 2  # equal padding on each side

hann = sp.signal.windows.hann(window_length)
padded_hann = np.concatenate([
    np.zeros(pad),
    hann,
    np.zeros(total_samples - window_length - pad)  # handles odd-length remainders
])

#hann = sp.signal.windows.hann(len(t))
#signalHannSine = signalSine * hann
signalHannSine = signalSine * padded_hann

def plot_signals(t,signalHannSine, signalSine):
    plt.plot(t, signalHannSine)
    plt.show()
    plt.plot(t, signalSine)
    plt.show()

#time, freq, amp = wavelet_scalogram(t, signal, wavelet = 'cgau2', n_scales=100, name="wavelet_scalogram") #Runs the Continuous Wavelet Transform (CWT) using a complex Morlet wavelet
