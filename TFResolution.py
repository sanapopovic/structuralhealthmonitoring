from transforms.wavelet_processing import wavelet_scalogram
import pandas as pd
import numpy as np
import scipy as sp
import sys, os
import transforms.SST_v2_processing as sst_processing_v2
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
freq = 1.33e6
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

def plot_signals_stft(t1,f1,a1,t2,f2,a2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.pcolormesh(t1, f1, a1, shading='gouraud')
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Frequency [MHz]")

    ax2.pcolormesh(t2, f2, a2, shading='gouraud')
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Frequency [MHz]")

    plt.tight_layout()
    plt.show()


def plot_signals_stft_sst(t1,f1,a1,t2,f2,a2,fmin=1e6,fmax=4.5e6):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.pcolormesh(t1, f1/ 1e6, a1, shading="gouraud")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Frequency [MHz]")
    ax1.set_ylim(fmin / 1e6, fmax / 1e6)

    ax2.pcolormesh(t2, f2/ 1e6, a2, shading="gouraud")
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Frequency [MHz]")
    ax2.set_ylim(fmin / 1e6, fmax / 1e6)

    plt.tight_layout()
    plt.show()

    


def STFT(t,signal,downsampling=1,hop=128,dB=False):
    ft, I, fs = stft_processing.stft(signal, t,win_length =256)

    # Convert Pandas objects to NumPy arrays if needed
    if isinstance(ft, (pd.DataFrame, pd.Series)):
        ft = ft.to_numpy()

    # Compute amplitude from complex STFT
    amplitude = np.abs(ft)

    # Frequency and time axes
    n_freq, n_time = amplitude.shape
    f = np.linspace(0, fs/2, n_freq)
    t_seg = np.arange(n_time) * hop / fs

    # Apply downsampling along the time axis
    t_plot = t_seg[::downsampling]
    amplitude_plot = amplitude[:, ::downsampling]

    # Convert to dB if requested
    if dB:
        amplitude_plot = 20 * np.log10(amplitude_plot + 1e-12)

    return t_plot,f,amplitude_plot


    










t_sin,f_sin,a_sin = STFT(t,signalSine,downsampling=1,hop=128,dB=False)
t_hann,f_hann,a_hann=STFT(t,signalHannSine,downsampling=1,hop=128,dB=False)
t_sin_stft_sst,f_sin_stft_sst,a_sin_stft_sst=sst_processing_v2.stft_sst(t,signalSine)
t_hann_stft_sst,f_hann_stft_sst,a_hann_stft_sst=sst_processing_v2.stft_sst(t,signalHannSine)
t_sin_cwt_sst,f_sin_cwt_sst,a_sin_cwt_sst = sst_processing_v2.cwt_sst(t,signalSine)
t_hann_cwt_sst,f_hann_cwt_sst,a_hann_cwt_sst = sst_processing_v2.cwt_sst(t,signalHannSine)
plot_signals_stft(t_sin,f_sin,a_sin,t_hann,f_hann,a_hann)
#plot_signals_stft_sst(t_sin_stft_sst,f_sin_stft_sst,a_sin_stft_sst,t_hann_stft_sst,f_hann_stft_sst,a_hann_stft_sst)

