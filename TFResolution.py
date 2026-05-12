from transforms.wavelet_processing import wavelet_scalogram
import pandas as pd
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

def plot_signals(t,signalHannSine, signalSine):
    plt.plot(t, signalHannSine)
    plt.show()
    plt.plot(t, signalSine)
    plt.show()

def wavelet(t,signal):
    print("lol")
    
def SST(t,signal):
    print("lol")

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

    # Create plots directory
    #folder = "plots"
    #os.makedirs(folder, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_plot, f, amplitude_plot, shading='gouraud')
    plt.xlabel("Time [ms]")
    plt.ylabel("Frequency [MHz]")
    plt.title("stft_plot")
    plt.colorbar(label='Amplitude (dB)' if dB else 'Amplitude')
    plt.tight_layout()
    plt.show()
    

    # Save plot
    #filepath = os.path.join(folder, f"{name}.png")
    #plt.savefig(filepath, dpi=300)
    #plt.close()

    #print(f"Plot saved to {filepath}")

STFT(t,signalSine,downsampling=1,hop=128,dB=False)
STFT(t,signalHannSine,downsampling=1,hop=128,dB=False)

#time, freq, amp = wavelet_scalogram(t, signal, wavelet = 'cgau2', n_scales=100, name="wavelet_scalogram") #Runs the Continuous Wavelet Transform (CWT) using a complex Morlet wavelet
