import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
from PyEMD import EMD



def stft(sig, time, nperseg=256, noverlap=None, window='hann'):

    time = time.to_numpy()
    sig = sig.to_numpy()

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    if noverlap is None:
        noverlap = nperseg // 2

    f, t_seg, Zxx = signal.stft( sig, fs=fs, window='hann',nperseg=256, noverlap=128,boundary=None)
    
    amplitude = np.abs(Zxx)

    return f, t_seg, amplitude, fs

#test

def plot_stft(f, t_seg, amplitude, downsampling=1, name="stft_plot", dB=False):
    
    
    # Convert Pandas objects to NumPy arrays if needed
    if isinstance(f, (pd.DataFrame, pd.Series)):
        f = f.to_numpy().squeeze()
    if isinstance(t_seg, (pd.DataFrame, pd.Series)):
        t_seg = t_seg.to_numpy().squeeze()
    if isinstance(amplitude, (pd.DataFrame, pd.Series)):
        amplitude = amplitude.to_numpy()

    # Apply downsampling along the time axis
    t_plot = t_seg[::downsampling]
    amplitude_plot = amplitude[:, ::downsampling]

    # Convert to dB if requested
    if dB:
        amplitude_plot = 20 * np.log10(amplitude_plot + 1e-12)  # add small value to avoid log(0)

    # Create plots directory
    folder = "plots"
    os.makedirs(folder, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_plot, f, amplitude_plot, shading='gouraud')
    plt.xlabel("Time [ms]")
    plt.ylabel("Frequency [MHz]")
    plt.title(name)
    plt.colorbar(label='Amplitude (dB)' if dB else 'Amplitude')
    plt.tight_layout()
    #plt.ylim(2,3.5)

    # Save plot
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {filepath}")


    time = time.to_numpy()
    sig = sig.to_numpy()

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    if noverlap is None:
        noverlap = nperseg // 2

    f, t_seg, Zxx = signal.stft( sig, fs=fs, window='hann',nperseg=256, noverlap=128,boundary=None)
    
    amplitude = np.abs(Zxx)

    return f, t_seg, amplitude, fs

#test


















































































































































































































































