import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os




def std(y):

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()

    y_mean = np.mean(y)
    y_s = np.std(y)

    y = (y-y_mean)/y_s

    return y




def stft(sig, time, win_length =256, hop= 128):

    if isinstance(sig, (pd.DataFrame, pd.Series)):
        sig = sig.to_numpy()
    if isinstance(time, (pd.DataFrame, pd.Series)):
        time = time.to_numpy()

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    w = signal.windows.hann(win_length, sym= False)


    SFT = signal.ShortTimeFFT(win=w, hop = hop, fs= fs)
    S = SFT.stft(sig)

    if SFT.invertible:
        I = SFT.istft(S)

    else:
        I = 0
    
    return S, I, fs 

def plot_stft(S, fs, hop, downsampling=1, name="stft_plot", dB=False):

    # Convert Pandas objects to NumPy arrays if needed
    if isinstance(S, (pd.DataFrame, pd.Series)):
        S = S.to_numpy()

    # Compute amplitude from complex STFT
    amplitude = np.abs(S)

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
    

    # Save plot
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Plot saved to {filepath}")

def frequency_amplitude(S, fs):

    # Convert Pandas objects to NumPy arrays if needed
    if isinstance(S, (pd.DataFrame, pd.Series)):
        S = S.to_numpy()

    # Magnitude of STFT
    amplitude = np.abs(S)

    # Average amplitude across time frames
    amp_freq = np.mean(amplitude, axis=1)

    # Frequency axis
    n_freq = S.shape[0]
    f = np.linspace(0, fs/2, n_freq)

    return f, amp_freq

def detect_ridges(S, min_height=0.1):
   
    S_mag = np.abs(S)
    n_freq, n_time = S_mag.shape
    ridges = []

    for t in range(n_time):
        peaks, _ = signal.find_peaks(S_mag[:, t], height=min_height)
        ridges.append(peaks)

    return ridges
