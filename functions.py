import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

def get_data(file):
   
    # Read CSV
    data = pd.read_csv(file)
    
    for col in data:
        data[col] = data[col].str.replace(',', '.')   
    data = data.astype(float)
    
    return data

def plot(x, y, downsampling=1, name="plot"):
    

    folder = "plots"
    os.makedirs(folder, exist_ok=True)

    # Convert Pandas to NumPy
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x_plot = x.to_numpy().squeeze()
    else:
        x_plot = np.array(x)

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y_plot = y.to_numpy().squeeze()
    else:
        y_plot = np.array(y)

    plt.figure(figsize=(8,4))
    if y_plot.ndim == 1:
        plt.plot(x_plot[::downsampling], y_plot[::downsampling], linewidth=1)
    elif y_plot.ndim == 2:
        # 2D spectrogram plot
        plt.pcolormesh(x_plot, np.arange(y_plot.shape[0]), y_plot, shading='gouraud')
        plt.xlabel("Time")
        plt.ylabel("Frequency bin")
        plt.colorbar(label='Amplitude')
    else:
        raise ValueError("y_plot must be 1D or 2D")   

    plt.title(name)
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filepath}")

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
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(name)
    plt.colorbar(label='Amplitude (dB)' if dB else 'Amplitude')
    plt.tight_layout()

    # Save plot
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {filepath}")

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