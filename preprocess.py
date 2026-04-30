import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import random

def get_data(file):
    data = pd.read_excel(file)
    data.columns = data.columns.str.strip()
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


def noise(signal, snr_db = 20):
    

    signal = np.asarray(signal)

    signal_power = np.mean(signal**2)   
    snr_linear = 10**(snr_db / 10)
    
    noise_power = signal_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise

    return noisy_signal


def pad_to_length(x, target_len):
    x = np.asarray(x)
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)), mode='constant')
    return x


def create_signal(base_harmonic, second_harmonic, beta, noise_level,
                  base_mode, second_mode, modes_base, modes_harmonic):

    # Max amplitudes
    A1 = np.max(base_harmonic[base_mode].to_numpy())
    A2 = np.max(second_harmonic[second_mode].to_numpy())

    A2_target = beta * (A1 ** 2)
    second_scale = A2_target / A2

    # ---- lengths ----
    base_len = len(base_harmonic["Sum Propagated signal (nm)"].to_numpy())
    second_len = len(second_harmonic[second_mode].to_numpy())
    max_len = max(base_len, second_len)

    # ---- choose correct time array (LONGEST one) ----
    if base_len >= second_len:
        time = base_harmonic["Propagation time (micsec)"].to_numpy()
    else:
        time = second_harmonic["Propagation time (micsec)"].to_numpy()

    # Initialize signal
    signal = np.zeros(max_len)

    # Base modes
    for mode in modes_base:
        sig = base_harmonic[mode].to_numpy()
        signal += pad_to_length(sig, max_len)

    # Harmonic modes
    for mode in modes_harmonic:
        sig = second_harmonic[mode].to_numpy()
        signal += second_scale * pad_to_length(sig, max_len)

    # Noise
    level = noise_level * A2 * second_scale
    noise = np.random.normal(0, level, max_len)

    noisy_signal = signal + noise

    return time, noisy_signal, second_scale*second_harmonic

