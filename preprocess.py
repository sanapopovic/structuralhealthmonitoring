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


def resample_to_time(signal, old_time, new_time):
    """
    Resample signal onto a new time grid while preserving timing.
    """
    return np.interp(new_time, old_time, signal)


def create_signal(base_harmonic, second_harmonic,
                  beta, noise_level,
                  base_mode, second_mode,
                  modes_base, modes_harmonic, distance):

    # ------------------------------------------------------------
    # 1. Extract time vectors
    # ------------------------------------------------------------
    t_base = base_harmonic["Propagation time (micsec)"].to_numpy()
    t_harm = second_harmonic["Propagation time (micsec)"].to_numpy()

    # ------------------------------------------------------------
    # 2. Build a COMMON physical time grid
    #    (this is the critical fix)
    # ------------------------------------------------------------
    dt_base = np.mean(np.diff(t_base))
    dt_harm = np.mean(np.diff(t_harm))

    dt = min(dt_base, dt_harm)  # safest Nyquist-preserving choice

    t_start = max(t_base[0], t_harm[0])
    t_end   = min(t_base[-1], t_harm[-1])

    t = np.arange(t_start, t_end, dt)

    # ------------------------------------------------------------
    # 3. Amplitude scaling (β is PURE amplitude control)
    # ------------------------------------------------------------
    A1 = (np.max(base_harmonic[base_mode].to_numpy()))*1e-9
    A2 = (np.max(second_harmonic[second_mode].to_numpy()))*1e-9

    k = (2*np.pi*1.33*1e6)/(8.303885011*1e3) #frequency and phase velocity
    A2_prime = (beta*(k**2)*(A1**2)*distance)/8
    second_scale = A2_prime/A2  # avoid divide-by-zero

    # ------------------------------------------------------------
    # 4. Initialize signal
    # ------------------------------------------------------------
    signal = np.zeros(len(t))

    # ------------------------------------------------------------
    # 5. Add base modes (resampled onto common grid)
    # ------------------------------------------------------------
    for mode in modes_base:
        sig = base_harmonic[mode].to_numpy()
        sig_rs = np.interp(t, t_base, sig)
        signal += sig_rs

    # ------------------------------------------------------------
    # 6. Add harmonic modes (resampled + scaled)
    # ------------------------------------------------------------
    for mode in modes_harmonic:
        sig = second_harmonic[mode].to_numpy()
        sig_rs = np.interp(t, t_harm, sig)
        signal += second_scale * sig_rs

    # ------------------------------------------------------------
    # 7. Add noise 
    # ------------------------------------------------------------
    noise_std = noise_level * A2_prime*1e9 / 2
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, len(t))

    noisy_signal = signal + noise

    # ------------------------------------------------------------
    # 8. Return consistent dataset
    # ------------------------------------------------------------
    return t, noisy_signal, second_scale

