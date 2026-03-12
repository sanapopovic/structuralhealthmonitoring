import pywt 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def wavelet_decompose(sig, wavelet='db4', level=None):
    if isinstance(sig, (pd.Series, pd.DataFrame)):
        sig = sig.to_numpy().squeeze()

    sig = np.asarray(sig, dtype=float).copy()

    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(sig), w.dec_len)
    print("max_level =", max_level)

    if level is None:
        level =  max_level  # choose up to 5 levels

    coeffs = pywt.wavedec(sig, wavelet=w, level=level)
    return coeffs  

def wavelet_scalogram(t, sig, wavelet='cmor1.5-1.0', n_scales=100, name="wavelet_scalogram"):
    sig = np.asarray(sig).squeeze()
    t = np.asarray(t).squeeze()

    dt = np.mean(np.diff(t))  # sampling period
    widths = np.geomspace(1, 512, n_scales)

    cwtmatr, freqs = pywt.cwt(sig, widths, wavelet, sampling_period=dt)
    power = np.abs(cwtmatr)

    folder = "plots"
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(8,4))
    plt.pcolormesh(t, freqs, power, shading='gouraud')
    plt.yscale('log')
    plt.xlabel("Time")
    plt.ylabel("Frequency [Hz]")
    plt.title(name)
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {filepath}")
