import pywt 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def wavelet_decompose(sig, wavelet='db4', level=None):  #this is a multi-level decomposition, not a single level
    if isinstance(sig, (pd.Series, pd.DataFrame)):  #signal coverted to numpy array if it is a pandas Series or DataFrame
        sig = sig.to_numpy().squeeze() #squeeze is used to remove any extra dimensions, ensuring we have a 1D array for processing.

    sig = np.asarray(sig, dtype=float).copy() #ensure the signal is a numpy array of type float, and create a copy to avoid modifying the original data.

    w = pywt.Wavelet(wavelet) #create a wavelet object based on the specified wavelet name. Using db4 (Daubechies 4)
    max_level = pywt.dwt_max_level(len(sig), w.dec_len) #calculate the maximum level of decomposition based on the length of the signal and the length of the wavelet filter. This ensures we don't decompose beyond what the signal can support.
    print("max_level =", max_level)

    if level is None:
        level =  max_level  #if level is not specified, use the maximum level of decomposition

    coeffs = pywt.wavedec(sig, wavelet=w, level=level) #actual Discrete Wavelet Transform
    return coeffs  


def wavelet_scalogram(t, sig, wavelet='cmor1.5-1.0', n_scales=100, name="wavelet_scalogram"): #wavelet time–frequency map (CWT)
    sig = np.asarray(sig).squeeze()
    t = np.asarray(t).squeeze()

    dt = np.mean(np.diff(t))  #sampling period
    widths = np.geomspace(1, 512, n_scales) #Creates n_scales wavelet scales from 1 to 512 on a logarithmic grid

    cwtmatr, freqs = pywt.cwt(sig, widths, wavelet, sampling_period=dt) #Runs the CWT. 2D matrix (frequency × time), freqs holds the corresponding frequencies in Hz.
    power = np.abs(cwtmatr) #Takes magnitude so you can plot amplitude/energy.

    folder = "plots"
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.pcolormesh(t, freqs, power, shading='gouraud') #Draws a heatmap: x = time, y = frequency, color = amplitude.
    plt.yscale('log')
    plt.xlabel("Time microseconds")
    plt.ylabel("Frequency [MHz]")
    plt.ylim(1.5,3.5)
    plt.title(name)
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {filepath}")
