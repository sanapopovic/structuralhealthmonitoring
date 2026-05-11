import pywt 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

 


def wavelet_scalogram(t, sig, wavelet = 'cgau2', n_scales=100, name="wavelet_scalogram"): #wavelet time–frequency map (CWT)
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
    return t, freqs, power