import pywt 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from ssqueezepy import cwt, icwt, Wavelet



def wavelet_scalogram_OG(t, sig, wavelet = 'cgau2', n_scales=100, name="wavelet_scalogram"): 
    #wavelet time–frequency map (CWT) 
    sig = np.asarray(sig).squeeze() 
    t = np.asarray(t).squeeze() 
    dt = np.mean(np.diff(t)) 
    #sampling period 
    widths = np.geomspace(1, 512, n_scales) 
    #Creates n_scales wavelet scales from 1 to 512 on a logarithmic grid 
    cwtmatr, freqs = pywt.cwt(sig, widths, wavelet, sampling_period=dt) 
    #Runs the CWT. 2D matrix (frequency × time), freqs holds the corresponding frequencies in Hz. 
    power = np.abs(cwtmatr) #Takes magnitude so you can plot amplitude/energy. 
    folder = "plots" 
    os.makedirs(folder, exist_ok=True) 
    plt.figure(figsize=(8,4)) 
    plt.pcolormesh(t, freqs, power, shading='gouraud') #Draws a heatmap: x = time, y = frequency, color = amplitude. 
    plt.yscale('linear') 
    plt.xlabel("Time microseconds") 
    plt.ylabel("Frequency [MHz]") 
    plt.ylim(1,4.5) 
    plt.title(name) 
    plt.colorbar(label="Amplitude") 
    plt.tight_layout() 
    filepath = os.path.join(folder, f"{name}.png") 
    plt.savefig(filepath, dpi=300) 
    plt.close() 
    print(f"Plot saved to {filepath}")

def wavelet_scalogram(
        
    t_us,                      # time in microseconds
    sig,
    wavelet='cgau2',
    fmin_mhz=1.0e6,
    fmax_mhz=4.5e6,
    n_freqs=600,
    name="wavelet_scalogram"
):


    #fmin_mhz and fmax_mhz is in Hz not MHz
    # ------------------------------------------------
    # Convert inputs
    # ------------------------------------------------
    sig = np.asarray(sig).squeeze()

    # Convert microseconds -> seconds
    t = np.asarray(t_us).squeeze() * 1e-6

    # ------------------------------------------------
    # Sampling period in seconds
    # ------------------------------------------------
    dt = np.mean(np.diff(t))

    # ------------------------------------------------
    # Desired frequencies in Hz
    # ------------------------------------------------
    freqs_target = np.linspace(
        fmin_mhz,
        fmax_mhz,
        n_freqs
    )

    # ------------------------------------------------
    # Compute scales correctly
    # ------------------------------------------------
    fc = pywt.central_frequency(wavelet)

    scales = fc / (freqs_target * dt)

    # ------------------------------------------------
    # Continuous Wavelet Transform
    # ------------------------------------------------
    cwtmatr, freqs = pywt.cwt(
        sig,
        scales,
        wavelet,
        sampling_period=dt
    )

    # ------------------------------------------------
    # Convert frequencies to MHz
    # ------------------------------------------------
    freqs_mhz = freqs / 1e6

    # ------------------------------------------------
    # Magnitude
    # ------------------------------------------------
    power = np.abs(cwtmatr)

    # ------------------------------------------------
    # Create folder
    # ------------------------------------------------
    folder = "plots"
    os.makedirs(folder, exist_ok=True)

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    plt.figure(figsize=(10, 5))

    plt.pcolormesh(
        t_us,                  # original microseconds
        freqs_mhz,
        power,
        shading='gouraud'
    )

    plt.xlabel("Time [µs]")
    plt.ylabel("Frequency [MHz]")
    plt.title(name)

    plt.ylim(fmin_mhz/1e6, fmax_mhz/1e6)

    plt.colorbar(label="Amplitude")

    plt.tight_layout()

    # ------------------------------------------------
    # Save
    # ------------------------------------------------
    filepath = os.path.join(folder, f"{name}.png")

    plt.savefig(filepath, dpi=300)

    plt.close()

    print(f"Plot saved to {filepath}")

def reconstruct_frequency_band(
    t_us,
    sig,
    band_min,
    band_max,
    wavelet='cmor3.0-1.0',
    fmin=1e6,
    fmax=4.5e6,
    n_freqs=300
):
    """
    CWT band reconstruction with amplitude preservation (no normalization).
    """

    # -----------------------------
    # Convert inputs
    # -----------------------------
    sig = np.asarray(sig).squeeze()
    t = np.asarray(t_us).squeeze() * 1e-6  # µs → s

    dt = np.mean(np.diff(t))

    # -----------------------------
    # Frequency grid (Hz)
    # -----------------------------
    freqs_target = np.linspace(fmin, fmax, n_freqs)

    # -----------------------------
    # Scale conversion
    # -----------------------------
    fc = pywt.central_frequency(wavelet)
    scales = fc / (freqs_target * dt)

    # -----------------------------
    # CWT
    # -----------------------------
    cwtmatr, freqs = pywt.cwt(
        sig,
        scales,
        wavelet,
        sampling_period=dt
    )

    # -----------------------------
    # Band selection
    # -----------------------------
    mask = (freqs >= band_min) & (freqs <= band_max)

    cwt_band = cwtmatr[mask, :]
    scales_band = scales[mask]

    # -----------------------------
    # Amplitude-preserving reconstruction
    # -----------------------------
    reconstructed =np.real( np.sum(
    cwt_band / (scales_band[:, None]**2),
    axis=0))


    reconstructed *= np.mean(np.diff(np.log(scales_band)))

    return reconstructed


