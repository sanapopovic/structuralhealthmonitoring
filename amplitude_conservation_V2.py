import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocess
import decomp as d
from scipy.signal import hilbert

from transforms import Hilbert_Huang_processing
from transforms import SST_v2_processing


# -----------------------------
# CONFIG
# -----------------------------
noise_level = 0.0
beta_true = 10
distance = 0.2

A1_mode = "S2 Propagated signal (nm)"
A2_mode = "S4 Propagated signal (nm)"

modes_base = [
    "S2 Propagated signal (nm)",
    "A1 Propagated signal (nm)",
    "A4 Propagated signal (nm)"
]

modes_harmonic = [
    "S2 Propagated signal (nm)",
    "A1 Propagated signal (nm)",
    "A4 Propagated signal (nm)",
    "S4 Propagated signal (nm)"
]


dataset_base = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harm = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"


time_stamp_base = {"S2": 48.7581, "A1": 71.5683, "A4": 151.605}
time_stamp_harm = {"A1": 66.3568, "S4": 71.6575, "S2": 75.8788, "A4": 93.6327}


# -----------------------------
# AMPLITUDE (SINGLE DEFINITION)
# -----------------------------
def amplitude(x):
    return np.max(np.abs(hilbert(x)))


# -----------------------------
# BETA (CONSISTENT)
# -----------------------------
def compute_beta(A_S2, A_S4, beta_ref=10):

    k = (2 * np.pi * 1.33e6) / (8.303885011e3) * 1e-9

    beta = (A_S4 / (A_S2 ** 2)) * (8 / (distance * 1e9 * k**2))

    return abs(beta_ref - beta), beta


# -----------------------------
# TOA (UNCHANGED BUT CLEAN)
# -----------------------------
def ToA(recon_base, recon_harm, t):

    env_b = np.abs(hilbert(recon_base))
    env_h = np.abs(hilbert(recon_harm))

    res_b = d.align_to_envelope_with_time(env_b, t, time_stamp_base)
    res_h = d.align_to_envelope_with_time(env_h, t, time_stamp_harm)

    return res_b, res_h


# -----------------------------
# AMPLITUDE EXTRACTION FROM TOA
# -----------------------------
def amps(res_base, res_harm):
    return res_base["S2"]["peak_value"], res_harm["S4"]["peak_value"]


# =========================================================
# STFT (LINEAR, STABLE)
# =========================================================
def STFT(t, signal):

    return (
        SST_v2_processing.reconstruct_band_stft(
            t, signal,
            band_min=1.1e6,
            band_max=1.5e6,
            fmin=1e6,
            fmax=4.5e6,
            win_len=128,
            hop_len=2,
            n_fft=512
        ),
        SST_v2_processing.reconstruct_band_stft(
            t, signal,
            band_min=2.3e6,
            band_max=2.9e6,
            fmin=1e6,
            fmax=4.5e6,
            win_len=128,
            hop_len=2,
            n_fft=512
        )
    )


# =========================================================
# HHT
# =========================================================
def HHT(t, signal):

    dt = np.mean(np.diff(t))
    fs = 1.0 / dt * 1e6

    # IMPORTANT FIX:
    # no bandpass BEFORE EMD
    imf, _ = Hilbert_Huang_processing.emd(signal)

    recon_base = Hilbert_Huang_processing.bandpass_hilbert(
        imf, fs, 1.1e6, 1.5e6
    )

    recon_harm = Hilbert_Huang_processing.bandpass_hilbert(
        imf, fs, 2.3e6, 2.9e6
    )

    return recon_base, recon_harm


# =========================================================
# WAVELET
# =========================================================
def Wavelet(t, signal):

    return (
        SST_v2_processing.reconstruct_band_cwt(
            t, signal,
            band_min=1.1e6,
            band_max=1.5e6,
            wavelet="cmor7.5-5.5",
            fmin=1e6,
            fmax=4.5e6,
            n_freqs=400
        ),
        SST_v2_processing.reconstruct_band_cwt(
            t, signal,
            band_min=2.3e6,
            band_max=2.9e6,
            wavelet="cmor7.5-5.5",
            fmin=1e6,
            fmax=4.5e6,
            n_freqs=400
        )
    )


# -----------------------------
# PIPELINE
# -----------------------------
def run(method, t, signal):

    if method == "stft":
        rb, rh = STFT(t, signal)

    elif method == "hht":
        rb, rh = HHT(t, signal)

    elif method == "wavelet":
        rb, rh = Wavelet(t, signal)

    else:
        raise ValueError("unknown method")

    res_b, res_h = ToA(rb, rh, t)

    A_S2, A_S4 = amps(res_b, res_h)

    beta_err, beta_new = compute_beta(A_S2, A_S4)

    return A_S2, A_S4, beta_err, beta_new


# -----------------------------
# INITIAL VALUES (SAME METHOD)
# -----------------------------
def initial_values(data_base, data_harm):

    t, sig, _ = preprocess.create_signal(
        data_base, data_harm,
        beta_true, noise_level,
        A1_mode, A2_mode,
        modes_base, modes_harmonic,
        distance
    )

    rb, rh = STFT(t, sig)

    res_b, res_h = ToA(rb, rh, t)

    return amps(res_b, res_h)


# -----------------------------
# MAIN
# -----------------------------
def Main():

    #Plot noise level against beta error
    noise_list = [0,0.25,0.5,0.75,1,1.25,1.5]
    beta_error_list_stft = []
    beta_error_list_hht = []
    beta_error_list_wavelet = []

    for noise in noise_list:
        noise_level = noise


        data_base = preprocess.get_data(dataset_base)
        data_harm = preprocess.get_data(dataset_harm)

        t, signal, _ = preprocess.create_signal(
            data_base, data_harm,
            beta_true, noise_level,
            A1_mode, A2_mode,
            modes_base, modes_harmonic,
            distance
        )

        A_init_S2, A_init_S4 = initial_values(data_base, data_harm)

        print("INITIAL:", A_init_S2, A_init_S4)
        print("----------------------")

        for m in ["stft", "hht", "wavelet"]:

            A_S2, A_S4, beta_err, beta_new = run(m, t, signal)

            if m == "stft":
                beta_error_list_stft.append(beta_err)
            elif m == "hht":
                beta_error_list_hht.append(beta_err)
            else:
                beta_error_list_wavelet.append(beta_err)

            print(m.upper())
            print("S2:", A_S2)
            print("S4:", A_S4)
            print("beta new:",beta_new)
            print("beta error:", beta_err)
            print("----------------------")

    print(noise_list)
    print(beta_error_list_stft)
    plt.plot(noise_list,beta_error_list_stft)
    plt.xlabel("Noise Level")
    plt.ylabel("Beta Error")


    plt.plot(noise_list,beta_error_list_hht)
    plt.xlabel("Noise Level")
    plt.ylabel("Beta Error")


    plt.plot(noise_list,beta_error_list_wavelet)
    plt.xlabel("Noise Level")
    plt.ylabel("Beta Error")
    plt.show


Main()
