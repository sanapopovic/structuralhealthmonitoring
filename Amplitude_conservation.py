import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess
import time
import re
import decomp as d
from transforms import Hilbert_Huang_processing 
from transforms import wavelet_processing
from transforms import SST_v2_processing

"""
DICTIONARIES
"""
# "S2 Propagated signal (nm)"



#------------------------------
# Create initial signal modes: base S2 A1 A4, harmonic S2 S4 A1 A4
#200mm
modes_base = ["S2 Propagated signal (nm)", "A1 Propagated signal (nm)", "A4 Propagated signal (nm)"]
modes_harmonic = ["S2 Propagated signal (nm)", "A1 Propagated signal (nm)", "A4 Propagated signal (nm)", "S4 Propagated signal (nm)"]

time_stamp_base ={"S2": 48.7581,  "A1": 71.5683,  "A4": 151.605} #at 1.33 MHz
time_stamp_harmonic = {"A1": 66.3568,  "S4": 71.6575,  "S2": 75.8788,  "A4": 93.6327} #at 2.66 MHz


noise_level = 0 #Noise Level: 0 == 0%, 1.5 == 150%, should not be larger than 1.5
beta = 6

# Modes around which the beta parameter is taken, copy-paste from lists above
A1_mode = "S2 Propagated signal (nm)" # Mode of base harmonic
A2_mode = "S4 Propagated signal (nm)" # Mode of second harmonic

#data sets: as a string define which data set to be read
dataset_base = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harmonic = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"

#create the input/initial signal
data_harmonic = preprocess.get_data(dataset_harmonic)
data_base = preprocess.get_data(dataset_base)

t, signal, second_scale = preprocess.create_signal(data_base, data_harmonic, beta, noise_level, A1_mode, A2_mode, modes_base, modes_harmonic)


'''
# ---Finds the amplitude (Max value) ---
max_idx = np.argmax(signal) # Finds the array index of the highest value
max_time = t[max_idx]            # Gets the corresponding time
max_val = signal[max_idx]   # Gets the highest amplitude value

print(f"The max amplitude is {max_val} at {max_time} microsec.")
''' 

# Plot initial waveform
plt.plot(t, signal)
plt.title("Initial waveform: base(A1,S2,A4), harmonic(A1,S4,S2,A4)")
plt.xlabel("Time in microsec")
plt.show()


# max amp values inital wave (no noise, 200mm) 
A_max_S2 = 0.5387 # at 58.3349 microsec (1.33 MHz)
A_max_S4 = 1.7425 # at 75.3994 microsec ( 2.66 MHz)


# Reconstruction of the signals for each transform
#Input initial signal ---> output reconstructed signal base and reconstructed signal harmonic


#--- HHT ---

def HHT(t, signal):
    #Hilbert-Huang Specific:
    f_min_base = 1100000
    f_max_base = 1500000

    f_min_harmonic = 2300000
    f_max_harmonic = 2900000


    dt = np.mean(np.diff(t))
    fs = (1.0 / dt)*(10**6)

    imfs, residue = Hilbert_Huang_processing .emd(signal)

    Recon_base_H = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, f_min_base, f_max_base)
    Recon_harmonic_H = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, f_min_harmonic, f_max_harmonic)

    return Recon_base_H, Recon_harmonic_H


#--- STFT + SST ---

def STFT(t, signal):
    # --- time-frequency analysis -----------------------------------------------
    f_min_analyse = 1.0e6      # Hz — lower bound for TF display
    f_max_analyse = 4.5e6      # Hz — upper bound for TF display
    n_freq        = 400        # frequency bins (CWT)

    band_min_base      = 1_100_000   # Hz
    band_max_base      = 1_500_000
    band_min_harmonic  = 2_300_000
    band_max_harmonic  = 2_900_000

    # --- STFT parameters (from blind-decomp stage 1) ---------------------------
    stft_win_len = 128     # samples
    stft_hop_len = 2
    stft_n_fft   = 512


    # --- band reconstructions ---
    recon_base_stft = SST_v2_processing.reconstruct_band_stft(
        t, signal,
        band_min=band_min_base,
        band_max=band_max_base,
        fmin=f_min_analyse,
        fmax=f_max_analyse,
        win_len=stft_win_len,
        hop_len=stft_hop_len,
        n_fft=stft_n_fft,
    )

    recon_harmonic_stft = SST_v2_processing.reconstruct_band_stft(
        t, signal,
        band_min=band_min_harmonic,
        band_max=band_max_harmonic,
        fmin=f_min_analyse,
        fmax=f_max_analyse,
        win_len=stft_win_len,
        hop_len=stft_hop_len,
        n_fft=stft_n_fft,
    )

    return recon_base_stft, recon_harmonic_stft


#--- Wavelet + SST ---

def Wavelet(t, signal):
    # --- CWT wavelet -----------------------------------------------------------
    wavelet = "cmor3.0-1.0"

    f_min_analyse = 1.0e6      # Hz — lower bound for TF display
    f_max_analyse = 4.5e6      # Hz — upper bound for TF display
    n_freq        = 400        # frequency bins (CWT)

    band_min_base      = 1_100_000   # Hz
    band_max_base      = 1_500_000
    band_min_harmonic  = 2_300_000
    band_max_harmonic  = 2_900_000

    # --- band reconstructions ---
    recon_base_cwt = SST_v2_processing.reconstruct_band_cwt(
        t, signal,
        band_min=band_min_base,
        band_max=band_max_base,
        wavelet=wavelet,
        fmin=f_min_analyse,
        fmax=f_max_analyse,
        n_freqs=n_freq,
    )

    recon_harmonic_cwt = SST_v2_processing.reconstruct_band_cwt(
        t, signal,
        band_min=band_min_harmonic,
        band_max=band_max_harmonic,
        wavelet=wavelet,
        fmin=f_min_analyse,
        fmax=f_max_analyse,
        n_freqs=n_freq,
    )

    return recon_base_cwt, recon_harmonic_cwt


# Reconstructed Signals
recon_base_stft, recon_harmonic_stft = STFT(t, signal)
recon_base_hht, recon_harmonic_hht = HHT(t, signal)
recon_base_wt, recon_harmonic_wt = Wavelet(t, signal)


# --- time of arrival ---

def ToA(recon_base, recon_harmonic):
    upper_env, lower_env, mean_env = d.sift(recon_base)
    result_base_H = d.align_to_envelope_with_time(mean_env, t, time_stamp_base)

    upper_env, lower_env, mean_env = d.sift(recon_harmonic)
    result_harmonic_H = d.align_to_envelope_with_time(mean_env, t, time_stamp_harmonic)

    return result_base_H, result_harmonic_H



"""
# --- time of arrival ---
#input reconstructed signal base
#output 

upper_env, lower_env, mean_env = d.sift(---input---)
result_base_H = d.align_to_envelope_with_time(mean_env, t, time_stamp_base)


#input reconstructed signal harmonic
#output

upper_env, lower_env, mean_env = d.sift(---input---)
result_harmonic_H = d.align_to_envelope_with_time(mean_env, t, time_stamp_harmonic)

"""