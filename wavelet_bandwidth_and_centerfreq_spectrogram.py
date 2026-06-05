import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pywt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess
import time
import re
import decomp as d
from transforms import Hilbert_Huang_processing 
from transforms import wavelet_processing

# ═══════════════════════════════════════════════════════════════════════════
#  USER PARAMETERS — Wavelet Specific
# ═══════════════════════════════════════════════════════════════════════════

# --- signal construction  -------------------------
noise_level = 0
beta = 10
A1_mode = "S2 Propagated signal (nm)"
A2_mode = "S4 Propagated signal (nm)"

dataset_base = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harmonic = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"

modes_base = ["S2 Propagated signal (nm)","A1 Propagated signal (nm)","A4 Propagated signal (nm)"]
modes_harmonic = ["S2 Propagated signal (nm)", "S4 Propagated signal (nm)", "A1 Propagated signal (nm)","A4 Propagated signal (nm)"]



modes_harmonic = ["S2 Propagated signal (nm)", "S3 Propagated signal (nm)", "S4 Propagated signal (nm)", "A3 Propagated signal (nm)"]
modes_base = ["S2 Propagated signal (nm)", "S3 Propagated signal (nm)", "A3 Propagated signal (nm)"]
time_stamp_harmonic = {"S1": 68.4808,  "S2": 76.5127, "A4": 91.8515  , "S4": 64.9935, "A2": 71.7798, "S0": 70.7973, "A0": 70.7951}
time_stamp_base = {"S1": 68.4808,  "S2": 76.5127, "A4": 91.8515  , "S4": 64.9935, "A2": 71.7798, "S0": 70.7973, "A0": 70.7951}


# --- analysis parameters ---------------------------------------------------
f_min_analyse = 0.6e6      
f_max_analyse = 3.0e6      
n_freqs = 400              # Frequency resolution of the CWT grid

band_min_base = 1_100_000 
band_max_base = 1_500_000
band_min_harmonic = 2_300_000
band_max_harmonic = 2_900_000

# --- Wavelet Hyperparameters to Test ---------------------------------------
# options: "wavelet_bandwidth" or "wavelet_center_freq"
parameter = "wavelet_center_freq" 

# Range for Bandwidth (B) in cmorB-C
b_start, b_end, b_step = 0.5, 13.0, 0.5
# Range for Center Frequency (C) in cmorB-C
c_start, c_end, c_step = 1.0, 12.0, 0.5

# Defaults when not being swept
default_B = 3.5
default_C = 5.5

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & BUILD SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n[0] Loading data …")
data_base = preprocess.get_data(dataset_base)
data_harmonic = preprocess.get_data(dataset_harmonic)

print("[1] Building composite signal …")
t, signal, second_scale = preprocess.create_signal(
    data_base, data_harmonic, beta, noise_level,
    A1_mode, A2_mode, modes_base, modes_harmonic, distance=0.2
)

t_base = data_base["Propagation time (micsec)"].to_numpy()
t_harm = data_harmonic["Propagation time (micsec)"].to_numpy()

gt_base = np.zeros(len(t))
for mode in modes_base:
    gt_base += np.interp(t, t_base, data_base[mode].to_numpy())

gt_harmonic = np.zeros(len(t))
for mode in modes_harmonic:
    gt_harmonic += second_scale * np.interp(t, t_harm, data_harmonic[mode].to_numpy())


####


results = {'par': [], 'h_err': [], 'b_err': [], 't_err': []}

if parameter == "wavelet_bandwidth":
    test_range = np.arange(b_start, b_end, b_step)
    xlabel = "Wavelet Bandwidth (B)"
elif parameter == "wavelet_center_freq":
    test_range = np.arange(c_start, c_end, c_step)
    xlabel = "Wavelet Center Frequency (C)"

print(f"[2] Evaluating {parameter}...")

n_freq = 400 #How many freq bins



for val in test_range:
    # Construct the cmorB-C string
    if parameter == "wavelet_bandwidth":
        current_wavelet = f"cmor{val}-{default_C}"
    else:
        current_wavelet = f"cmor{default_B}-{val}"
    
    # Reconstruct Base
    recon_b = wavelet_processing.reconstruct_frequency_band(
        t, signal, band_min_base, band_max_base,
        wavelet=current_wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freqs
    )
    
    # Reconstruct Harmonic
    recon_h = wavelet_processing.reconstruct_frequency_band(
        t, signal, band_min_harmonic, band_max_harmonic,
        wavelet=current_wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freqs
    )
        
    wavelet_processing.wavelet_scalogram(t, signal, wavelet = current_wavelet, name= current_wavelet, fmin_mhz= f_min_analyse, fmax_mhz= f_max_analyse, n_freqs= n_freq)
    Recon_base_W = wavelet_processing.reconstruct_frequency_band(t, signal, band_min=band_min_base,band_max=band_max_base, wavelet=current_wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freq)
    Recon_harmonic_W = wavelet_processing.reconstruct_frequency_band(t, signal, band_min=band_min_harmonic,band_max=band_max_harmonic, wavelet=current_wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freq)

    upper_env, lower_env, mean_env = d.sift(Recon_base_W)
    result_base_W = d.align_to_envelope_with_time(mean_env, t, time_stamp_base)


    upper_env, lower_env, mean_env = d.sift(Recon_harmonic_W)
    result_harmonic_W = d.align_to_envelope_with_time(mean_env, t, time_stamp_harmonic)
                    
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, Recon_base_W)
    ax[0].set_title("Base Reconstruction")

    ax[1].plot(t, Recon_harmonic_W)
    ax[1].set_title("Harmonic Reconstruction")

    plt.tight_layout()

    print(max(signal))
    print(max(Recon_harmonic_W))


###




