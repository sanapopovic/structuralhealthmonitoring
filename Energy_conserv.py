import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess
import matplotlib.pyplot as plt
import time
import re
import decomp as d
from transforms import Hilbert_Huang_processing 
from transforms import wavelet_processing




#DO NOT EDIT THESE DICTIONARIES OR I WILL BE VERY ANGRY!!!
all_modes_harmonic = ["S0 Propagated signal (nm)", "S1 Propagated signal (nm)", "S2 Propagated signal (nm)", "S3 Propagated signal (nm)", "S4 Propagated signal (nm)",
         "S5 Propagated signal (nm)", "S6 Propagated signal (nm)", "S7 Propagated signal (nm)", "S8 Propagated signal (nm)", "A0 Propagated signal (nm)",
         "A1 Propagated signal (nm)", "A2 Propagated signal (nm)", "A3 Propagated signal (nm)", "A4 Propagated signal (nm)", "A5 Propagated signal (nm)",
         "A7 Propagated signal (nm)"]
all_modes_base = ["S0 Propagated signal (nm)", "S1 Propagated signal (nm)", "S2 Propagated signal (nm)", "S3 Propagated signal (nm)",
                  "A0 Propagated signal (nm)", "A1 Propagated signal (nm)", "A2 Propagated signal (nm)", "A3 Propagated signal (nm)",
                  "A4 Propagated signal (nm)"]
all_time_stamp200_base = {"S2": 46.9173,  "S3": 58.3698,  "A0": 70.4733, "S0": 71.1597,  "A1": 72.0227,  "A3": 78.9802,  "A2": 85.0981,  "S1": 85.6671}
all_time_stamp200_harmonic = {"S1": 68.4843,  "S2": 76.5262, "A4": 91.8005  , "S4": 64.8534, "A2": 71.7873, "S0": 70.7973, "A0": 70.7951, "A1": 66.4039, "A3": 83.1089,
                 "A5": 50.7399, "A6": 73.3755, "A7": 71.8784, "S3": 91.1849, "S5": 42.8571, "S6": 58.9979, "S7": 168.058, "S8": 188.394}
all_time_stamp250_base = {"S2": 58.6466,  "S3": 72.9622,  "A0": 88.0916,  "S0": 88.9496,  "A1": 90.0284,  "A3": 98.7252,  "A2": 106.373,  "S1": 107.084}
all_time_stamp250_harmonic = {"S5": 53.5714,  "A5": 63.4248,  "S6": 73.7474,  "S4": 81.0668,  "A1": 83.0049,  "S1": 85.6053,  "A0": 88.4938, "S0": 88.4967, 
                              "A2": 89.7341,  "A7": 89.8481,  "A6": 91.7193,  "S2": 95.6578,  "A3": 103.886,  "S3": 113.981,  "A4": 114.751,  "S7": 210.072,  "S8": 235.493}
all_time_stamp300_base = {"S2": 70.3759,  "S3": 87.5547,  "A0": 105.71,  "S0": 106.74,  "A1": 108.034,  "A3": 118.47,  "A2": 127.647,  "S1": 128.501}
all_time_stamp300_harmonic = {"S5": 64.2857,  "A5": 76.1098,  "S6": 88.4968,  "S4": 97.2801,  "A1": 99.6058,  "S1": 102.726,  "A0": 106.193,  "S0": 106.196,  "A2": 107.681,  
                              "A7": 107.818,  "A6": 110.063,  "S2": 114.789,  "A3": 124.663,  "S3": 136.777,  "A4": 137.701,  "S7": 252.087,  "S8": 282.591}
all_time_stamp350_base = {"S2": 82.1053,  "S3": 102.147,  "A0": 123.328,  "S0": 124.529,  "A1": 126.04,  "A3": 138.215,  "A2": 148.922,  "S1": 149.917}
all_time_stamp350_harmonic = {"S5": 75.00,  "A5": 88.7948,  "S6": 103.246,  "S4": 113.493,  "A1": 116.207,  "S1": 119.847,  "A0": 123.891,  "S0": 123.895,  "A2": 125.628, 
                              "A7": 125.787,  "A6": 128.407,  "S2": 133.921,  "A3": 145.441,  "S3": 159.574,  "A4": 160.651,  "S7": 294.101,  "S8": 329.69}


'''
ABOVE ARE ALL THE POSSIBLE MODES FOR THE BASE AND SECOND HARMONIC SHOWN, COPY-PASTE THE ENTIRE STRING INTO THE LIST BELOW. THE MODES IN THE LISTS BELOW
WILL BE INCLUDED IN THE SIGNAL. NEVER ALTER THE LISTS AND DICTIONARIES ABOVE!!!!!!!!!. ABOVE IS ALSO THE TIME OF ARRIVAL SHOWN FOR 200mm ONLY,
THE MODES YOU WANT TO BE FOUND IN THE SIGNAL SHOULD BE INCLUDED IN time_stamp AS A DICTIONARY

'''
modes_harmonic = ["S2 Propagated signal (nm)", "S3 Propagated signal (nm)", "S4 Propagated signal (nm)",
         "A3 Propagated signal (nm)"]
modes_base = ["S2 Propagated signal (nm)", "S3 Propagated signal (nm)",
                  "A3 Propagated signal (nm)",
                  ]


noise_level = 0
 #Noise Level: 0 == 0%, 1.5 == 150%, should not be larger than 1.5
beta = 7 #Non_Linearity Parameter: Realistic Range 6-12


# Modes around which the beta parameter is taken, copy-paste from lists above
A1_mode = "S2 Propagated signal (nm)" # Mode of base harmonic
A2_mode = "S4 Propagated signal (nm)" # Mode of second harmonic

#data sets: as a string define which data set to be read
dataset_base = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harmonic = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"

#Hilbert-Huang Specific:
f_min_base = 1100000
f_max_base = 1500000

f_min_harmonic = 2300000
f_max_harmonic = 2900000

f_bins = 2000
t_bins = 600
log_amplitude = True


plot_name1 = "HHT"
plot_name2 = "WT"

#Wavelet Specific:

wavelet = 'cmor3.0-1.0' #Type of wavelet to be used
f_min_analyse = 1.0e6 #In Hz
f_max_analyse = 4.5e6 #In Hz
n_freq = 400 #How many freq bins

band_min_base = 1100000
band_max_base = 1500000

band_min_harmonic = 2300000
band_max_harmonic = 2900000

# Simulated Signal

data_harmonic = preprocess.get_data(dataset_harmonic)
data_base = preprocess.get_data(dataset_base)

t, signal, data_harmonic = preprocess.create_signal(data_base, data_harmonic, beta, noise_level, "S2 Propagated signal (nm)", "S4 Propagated signal (nm)", modes_base, modes_harmonic )


dt = np.mean(np.diff(t))
fs = (1.0 / dt)*(10**6)

# Reconstruction HHT

imfs, residue = Hilbert_Huang_processing .emd(signal)
inst_amp, inst_freq = Hilbert_Huang_processing .hilbert_analysis(imfs, fs)
fig, ax, H, T, F = Hilbert_Huang_processing.plot_hilbert_spectrum(inst_freq, inst_amp, t, fs, log_amplitude=log_amplitude, f_bins=f_bins ,t_bins=t_bins, name= plot_name1)

Recon_harmonic_H = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, f_min_base, f_max_harmonic)
Recon_base_H = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, f_min_base, f_max_base)

# Reconstruction Wavelet

wavelet_processing.wavelet_scalogram(t, signal, wavelet = wavelet, name= plot_name2, fmin_mhz= f_min_analyse, fmax_mhz= f_max_analyse, n_freqs= n_freq)
Recon_base_W = wavelet_processing.reconstruct_frequency_band(t, signal, band_min=band_min_base,band_max=band_max_base, wavelet=wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freq)
Recon_harmonic_W = wavelet_processing.reconstruct_frequency_band(t, signal, band_min=band_min_harmonic,band_max=band_max_harmonic, wavelet=wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freq)
