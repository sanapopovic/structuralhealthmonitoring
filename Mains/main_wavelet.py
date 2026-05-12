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




'''
DO NOT EDIT THESE DICTIONARIES OR I WILL BE VERY ANGRY!!!
'''
all_modes_harmonic = ["S0 Propagated signal (nm)", "S1 Propagated signal (nm)", "S2 Propagated signal (nm)", "S3 Propagated signal (nm)", "S4 Propagated signal (nm)",
         "S5 Propagated signal (nm)", "S6 Propagated signal (nm)", "S7 Propagated signal (nm)", "S8 Propagated signal (nm)", "A0 Propagated signal (nm)",
         "A1 Propagated signal (nm)", "A2 Propagated signal (nm)", "A3 Propagated signal (nm)", "A4 Propagated signal (nm)", "A5 Propagated signal (nm)",
         "A7 Propagated signal (nm)"]
all_modes_base = ["S0 Propagated signal (nm)", "S1 Propagated signal (nm)", "S2 Propagated signal (nm)", "S3 Propagated signal (nm)",
                  "A0 Propagated signal (nm)", "A1 Propagated signal (nm)", "A2 Propagated signal (nm)", "A3 Propagated signal (nm)",
                  "A4 Propagated signal (nm)"]
all_time_stamp200_harmonic = {"S1": 68.4808,  "S2": 76.5127, "A4": 93.4223  , "S4": 64.9935, "A2": 71.7798, "S0": 70.7973, "A0": 70.7951, "A1": 66.3644, "A3": 82.2203,
                 "A5": 52.0663, "A6": 66.8718, "A7": 68.4287, "S3": 90.1157, "S5": 40.8027, "S6": 59.2187, "S7": 148.571, "S8": 125.379}



'''
ABOVE ARE ALL THE POSSIBLE MODES FOR THE BASE AND SECOND HARMONIC SHOWN, COPY-PASTE THE ENTIRE STRING INTO THE LIST BELOW. THE MODES IN THE LISTS BELOW
WILL BE INCLUDED IN THE SIGNAL. NEVER ALTER THE LISTS AND DICTIONARIES ABOVE!!!!!!!!!. ABOVE IS ALSO THE TIME OF ARRIVAL SHOWN FOR 200mm ONLY,
THE MODES YOU WANT TO BE FOUND IN THE SIGNAL SHOULD BE INCLUDED IN time_stamp AS A DICTIONARY

'''
modes_harmonic = ["S0 Propagated signal (nm)", "S1 Propagated signal (nm)", "S2 Propagated signal (nm)", "S3 Propagated signal (nm)", "S4 Propagated signal (nm)",
         "S5 Propagated signal (nm)", "S6 Propagated signal (nm)", "S7 Propagated signal (nm)", "S8 Propagated signal (nm)", "A0 Propagated signal (nm)",
         "A1 Propagated signal (nm)", "A2 Propagated signal (nm)", "A3 Propagated signal (nm)", "A4 Propagated signal (nm)", "A5 Propagated signal (nm)",
         "A7 Propagated signal (nm)"]
modes_base = ["S0 Propagated signal (nm)", "S1 Propagated signal (nm)", "S2 Propagated signal (nm)", "S3 Propagated signal (nm)",
                  "A0 Propagated signal (nm)", "A1 Propagated signal (nm)", "A2 Propagated signal (nm)", "A3 Propagated signal (nm)",
                  "A4 Propagated signal (nm)"]
time_stamp_harmonic = {"S1": 68.4808,  "S2": 76.5127, "A4": 91.8515  , "S4": 64.9935, "A2": 71.7798, "S0": 70.7973, "A0": 70.7951}
time_stamp_base = {"S1": 68.4808,  "S2": 76.5127, "A4": 91.8515  , "S4": 64.9935, "A2": 71.7798, "S0": 70.7973, "A0": 70.7951}

noise_level = 0 #Noise Level: 0 == 0%, 1.5 == 150%, should not be larger than 1.5
beta = 6

# Modes around which the beta parameter is taken, copy-paste from lists above
A1_mode = "S2 Propagated signal (nm)" # Mode of base harmonic
A2_mode = "S4 Propagated signal (nm)" # Mode of second harmonic

#data sets: as a string define which data set to be read
dataset_base = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harmonic = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"


#Wavelet Specific:

wavelet = 'cmor3.0-1.0' #Type of wavelet to be used
f_min_analyse = 1.0e6 #In Hz
f_max_analyse = 4.5e6 #In Hz
n_freq = 400 #How many freq bins

band_min_base = 1100000
band_max_base = 1500000

band_min_harmonic = 2300000
band_max_harmonic = 2900000


plot_name = 'test'
#Code below


data_harmonic = preprocess.get_data(dataset_harmonic)
data_base = preprocess.get_data(dataset_base)

t, signal, second_scale = preprocess.create_signal(data_base, data_harmonic, beta, noise_level, A1_mode, A2_mode, modes_base, modes_harmonic )


wavelet_processing.wavelet_scalogram(t, signal, wavelet = wavelet, name= plot_name, fmin_mhz= f_min_analyse, fmax_mhz= f_max_analyse, n_freqs= n_freq)
Recon_base_W = wavelet_processing.reconstruct_frequency_band(t, signal, band_min=band_min_base,band_max=band_max_base, wavelet=wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freq)
Recon_harmonic_W = wavelet_processing.reconstruct_frequency_band(t, signal, band_min=band_min_harmonic,band_max=band_max_harmonic, wavelet=wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freq)

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
plt.show()
print(max(signal))
print(max(Recon_harmonic_W))