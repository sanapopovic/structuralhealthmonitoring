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

"""
DICTIONARIES
"""
# "S2 Propagated signal (nm)"



#------------------------------
# Create initial signal modes: base S2 A1 A4, harmonic S2 S4 A1 A4
#200mm
modes_base = ["S2 Propagated signal (nm)", "A1 Propagated signal (nm)", "A4 Propagated signal (nm)"]
modes_harmonic = []

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

#Wavelet Specific:

wavelet = 'cmor3.0-1.0' #Type of wavelet to be used
f_min_analyse = 1.0e6 #In Hz
f_max_analyse = 4.5e6 #In Hz
n_freq = 400 #How many freq bins

band_min_base = 1100000
band_max_base = 1500000

band_min_harmonic = 2300000
band_max_harmonic = 2900000

#create the input/initial signal
data_harmonic = preprocess.get_data(dataset_harmonic)
data_base = preprocess.get_data(dataset_base)

t, signal, second_scale = preprocess.create_signal(data_base, data_harmonic, beta, noise_level, A1_mode, A2_mode, modes_base, modes_harmonic)


# Plot initial waveform
plt.plot(t, signal)
plt.title("initial waveform")
plt.show()