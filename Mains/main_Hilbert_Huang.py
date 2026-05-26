import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess
import matplotlib.pyplot as plt
import decomp as d
import scipy.signal as s
from transforms import Hilbert_Huang_processing 


#DO NOT EDIT THESE DICTIONARIES OR I WILL BE VERY ANGRY!!!
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
modes_base = ["S2 Propagated signal (nm)", "A1 Propagated signal (nm)", "A4 Propagated signal (nm)"]
modes_harmonic = ["S2 Propagated signal (nm)", "A1 Propagated signal (nm)", "A4 Propagated signal (nm)", "S4 Propagated signal (nm)"]

time_stamp_base ={"S2": 48.7581,  "A1": 71.5683,  "A4": 151.605} #at 1.33 MHz
time_stamp_harmonic = {"A1": 66.3568,  "S4": 71.6575,  "S2": 75.8788,  "A4": 93.6327} #at 2.66 MHz

noise_level = 1.5 #Noise Level: 0 == 0%, 1.5 == 150%, should not be larger than 1.5
beta = 6

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

plot_name = 'test'

# Code below

data_harmonic = preprocess.get_data(dataset_harmonic)
data_base = preprocess.get_data(dataset_base)

t, signal, data_harmonic = preprocess.create_signal(data_base, data_harmonic, beta, noise_level, "S2 Propagated signal (nm)", "S4 Propagated signal (nm)", modes_base, modes_harmonic )


dt = np.mean(np.diff(t))
fs = (1.0 / dt)*(10**6)


imfs, residue = Hilbert_Huang_processing .emd(signal)
inst_amp, inst_freq = Hilbert_Huang_processing .hilbert_analysis(imfs, fs)
fig, ax, H, T, F = Hilbert_Huang_processing.plot_hilbert_spectrum(inst_freq, inst_amp, t, fs, log_amplitude=log_amplitude, f_bins=f_bins ,t_bins=t_bins, name= plot_name)

Recon_harmonic_H = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, f_min_harmonic, f_max_harmonic)
Recon_base_H = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, f_min_base, f_max_base)

upper_env, lower_env, mean_env = d.sift(Recon_base_H)
result_base_H = d.align_to_envelope_with_time(mean_env, t, time_stamp_base)


upper_env, lower_env, mean_env = d.sift(Recon_harmonic_H)
#env = d.sliding_rms_envelope(Recon_harmonic_H, int(fs/(2.6*10e6)))
env = s.envelope(Recon_harmonic_H)
env = env[0]
env, lower_env, mean_env = d.sift(env)
env, lower_env, mean_env = d.sift(env)
env, lower_env, mean_env = d.sift(env)
#env = s.envelope(Recon_harmonic_H)
#env = d.lock_in_envelope(Recon_harmonic_H, fs, 2.66*10e6)
#env = d.kalman_envelope(Recon_harmonic_H, fs, 2.66*10e6)
#env = d.amplitude_envelope(Recon_harmonic_H)
#env = d.smooth_envelope(Recon_harmonic_H, fs,2.6*10e6)
result_harmonic_H = d.align_to_envelope_with_time(mean_env, t, time_stamp_harmonic)



# plot IMFs
plt.figure(figsize=(10,6))
plt.subplot(len(imfs)+1,1,1)
plt.plot(t, signal)
plt.title("Original Signal")

for i, imf in enumerate(imfs):
    plt.subplot(len(imfs)+1,1,i+2)
    plt.plot(t, imf)
    plt.title(f"IMF {i+1}")

plt.tight_layout()
plt.show()

plt.plot(t, signal)
plt.show()

fig, ax = plt.subplots(2, 1)
ax[0].plot(t, Recon_base_H)
ax[0].set_title("Base Reconstruction")


ax[1].plot(t, Recon_harmonic_H)
ax[1].plot(t,env)
ax[1].set_title("Harmonic Reconstruction")

plt.tight_layout()
plt.show()


imfs, residue = Hilbert_Huang_processing .emd(Recon_harmonic_H)
inst_amp, inst_freq = Hilbert_Huang_processing .hilbert_analysis(imfs, fs)
fig, ax, H, T, F = Hilbert_Huang_processing.plot_hilbert_spectrum(inst_freq, inst_amp, t, fs, log_amplitude=log_amplitude, f_bins=f_bins ,t_bins=t_bins, name= plot_name)






