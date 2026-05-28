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
import scipy
from transforms import Hilbert_Huang_processing 
from transforms import wavelet_processing
from transforms import SST_v2_processing

"""
DICTIONARIES
"""
# "S2 Propagated signal (nm)"



#------------------------------
# Create initial signal modes: base S2 A1 A4, harmonic S2 S4 A1 A4

modes_base = ["S2 Propagated signal (nm)", "A1 Propagated signal (nm)", "A4 Propagated signal (nm)"]
modes_harmonic = ["S2 Propagated signal (nm)", "A1 Propagated signal (nm)", "A4 Propagated signal (nm)", "S4 Propagated signal (nm)"]

#--- Time of arrivals ---
time_stamp_base_200mm ={"S2": 48.7581,  "A1": 71.5683,  "A4": 151.605} #at 1.33 MHz
time_stamp_harmonic_200mm = {"A1": 66.3568,  "S4": 71.6575,  "S2": 75.8788,  "A4": 93.6327} #at 2.66 MHz
time_stamp_base_250mm ={"S2": 60.9477,  "A1": 89.4603,  "A4": 189.506} #at 1.33 MHz
time_stamp_harmonic_250mm = {"A1": 82.946,  "S4": 89.5719,  "S2": 94.8485,  "A4": 117.041} #at 2.66 MHz
time_stamp_base_300mm ={"S2": 73.1372,  "A1": 107.352,  "A4": 227.407} #at 1.33 MHz
time_stamp_harmonic_300mm = {"A1": 99.5352,  "S4": 107.486,  "S2": 113.818,  "A4": 140.449} #at 2.66 MHz
time_stamp_base_350mm ={"S2": 85.3267,  "A1": 125.244,  "A4": 265.309} #at 1.33 MHz
time_stamp_harmonic_350mm = {"A1": 116.124,  "S4": 125.401,  "S2": 132.788,  "A4": 163.857} #at 2.66 MHz


noise_level = 0 #Noise Level: 0 == 0%, 1.5 == 150%, should not be larger than 1.5
beta = 10
distance = 0.2

# Modes around which the beta parameter is taken, copy-paste from lists above
A1_mode = "S2 Propagated signal (nm)" # Mode of base harmonic
A2_mode = "S4 Propagated signal (nm)" # Mode of second harmonic

#data sets: as a string define which data set to be read
dataset_base_200mm = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harmonic_200mm = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"
dataset_base_250mm = "Data/In-plane_TemporalResponse@7.9866MHzmm@250mm.xlsx"
dataset_harmonic_250mm = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@250mm.xlsx"
dataset_base_300mm = "Data/In-plane_TemporalResponse@7.9866MHzmm@300mm.xlsx"
dataset_harmonic_300mm = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@300mm.xlsx"
dataset_base_350mm = "Data/In-plane_TemporalResponse@7.9866MHzmm@350mm.xlsx"
dataset_harmonic_350mm = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@350mm.xlsx"

#create the input/initial signal
data_harmonic = preprocess.get_data(dataset_harmonic_200mm)
data_base = preprocess.get_data(dataset_base_200mm)


t, signal, second_scale = preprocess.create_signal(data_base, data_harmonic, beta, noise_level, A1_mode, A2_mode, modes_base, modes_harmonic, distance)


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
#plt.show()


# max amp values inital wave (no noise, 200mm) 
#_max_S2 = 0.5387 # at 58.3349 microsec (1.33 MHz)
#_max_S4 = 1.7425 # at 75.3994 microsec ( 2.66 MHz)


# --- Reconstruction of the signals for each transform ---
"""Input initial signal ---> output reconstructed signal base and reconstructed signal harmonic""" 


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


# --- time of arrival ---

def ToA(recon_base, recon_harmonic, time_stamp_base, time_stamp_harmonic, noise_level, method):

    smoothing_factors = {
        "hht": {
            0.0:  (0, 0),
            0.25: (0, 0),
            0.5:  (2, 2),
            0.75: (3, 3),
            1.0:  (4, 4),
            1.25: (5, 5),
            1.5:  (6, 6),
        },
        "stft": {
            0.0:  (0, 0),
            0.25: (1, 1),
            0.5:  (2, 2),
            0.75: (3, 3),
            1.0:  (4, 4),
            1.25: (5, 5),
            1.5:  (6, 6),
        },
        "wt": {
            0.0:  (0, 0),
            0.25: (0, 0),
            0.5:  (1, 1),
            0.75: (2, 2),
            1.0:  (3, 3),
            1.25: (4, 4),
            1.5:  (5, 5),
        },
    }

    if method not in smoothing_factors:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(smoothing_factors.keys())}")
    if noise_level not in smoothing_factors[method]:
        raise ValueError(f"Unknown noise level '{noise_level}'.")

    k, j = smoothing_factors[method][noise_level]

    env = scipy.signal.envelope(recon_base)[0]
    for i in range(k):
        env, lower_env, mean_env = d.sift(env)

    plt.plot(t, recon_base)
    plt.plot(t, env)
    plt.show()

    result_base_H = d.align_to_envelope_with_time(env, t, time_stamp_base)

    env = scipy.signal.envelope(recon_harmonic)[0]
    for i in range(j):
        env, lower_env, mean_env = d.sift(env)

    plt.plot(t, recon_harmonic)
    plt.plot(t, env)
    plt.show()

    result_harmonic_H = d.align_to_envelope_with_time(env, t, time_stamp_harmonic)

    return result_base_H, result_harmonic_H

# --- Amplitude of S2 and S4 ---

def amps(result_base, result_harmonic):
    """Recovers amplitude values of S2 and S4"""
    S2_peak = result_base['S2']['peak_value']
    S4_peak = result_harmonic['S4']['peak_value']
    return S2_peak, S4_peak


# --- Amplitude difference --- 

def A_diff(A_max_S2_init, A_max_S4_init, A_max_S2_after, A_max_S4_after):
    """Finds difference (decrease) of the S2 and S4 max amplitudes (in % of initial amplitude value),
    Inputs| Max amplitude values before and after reconstruction
    Outputs| Percental decrease of S2 and S4 amplitudes
    """

    S2_diff = ((A_max_S2_init - A_max_S2_after)/ A_max_S2_init) * 100
    S4_diff = ((A_max_S4_init - A_max_S4_after)/ A_max_S4_init) * 100

    return S2_diff, S4_diff


# --- Beta parameter difference --- 

def Beta_diff(Amp_S2, Amp_S4, beta_predefined=6):
    """Calculates beta after transforms and compairs it to the pre defined beta (predefined to create initial wave)
    Inputs| Amplitude of S2 and S4 after reconstruction
    Outputs| Absolute value of the difference of the calculated beta compared to the predefined beta (=6), 
    """

    beta = Amp_S4 / (Amp_S2 ** 2)

    beta_diff = abs(beta_predefined - beta)

    return beta_diff


# --- Finding max amplitude S2 of initial signal ---

def A_max_S2_init(data_base, data_harmonic, noise_level):

    #Create signal with only S2 + noise is present
    modes_base = ["S2 Propagated signal (nm)"]
    modes_harmonic = []
    t, signal, second_scale = preprocess.create_signal(data_base, data_harmonic, beta, noise_level, A1_mode, A2_mode, modes_base, modes_harmonic, distance)

    #Determine amplitude of S2
    max_idx = np.argmax(signal) # Finds the array index of the highest value
    max_time = t[max_idx]            # Gets the corresponding time
    max_val = signal[max_idx]   # Gets the highest amplitude value

    return max_val
    

# --- Finding max amplitude S4 of initial signal ---

def A_max_S4_init(data_base, data_harmonic, noise_level):

    #Create signal with only S4 + noise is present
    modes_base = []
    modes_harmonic = ["S4 Propagated signal (nm)"]
    t, signal, second_scale = preprocess.create_signal(data_base, data_harmonic, beta, noise_level, A1_mode, A2_mode, modes_base, modes_harmonic, distance)

    #Determine amplitude of S2
    max_idx = np.argmax(signal) # Finds the array index of the highest value
    max_time = t[max_idx]            # Gets the corresponding time
    max_val = signal[max_idx]   # Gets the highest amplitude value

    return max_val








# --- Main function -----------------------------------------------
noise_lvl_list = [0.0]
def Main(Length,beta = 6):

    if Length == 200:
        dataset_base = dataset_base_200mm
        dataset_harmonic = dataset_harmonic_200mm
        time_stamp_base = time_stamp_base_200mm 
        time_stamp_harmonic = time_stamp_harmonic_200mm 

    elif Length == 250:
        dataset_base = dataset_base_250mm
        dataset_harmonic = dataset_harmonic_250mm
        time_stamp_base = time_stamp_base_250mm 
        time_stamp_harmonic = time_stamp_harmonic_250mm

    elif Length == 300:
        dataset_base = dataset_base_300mm
        dataset_harmonic = dataset_harmonic_300mm
        time_stamp_base = time_stamp_base_300mm 
        time_stamp_harmonic = time_stamp_harmonic_300mm

    elif Length == 350:
        dataset_base = dataset_base_350mm
        dataset_harmonic = dataset_harmonic_350mm
        time_stamp_base = time_stamp_base_350mm 
        time_stamp_harmonic = time_stamp_harmonic_350mm

    else: 
        return print("Incorrect Length input")


    data_base = preprocess.get_data(dataset_base)
    data_harmonic = preprocess.get_data(dataset_harmonic)
    
    amp_diff_stft_list = [] #in %
    amp_diff_hht_list = [] #in %
    amp_diff_wt_list = [] #in %
    beta_diff_stft_list = []
    beta_diff_hht_list = []
    beta_diff_wt_list = []
    for noise in noise_lvl_list:
        #Create signal
        t, signal, second_scale = preprocess.create_signal(data_base, data_harmonic, beta, noise, A1_mode, A2_mode, modes_base, modes_harmonic, distance)

        #Find max initial amplitudes
        A_max_S2 = A_max_S2_init(data_base, data_harmonic,0)
        A_max_S4 = A_max_S4_init(data_base, data_harmonic,0)

        #Implement transforms
        recon_base_stft, recon_harmonic_stft = STFT(t, signal)
        recon_base_hht, recon_harmonic_hht = HHT(t, signal)
        recon_base_wt, recon_harmonic_wt = Wavelet(t, signal)

        #Mode decomposition (time of arrival)
        result_base_stft, result_harmonic_stft = ToA(recon_base_stft, recon_harmonic_stft, time_stamp_base, time_stamp_harmonic, noise, "stft")
        result_base_hht, result_harmonic_hht = ToA(recon_base_hht, recon_harmonic_hht, time_stamp_base, time_stamp_harmonic, noise, "hht")
        result_base_wt, result_harmonic_wt = ToA(recon_base_wt, recon_harmonic_wt, time_stamp_base, time_stamp_harmonic, noise, "wt")

        #Amplitudes of decomposed S2 and S4
        A_S2_stft, A_S4_stft = amps(result_base_stft, result_harmonic_stft)
        A_S2_hht, A_S4_hht = amps(result_base_hht, result_harmonic_hht)
        A_S2_wt, A_S4_wt = amps(result_base_wt, result_harmonic_wt)

        #Amplitudes difference calculation
        S2_diff_stft, S4_diff_stft = A_diff(A_max_S2,A_max_S4, A_S2_stft, A_S4_stft) #in %
        S2_diff_hht, S4_diff_hht = A_diff(A_max_S2,A_max_S4, A_S2_hht, A_S4_hht) #in %
        S2_diff_wt, S4_diff_wt = A_diff(A_max_S2,A_max_S4, A_S2_wt, A_S4_wt) #in %

        amp_diff_stft_list.append([S2_diff_stft, S4_diff_stft])
        amp_diff_hht_list.append([S2_diff_hht, S4_diff_hht])
        amp_diff_wt_list.append([S2_diff_wt, S4_diff_wt])

        #Beta difference calculation
        Beta_diff_stft = Beta_diff(A_S2_stft, A_S4_stft)
        Beta_diff_hht = Beta_diff(A_S2_hht, A_S4_hht)
        Beta_diff_wt = Beta_diff(A_S2_wt, A_S4_wt)

        beta_diff_stft_list.append(Beta_diff_stft)
        beta_diff_hht_list.append(Beta_diff_hht)
        beta_diff_wt_list.append(Beta_diff_wt)


    print(f"Amp difference stft:{amp_diff_stft_list}")
    print(f"Amp difference hht:{amp_diff_hht_list}")
    print(f"Amp difference wavelet:{amp_diff_wt_list}")
    print("-------------------------------------")
    print(f"Beta difference stft:{beta_diff_stft_list}")
    print(f"Beta difference hht:{beta_diff_hht_list}")
    print(f"Beta difference wavelet:{beta_diff_wt_list}")

    return amp_diff_stft_list, amp_diff_hht_list, amp_diff_wt_list, beta_diff_stft_list, beta_diff_hht_list, beta_diff_wt_list


        
amp_diff_stft_list, amp_diff_hht_list, amp_diff_wt_list, beta_diff_stft_list, beta_diff_hht_list, beta_diff_wt_list = Main(200)





"""
# --- main pipeline -------------------------------------------------------------------

# Reconstructed Signals
recon_base_stft, recon_harmonic_stft = STFT(t, signal)
recon_base_hht, recon_harmonic_hht = HHT(t, signal)
recon_base_wt, recon_harmonic_wt = Wavelet(t, signal)

# --- Mode decomposition ---
result_base_stft, result_harmonic_stft = ToA(recon_base_stft, recon_harmonic_stft)
result_base_hht, result_harmonic_hht = ToA(recon_base_hht, recon_harmonic_hht)
result_base_wt, result_harmonic_wt = ToA(recon_base_wt, recon_harmonic_wt)

# --- Amplitudes S2 and S4 ---
A_S2_stft, A_S4_stft = amps(result_base_stft, result_harmonic_stft)
A_S2_hht, A_S4_hht = amps(result_base_hht, result_harmonic_hht)
A_S2_wt, A_S4_wt = amps(result_base_wt, result_harmonic_wt)

# --- Amplitudes difference ---
"""
# max amp values inital wave (no noise, 200mm) 
#A_max_S2 = 0.5387 # at 58.3349 microsec (1.33 MHz)
#A_max_S4 = 1.7425 # at 75.3994 microsec ( 2.66 MHz)
"""
S2_diff_stft, S4_diff_stft = A_diff(A_max_S2,A_max_S4, A_S2_stft, A_S4_stft)
S2_diff_hht, S4_diff_hht = A_diff(A_max_S2,A_max_S4, A_S2_hht, A_S4_hht)
S2_diff_wt, S4_diff_wt = A_diff(A_max_S2,A_max_S4, A_S2_wt, A_S4_wt)

# --- Beta difference ---
Beta_diff_stft = Beta_diff(A_S2_stft, A_S4_stft)
Beta_diff_hht = Beta_diff(A_S2_hht, A_S4_hht)
Beta_diff_wt = Beta_diff(A_S2_wt, A_S4_wt)


print(f"A Difference(S2,S4) in %| stft:{S2_diff_stft, S4_diff_stft},  hht:{S2_diff_hht, S4_diff_hht},  wavelet:{S2_diff_wt, S4_diff_wt}")

print(f"Beta difference| stft:{Beta_diff_stft},  hht:{Beta_diff_hht},  wavelet:{Beta_diff_wt}")
"""