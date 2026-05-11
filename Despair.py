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

noise_start = 0
noise_end = 1.5
noise_level_step = 0.5 #Noise Level: 0 == 0%, 1.5 == 150%, should not be larger than 1.5
beta_start = 7 #Non_Linearity Parameter: Realistic Range 6-12
beta_step = 2
beta_end = 11

# Modes around which the beta parameter is taken, copy-paste from lists above
A1_mode = "S2 Propagated signal (nm)" # Mode of base harmonic
A2_mode = "S4 Propagated signal (nm)" # Mode of second harmonic

#data sets: as a string define which data set to be read
dataset_base = ["Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"]
dataset_harmonic = ["Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"]

#Hilbert-Huang Specific:
f_min_base = 1100000
f_max_base = 1500000

f_min_harmonic = 2300000
f_max_harmonic = 2900000

f_bins = 2000
t_bins = 600
log_amplitude = True

#Wavelet Specific:

wavelet = 'cmor3.0-1.0' #Type of wavelet to be used
f_min_analyse = 1.0e6 #In Hz
f_max_analyse = 4.5e6 #In Hz
n_freq = 400 #How many freq bins

band_min_base = 1100000
band_max_base = 1500000

band_min_harmonic = 2300000
band_max_harmonic = 2900000

#SST Specific:

'''
Start Code: DO NOT EDIT BELOW THIS LINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

'''

def safe_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def detect_plane_type(filename: str) -> str:
    match = re.search(r"(In-plane|Out-of-plane)", filename, re.IGNORECASE)
    return match.group(1) if match else "Unknown"


def The_Function(noise_start, noise_level_step,noise_stop, beta_start, beta_step, beta_end, dataset_base, dataset_harmonic):

    Result_base_H = {}
    Result_harmonic_H = {}
    Result_base_W = {}
    Result_harmonic_W = {}
    Result_base_S = {}
    Result_harmonic_S = {}

    for name_base, name_harmonic in zip(dataset_base, dataset_harmonic):

        beta = beta_start
        data_harmonic = preprocess.get_data(name_harmonic)
        data_base = preprocess.get_data(name_base)

        distance = name_base.split("@")[2].split("mm")[0]
        data_type = detect_plane_type(name_base)

        while beta <= beta_end:
            noise_level = noise_start
            while noise_level <= noise_stop:

                t, signal, second_scale = preprocess.create_signal(data_base, data_harmonic, beta, noise_level, A1_mode, A2_mode, modes_base, modes_harmonic )

                dt = np.mean(np.diff(t))
                fs = (1.0 / dt)*(10**6)
                print(fs)

                plot_name_H = safe_filename(f"Hilbert-Huang Noise Level {noise_level}, Beta {beta}, Distance {distance}mm {data_type}")
                plot_name_W = safe_filename(f"Wavelet Noise Level {noise_level}, Beta {beta}, Distance {distance}mm {data_type}")
                plot_name_S = safe_filename(f"SST Noise Level {noise_level}, Beta {beta}, Distance {distance}mm {data_type}")
                #Hilbert_Huang

                t0 = time.perf_counter()

                imfs, residue = Hilbert_Huang_processing .emd(signal)
                inst_amp, inst_freq = Hilbert_Huang_processing .hilbert_analysis(imfs, fs)
                fig, ax, H, T, F = Hilbert_Huang_processing.plot_hilbert_spectrum(inst_freq, inst_amp, t, fs, log_amplitude=log_amplitude, f_bins=f_bins ,t_bins=t_bins, name= plot_name_H)

                Recon_harmonic_H = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, f_min_base, f_max_harmonic)
                Recon_base_H = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, f_min_base, f_max_base)

                upper_env, lower_env, mean_env = d.sift(Recon_base_H)
                result_base_H = d.align_to_envelope_with_time(mean_env, t, time_stamp_base)
                result_base_H["scale"] = second_scale
                result_base_H["reconstruction"] = Recon_base_H
                result_base_H["t"] = t

                upper_env, lower_env, mean_env = d.sift(Recon_harmonic_H)
                result_harmonic_H = d.align_to_envelope_with_time(mean_env, t, time_stamp_harmonic)
                result_harmonic_H["scale"] = second_scale
                result_harmonic_H["reconstruction"] = Recon_harmonic_H
                result_harmonic_H["t"] = t

                Result_base_H[plot_name_H] = result_base_H
                Result_harmonic_H[plot_name_H] = result_harmonic_H
                

                t1 = time.perf_counter()


                #Wavelet

                wavelet_processing.wavelet_scalogram(t, signal, wavelet = wavelet, name= plot_name_W, fmin_mhz= f_min_analyse, fmax_mhz= f_max_analyse, n_freqs= n_freq)
                Recon_base_W = wavelet_processing.reconstruct_frequency_band(t, signal, band_min=band_min_base,band_max=band_max_base, wavelet=wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freq)
                Recon_harmonic_W = wavelet_processing.reconstruct_frequency_band(t, signal, band_min=band_min_harmonic,band_max=band_max_harmonic, wavelet=wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freq)

                upper_env, lower_env, mean_env = d.sift(Recon_base_W)
                result_base_W = d.align_to_envelope_with_time(mean_env, t, time_stamp_base)
                result_base_W["scale"] = second_scale
                result_base_W["reconstruction"] = Recon_base_W
                result_base_W["t"] = t

                upper_env, lower_env, mean_env = d.sift(Recon_harmonic_W)
                result_harmonic_W = d.align_to_envelope_with_time(mean_env, t, time_stamp_harmonic)
                result_harmonic_W["scale"] = second_scale
                result_harmonic_W["reconstruction"] = Recon_harmonic_W
                result_harmonic_W["t"] = t

                Result_base_W[plot_name_W] = result_base_W
                Result_harmonic_W[plot_name_W] = result_harmonic_W

                t2 = time.perf_counter()

                #SST

                Result_base_S[plot_name_S] = result_base_W
                Result_harmonic_S[plot_name_S] = result_harmonic_W

                t3 = time.perf_counter()

                print("Elapsed time for Hilbert-Huang:", t1-t0, "seconds")
                print("Elapsed time for Wavelet:", t2-t1, "seconds")
                print("Elapsed time for STFT+SST:", t3-t2, "seconds")

                if noise_level_step ==0:
                    break
                else:
                    noise_level += noise_level_step
            
            if beta_step ==0:
                break
            else:
                beta +=beta_step

    return Result_base_H, Result_harmonic_H ,Result_base_W, Result_harmonic_W, Result_base_S, Result_harmonic_S

def time_eval(Result_base_H, Result_harmonic_H ,Result_base_W, Result_harmonic_W, Result_base_S, Result_harmonic_S):
    
    Eval_base = {}
    Eval_harmonic = {}
    
    for instance in Result_base_H:
        Eval_base[instance] = {}
        time_base = []
        for element in Result_base_H[instance]:
             
            if element == "scale" or  element == "reconstruction" or element == "t" :
                continue
            if element is None:
                continue

            time_base.append(abs(Result_base_H[instance][element]["peak_time"]-time_stamp_base[element]))
                                                                                            
        time_base = np.array(time_base)
        Eval_base[instance]["mean"] = np.mean(time_base)
        Eval_base[instance]["std"] = np.std(time_base)
        
    for instance in Result_base_W:
        Eval_base[instance] = {}
        time_base = []
        for element in Result_base_W[instance]:
             
            if element == "scale" or  element == "reconstruction" or element == "t":
                continue
            if element is None:
                continue

            time_base.append(abs(Result_base_W[instance][element]["peak_time"]-time_stamp_base[element]))
                                                                                            
        time_base = np.array(time_base)
        Eval_base[instance]["mean"] = np.mean(time_base)
        Eval_base[instance]["std"] = np.std(time_base)
        
    for instance in Result_base_S:
        Eval_base[instance] = {}
        time_base = []
        for element in Result_base_S[instance]:
             
            if element == "scale" or  element == "reconstruction" or element == "t":
                continue
            if element is None:
                continue

            time_base.append(abs(Result_base_S[instance][element]["peak_time"]-time_stamp_base[element]))
                                                                                            
        time_base = np.array(time_base)
        Eval_base[instance]["mean"] = np.mean(time_base)
        Eval_base[instance]["std"] = np.std(time_base)
    
    for instance in Result_harmonic_H:
        Eval_harmonic[instance] = {}
        time_base = []
        for element in Result_harmonic_H[instance]:
             
            if element == "scale" or  element == "reconstruction" or element == "t":
                continue
            if element is None:
                continue

            time_base.append(abs(Result_harmonic_H[instance][element]["peak_time"]-time_stamp_harmonic[element]))
                                                                                            
        time_base = np.array(time_base)
        Eval_harmonic[instance]["mean"] = np.mean(time_base)
        Eval_harmonic[instance]["std"] = np.std(time_base)
    for instance in Result_harmonic_W:
        Eval_harmonic[instance] = {}
        time_base = []
        for element in Result_harmonic_W[instance]:
             
            if element == "scale" or  element == "reconstruction" or element == "t":
                continue
            if element is None:
                continue

            time_base.append(abs(Result_harmonic_W[instance][element]["peak_time"]-time_stamp_harmonic[element]))
                                                                                            
        time_base = np.array(time_base)
        Eval_harmonic[instance]["mean"] = np.mean(time_base)
        Eval_harmonic[instance]["std"] = np.std(time_base)
    for instance in Result_harmonic_S:
        Eval_harmonic[instance] = {}
        time_base = []
        for element in Result_harmonic_S[instance]:
             
            if element == "scale" or  element == "reconstruction" or element == "t":
                continue
            if element is None:
                continue

            time_base.append(abs(Result_harmonic_S[instance][element]["peak_time"]-time_stamp_harmonic[element]))
                                                                                            
        time_base = np.array(time_base)
        Eval_harmonic[instance]["mean"] = np.mean(time_base)
        Eval_harmonic[instance]["std"] = np.std(time_base)
    
        

    return Eval_base, Eval_harmonic
            

Result_base_H, Result_harmonic_H ,Result_base_W, Result_harmonic_W, Result_base_S, Result_harmonic_S = The_Function(noise_start,noise_level_step,noise_end, beta_start, beta_step, beta_end, dataset_base, dataset_harmonic)

Eval_base, Eval_harmonic = time_eval(Result_base_H, Result_harmonic_H ,Result_base_W, Result_harmonic_W, Result_base_S, Result_harmonic_S)
'''
END CODE: BELOW THIS LINE WRITE EVALUATION CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
print(Result_harmonic_H["Hilbert-Huang Noise Level 0, Beta 9, Distance 200mm In-plane"])
print(Eval_harmonic["Wavelet Noise Level 1.5, Beta 9, Distance 200mm In-plane"])