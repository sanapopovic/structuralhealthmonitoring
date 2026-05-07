import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess
import matplotlib.pyplot as plt
import decomp as d
from transforms import Hilbert_Huang_processing 


data_harmonic = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx")
data_base = preprocess.get_data(r"Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx")
#data = preprocess.get_data(r"Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx")



modes_harmonic = ["S0 Propagated signal (nm)", "S1 Propagated signal (nm)", "S2 Propagated signal (nm)", "S3 Propagated signal (nm)", "S4 Propagated signal (nm)",
         "S5 Propagated signal (nm)", "S6 Propagated signal (nm)", "S7 Propagated signal (nm)", "S8 Propagated signal (nm)", "A0 Propagated signal (nm)",
         "A1 Propagated signal (nm)", "A2 Propagated signal (nm)", "A3 Propagated signal (nm)", "A4 Propagated signal (nm)", "A5 Propagated signal (nm)",
         "A7 Propagated signal (nm)"]
modes_base = ["S0 Propagated signal (nm)", "S1 Propagated signal (nm)", "S2 Propagated signal (nm)", "S3 Propagated signal (nm)",
                  "A0 Propagated signal (nm)", "A1 Propagated signal (nm)", "A2 Propagated signal (nm)", "A3 Propagated signal (nm)",
                  "A4 Propagated signal (nm)"]

time_stamp200 = {"S1": 68.4808,  "S2": 76.5127, "A4": 91.8515  , "S4": 64.9935, "A2": 71.7798, "S0": 70.7973, "A0": 70.7951}
time_stamp250 = {"S1":119.841, "S2": 133.897}


t, signal, data_harmonic = preprocess.create_signal(data_base, data_harmonic, 0, 0, "S2 Propagated signal (nm)", "S4 Propagated signal (nm)", modes_base, modes_harmonic )


#t = data_base["Propagation time (micsec)"]
#signal = data_base["Sum Propagated signal (nm)"]
#signal = preprocess.noise(signal, 20)


dt = np.mean(np.diff(t))
fs = (1.0 / dt)*(10**6)

imfs, residue = Hilbert_Huang_processing .emd(signal)
print(len(imfs))

inst_amp, inst_freq = Hilbert_Huang_processing .hilbert_analysis(imfs, fs)

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

fig, ax, H, T, F = Hilbert_Huang_processing.plot_hilbert_spectrum(inst_freq, inst_amp, t, fs)


#Imf_r = Hilbert_Huang_processing.Bandpass(signal, fs= fs, freq=2600000, bandwidth=200000  ) # BANDWIDTH FILTER
#Imf_r = Hilbert_Huang_processing.reconstruction(inst_amp, inst_freq, fs, 2400000,2800000)
#Imf_r = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, 2300000,2900000)
Imf_r = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, 1100000, 1500000)
#Imf_a = Hilbert_Huang_processing.imf_extraction(imfs, inst_freq, 2600000, 200000)
#print("The amount of possible IMF's are:",len(Imf_r))
#Imf_a = Imf_a[0]

upper_env, lower_env, mean_env = d.sift(Imf_r)

#result = d.align_to_envelope_with_time(mean_env, t , time_stamp200)
result = d.peak_time(mean_env, t, time_stamp200)

#decomposed = d.Hann_decomp(result, 'Hann', L = 10, threshold=0.1)



print(result['S1'])

s0 = data_harmonic["S0 Propagated signal (nm)"].to_numpy()
a0 = data_harmonic["A2 Propagated signal (nm)"].to_numpy()
s4 = data_harmonic["S4 Propagated signal (nm)"].to_numpy()
s1 = data_harmonic["S1 Propagated signal (nm)"].to_numpy()
s2 = data_harmonic["S2 Propagated signal (nm)"].to_numpy()
s5 = data_harmonic["S5 Propagated signal (nm)"].to_numpy()
a4 = data_harmonic["A4 Propagated signal (nm)"].to_numpy()
print('S0', np.max(s0), 'S1', np.max(s1), 'S2', np.max(s2), 'A4:', np.max(a4))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original")

plt.subplot(2, 1, 2)
plt.plot(t, Imf_r)
plt.title("Filtered IMF")

plt.tight_layout()
plt.show()

