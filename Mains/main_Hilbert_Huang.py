import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess
import matplotlib.pyplot as plt
from transforms import Hilbert_Huang_processing 


#data = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")
data = preprocess.get_data(r"Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.csv")


t = data["Propagation time (micsec)"].to_numpy()
signal = data["Sum Propagated signal (nm)"].to_numpy()

dt = np.mean(np.diff(t))
fs = (1.0 / dt)*(10**6)

imfs, residue = Hilbert_Huang_processing .emd(signal)

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

Hilbert_Huang_processing .plot_hilbert_spectrum(inst_freq, inst_amp, t)


#Imf_r = Hilbert_Huang_processing.extract_imf_at_frequency(signal, fs= fs, freq=2600000, bandwidth=200000  ) #NOT HILBERT BUT BANDWIDTH FILTER
Imf_r = Hilbert_Huang_processing.imf_extraction(imfs, inst_freq, 1300000, 200000)
print("The amount of possible IMF's are:",len(Imf_r))
Imf_r = Imf_r[0]


plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original")

plt.subplot(2, 1, 2)
plt.plot(t, Imf_r)
plt.title("Filtered IMF")

plt.tight_layout()
plt.show()
