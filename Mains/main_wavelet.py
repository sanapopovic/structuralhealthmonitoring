import numpy as np
import scipy as sp
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import preprocess
import transforms.stft_processing as stft_processing
import matplotlib.pyplot as plt
from transforms.wavelet_processing import wavelet_decompose
from transforms.wavelet_processing import wavelet_scalogram

#All files should be uploaded as csv
data = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")

t = data["Propagation time (micsec)"] #time axis in microseconds for this experiment
y = data["Sum Propagated signal (nm)"] #measured signal, in nanometres

coeffs = wavelet_decompose(y, wavelet='db4', level=None) #Runs a multilevel Discrete Wavelet Transform on y using the Daubechies‑4 wavelet

cD1 = coeffs[-1] #grab first detail level (finest scale)

#simple time axis to match its length
t_cD1 = np.linspace(t.iloc[0], t.iloc[-1], len(cD1))
plt.figure(figsize=(8,4))
plt.plot(t_cD1, cD1, linewidth=1)
plt.xlabel("Time [µs]")
plt.ylabel("Wavelet detail coefficient (level 1)")
plt.title("wavelet_detail_L1")
plt.savefig("plots/wavelet_detail_L1.png", dpi=300, bbox_inches='tight')
plt.close()



wavelet_scalogram(t, y, wavelet='cmor1.5-1.0', n_scales=100, name="wavelet_scalogram") #Runs the Continuous Wavelet Transform (CWT) using a complex Morlet wavelet
