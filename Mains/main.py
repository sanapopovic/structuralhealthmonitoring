import numpy as np
import scipy as sp
import preprocess
import transforms.stft_processing as stft_processing
import matplotlib.pyplot as plt
from transforms.wavelet_processing import wavelet_decompose
from transforms.wavelet_processing import wavelet_scalogram

#All files should be uploaded as csv
data = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")

t = data["Propagation time (micsec)"]
y = data["Sum Propagated signal (nm)"]

preprocess.plot(t, y, 10, 'time_vs_volt')

f, t_seg, amplitude, fs = stft_processing.stft(y, t)

stft_processing.plot_stft(f, t_seg, amplitude, downsampling=1, name="spectrogram", dB=True)

coeffs = wavelet_decompose(y, wavelet='db4', level=None)

# grab first detail level (finest scale)
cD1 = coeffs[-1]

# simple time axis to match its length
t_cD1 = np.linspace(t.iloc[0], t.iloc[-1], len(cD1))

preprocess.plot(t_cD1, cD1, 1, 'wavelet_detail_L1')

wavelet_scalogram(t, y, wavelet='cmor1.5-1.0', n_scales=100, name="wavelet_scalogram")
