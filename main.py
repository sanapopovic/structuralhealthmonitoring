import numpy as np
import scipy as sp
import preprocess
import transforms.stft_processing as stft_processing
import matplotlib.pyplot as plt

#All files should be uploaded as csv
data = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")

t = data["Propagation time (micsec)"]
y = data["Sum Propagated signal (nm)"]

preprocess.plot(t, y, 10, 'time_vs_volt')

f, t_seg, amplitude, fs = stft_processing.stft(y, t)

stft_processing.plot_stft(f, t_seg, amplitude, downsampling=1, name="spectrogram")
