import numpy as np
import scipy as sp
import preprocess
import transforms.stft_processing as stft_processing
import matplotlib.pyplot as plt

win_length = 512  # longer for better freq resolution
hop = 64  #longer for shorter computatuin time

#All files should be uploaded as csv
data = preprocess.get_data(r"Data\In-plane_TemporalResponse@7.9866MHzmm@200mm.csv")

t = data["Propagation time (micsec)"]
y = data["Sum Propagated signal (nm)"]

y_std = stft_processing.std(y)

preprocess.plot(t, y, 1, 'time_vs_volt')



S, x_rec, fs = stft_processing.stft(y, t, win_length= win_length, hop=hop)

stft_processing.plot_stft(S, fs=fs, hop=hop, dB=True, name="spectrogram")

f, amp = stft_processing.frequency_amplitude(S, fs=fs)

preprocess.plot(f,amp,1, "Average Frequency Amplitude")

ridges = stft_processing.detect_ridges(S)

