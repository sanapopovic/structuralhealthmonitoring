import numpy as np
import scipy as sp
import functions as func
import matplotlib.pyplot as plt

win_length = 512  # longer for better freq resolution
hop = 64  #longer for shorter computatuin time

#All files should be uploaded as csv
data = func.get_data(r"Data\In-plane_TemporalResponse@7.9866MHzmm@200mm.csv")

t = data["Propagation time (micsec)"]
y = data["Sum Propagated signal (nm)"]

y_std = func.std(y)

func.plot(t, y, 1, 'time_vs_volt')



S, x_rec, fs = func.stft(y, t, win_length= win_length, hop=hop)

func.plot_stft(S, fs=fs, hop=hop, dB=True, name="spectrogram")

f, amp = func.frequency_amplitude(S, fs=fs)

func.plot(f,amp,1, "Average Frequency Amplitude")

ridges = func.detect_ridges(S)

