import numpy as np
import scipy as sp
import functions as func
import matplotlib.pyplot as plt

#All files should be uploaded as csv
data = func.get_data(r"Data\In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")

t = data["Propagation time (micsec)"]
y = data["Sum Propagated signal (nm)"]

func.plot(t, y, 10, 'time_vs_volt')

f, t_seg, amplitude, fs = func.stft(y, t)

func.plot_stft(f, t_seg, amplitude, downsampling=1, name="Spectrogram")


#Hello