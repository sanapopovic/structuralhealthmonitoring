from transforms.wavelet_processing import wavelet_scalogram

import numpy as np
import scipy as sp
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import preprocess
import transforms.stft_processing as stft_processing
import matplotlib.pyplot as plt
from transforms.wavelet_processing import wavelet_scalogram


#All files should be uploaded as csv
data = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")

t = data["Propagation time (micsec)"] #time axis in microseconds for this experiment
y = data["Sum Propagated signal (nm)"] #measured signal, in nanometres

time, freq, amp = wavelet_scalogram(t, y, wavelet = 'cgau2', n_scales=100, name="wavelet_scalogram") #Runs the Continuous Wavelet Transform (CWT) using a complex Morlet wavelet



# guesses = [] #in f-t
# peaks = []

# for g in guesses:
# #define subdomain of amp through guess +-10 in both t and freq
# #find global maximum
#     peaks.append()



#other method with scipy


print(amp)

# sp.signal.find_peaks(amp, distance=)