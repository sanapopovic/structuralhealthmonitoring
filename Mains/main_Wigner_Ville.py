import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess
import matplotlib.pyplot as plt
from transforms import Wigner_Ville

data = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")
#data = preprocess.get_data(r"Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.csv")


t = data["Propagation time (micsec)"].to_numpy()
signal = data["Sum Propagated signal (nm)"].to_numpy()

dt = np.mean(np.diff(t))
fs = (1.0 / dt)*(10**6)

#W = Wigner_Ville.wigner_ville_distribution(signal)

# Plot results
#Wigner_Ville.plot_wvd(signal, fs=fs)
Wigner_Ville.plot_spectrogram(signal, fs, use_db=True)