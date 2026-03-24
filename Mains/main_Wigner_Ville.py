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

W_raw = Wigner_Ville.wvd(signal)
W_sp = Wigner_Ville.spwvd(signal, 2, 2)
W_cw = Wigner_Ville.wvd_choi_williams(signal, sigma=0.5)
W_gauss = Wigner_Ville.wvd_gaussian(signal, sigma_t=0.3, sigma_f=0.3)

Wigner_Ville.plot_tfr(W_raw, "Raw Wigner-Ville", fs=fs)
Wigner_Ville.plot_tfr(W_sp, "Smoothed Pseudo WVD", fs=fs)
Wigner_Ville.plot_tfr(W_cw, "Choi-Williams", fs=fs)
Wigner_Ville.plot_tfr(W_gauss, "Gaussian Kernel WVD", fs=fs)