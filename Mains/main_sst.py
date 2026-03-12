import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocess
from transforms import stft_processing 
from transforms import sst_processing 


# ── Load real data ──────────────────────────────────────────────────────────
data = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")


t = data["Propagation time (micsec)"]
y = data["Sum Propagated signal (nm)"]


# ── Raw signal plot ─────────────────────────────────────────────────────────
preprocess.plot(t, y, 1, 'time_vs_volt')


# ── STFT (original, from functions.py) ─────────────────────────────────────
f, t_seg, amplitude, fs = stft_processing.stft(y, t)
stft_processing.plot_stft(f, t_seg, amplitude, downsampling=1, name="Spectrogram_version_sst_v1",dB=True)


# ── SST — sharpened version of the STFT above ──────────────────────────────
f_sst, t_sst, Tx_amp, Sx_amp, fs_sst = sst_processing.sst(y, t, win_len=256, hop_len=1)
sst_processing.plot_sst(f_sst, t_sst, Tx_amp, name="SST_Spectrogram_v1", dB=True)


# ── Inverse SST — reconstruct signal and compare ───────────────────────────
_, _, Tx_c, _, _ = sst_processing.sst_complex(y, t, win_len=256, hop_len=1)
x_rec = sst_processing.isst(Tx_c, win_len=256, hop_len=1)


t_np = t.to_numpy()
n = min(len(t_np), len(x_rec))
preprocess.plot(t_np[:n], x_rec[:n], 1, name="SST_Reconstructed_v1")
''' 
#Part2 2: Synchrosqueezing Transform (SST) and Inverse SST (iSST):
from sst_processing import get_data, stft, plot_stft, plot_sst, plot  #,sst_stft, sst_cwt,
import pandas as pd
import numpy as np


# --- load your real data ---
# data = get_data("your_file.csv")
# sig  = data["signal_column"]
# time = data["time_column"]


# --- or use a test signal ---
fs = 1000
t  = np.arange(0, 2, 1/fs)
x  = np.sin(2 * np.pi * (100 * t + 50 * t**2))


sig  = pd.Series(x)
time = pd.Series(t)


# --- run transforms ---
f_stft, t_stft, amp_stft, _          = stft(sig, time, nperseg=256)
f_sst,  t_sst,  sst_amp, stft_amp, _ = sst_stft(sig, time, nperseg=256)


# --- plot ---
plot_stft(f_stft, t_stft, amp_stft, name="my_stft", dB=True)
plot_sst(f_sst,   t_sst,  sst_amp,  name="my_sst",  dB=True)
''' 



















