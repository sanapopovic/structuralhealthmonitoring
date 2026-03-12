
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
S, I, fs = stft_processing.stft(y, t)
stft_processing.plot_stft(S, fs, hop=64, downsampling=1, name="STFT_Spectogram",dB=True)