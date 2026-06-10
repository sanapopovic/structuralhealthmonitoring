import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pywt
from transforms.SST_v2_processing import cwt_sst, reconstruct_band_cwt, plot_reconstruction

# Make sure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocess
# Assuming the wavelet functions you provided are in a file named wavelet_processing.py
# If they are in the same script, just ensure they are defined above.
from transforms import wavelet_processing 

# ═══════════════════════════════════════════════════════════════════════════
#  USER PARAMETERS — Wavelet Specific
# ═══════════════════════════════════════════════════════════════════════════

# --- signal construction  -------------------------
noise_level = 0
beta = 10
A1_mode = "S2 Propagated signal (nm)"
A2_mode = "S4 Propagated signal (nm)"

dataset_base = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harmonic = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"

modes_base = ["S2 Propagated signal (nm)","A1 Propagated signal (nm)","A4 Propagated signal (nm)"]
modes_harmonic = ["S2 Propagated signal (nm)", "S4 Propagated signal (nm)", "A1 Propagated signal (nm)","A4 Propagated signal (nm)"]

# --- analysis parameters ---------------------------------------------------
f_min_analyse = 1.0e6      
f_max_analyse = 4.5e6      
n_freqs = 400              # Frequency resolution of the CWT grid

band_min_base = 1_100_000 
band_max_base = 1_500_000
band_min_harmonic = 2_300_000
band_max_harmonic = 2_900_000

# --- Wavelet Hyperparameters to Test ---------------------------------------
# options: "wavelet_bandwidth" or "wavelet_center_freq"
parameter = "wavelet_bandwidth" 

# Range for Bandwidth (B) in cmorB-C
b_start, b_end, b_step = 0.5, 13.0, 0.5
# Range for Center Frequency (C) in cmorB-C
c_start, c_end, c_step = 0.5, 5.0, 0.5

# Defaults when not being swept
default_B = 3.5
default_C = 5.5

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & BUILD SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n[0] Loading data …")
data_base = preprocess.get_data(dataset_base)
data_harmonic = preprocess.get_data(dataset_harmonic)

print("[1] Building composite signal …")
t, signal, second_scale = preprocess.create_signal(
    data_base, data_harmonic, beta, noise_level,
    A1_mode, A2_mode, modes_base, modes_harmonic, distance= 200
)

t_base = data_base["Propagation time (micsec)"].to_numpy()
t_harm = data_harmonic["Propagation time (micsec)"].to_numpy()

gt_base = np.zeros(len(t))
for mode in modes_base:
    gt_base += np.interp(t, t_base, data_base[mode].to_numpy())

gt_harmonic = np.zeros(len(t))
for mode in modes_harmonic:
    gt_harmonic += second_scale * np.interp(t, t_harm, data_harmonic[mode].to_numpy())

# ═══════════════════════════════════════════════════════════════════════════
#  ERROR PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_residuals(r_h, r_b, og_h, og_b):
    h_error = np.abs(r_h - og_h)
    b_error = np.abs(r_b - og_b)
    total_error = np.abs((r_h + r_b) - (og_h + og_b))
    return np.sum(h_error), np.sum(b_error), np.sum(total_error)

# ═══════════════════════════════════════════════════════════════════════════
#  EXECUTE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

results = {'par': [], 'h_err': [], 'b_err': [], 't_err': []}

if parameter == "wavelet_bandwidth":
    test_range = np.arange(b_start, b_end, b_step)
    xlabel = "Wavelet Bandwidth (B)"
elif parameter == "wavelet_center_freq":
    test_range = np.arange(c_start, c_end, c_step)
    xlabel = "Wavelet Center Frequency (C)"

print(f"[2] Evaluating {parameter}...")

for val in test_range:
    # Construct the cmorB-C string
    if parameter == "wavelet_bandwidth":
        current_wavelet = f"cmor{val}-{default_C}"
    else:
        current_wavelet = f"cmor{default_B}-{val}"
    
    # Reconstruct Base
    recon_b = reconstruct_band_cwt(
        t, signal, band_min_base, band_max_base,
        wavelet=current_wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freqs
    )
    
    # Reconstruct Harmonic
    recon_h = reconstruct_band_cwt(
        t, signal, band_min_harmonic, band_max_harmonic,
        wavelet=current_wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freqs
    )
    
    sum_h, sum_b, sum_t = get_residuals(recon_h, recon_b, gt_harmonic, gt_base)
    
    results['par'].append(val)
    results['h_err'].append(sum_h)
    results['b_err'].append(sum_b)
    results['t_err'].append(sum_t)
    print(f"  Tested {current_wavelet} -> Total Error: {sum_t:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

plt.figure(figsize=(10, 6))
plt.plot(results['par'], results['h_err'], 'o-', color="blue", label="Harmonic Error")
plt.plot(results['par'], results['b_err'], 'o-', color="red", label="Base Error")
plt.plot(results['par'], results['t_err'], 'o--', color="green", label="Total Error", alpha=0.7)

plt.xlabel(xlabel)
plt.ylabel("Summed Absolute Error (nm)")
plt.title(f"CWT Reconstruction Error vs {parameter}")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/wavelet_error_{parameter}.png", dpi=300)
print(f"\nDone — Plot saved to plots/wavelet_error_SST_{parameter}.png")