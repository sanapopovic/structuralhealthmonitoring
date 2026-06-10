import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, issq_cwt, Wavelet
from transforms import SST_v2_processing          
import preprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
band_min_base = 1_100_000 
band_max_base = 1_500_000
band_min_harmonic = 2_300_000
band_max_harmonic = 2_900_000

# options: "wavelet_bandwidth" or "wavelet_center_freq"
parameter = "wavelet_center_freq" 

# Range for Morse Beta (Proxy for Bandwidth)
b_start, b_end, b_step = 0.5, 20, 0.5
# Range for Morlet Mu (Proxy for Center Frequency)
c_start, c_end, c_step = 0.5, 15.0, 0.5 

default_B = 3.5
default_C = 3.5

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & BUILD SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n[0] Loading data …")
data_base = preprocess.get_data(dataset_base)
data_harmonic = preprocess.get_data(dataset_harmonic)

print("[1] Building composite signal …")
t, signal, second_scale = preprocess.create_signal(
    data_base, data_harmonic, beta, noise_level,
    A1_mode, A2_mode, modes_base, modes_harmonic, distance=200
)

t_base = data_base["Propagation time (micsec)"].to_numpy()
t_harm = data_harmonic["Propagation time (micsec)"].to_numpy()

gt_base = np.zeros(len(t))
for mode in modes_base:
    gt_base += np.interp(t, t_base, data_base[mode].to_numpy())

gt_harmonic = np.zeros(len(t))
for mode in modes_harmonic:
    gt_harmonic += second_scale * np.interp(t, t_harm, data_harmonic[mode].to_numpy())

# Calculate sampling frequency once
fs = 1 / (np.mean(np.diff(t)) * 1e-6)

# ═══════════════════════════════════════════════════════════════════════════
#  EXECUTE EVALUATION (UPDATED TO USE SST)
# ═══════════════════════════════════════════════════════════════════════════

results = {'par': [], 'h_err': [], 'b_err': [], 't_err': []}

if parameter == "wavelet_bandwidth":
    test_range = np.arange(b_start, b_end, b_step)
    xlabel = "Wavelet Bandwidth (Morse Beta)"
elif parameter == "wavelet_center_freq":
    test_range = np.arange(c_start, c_end, c_step)
    xlabel = "Wavelet Center Frequency (Morlet Mu)"

print(f"[2] Evaluating {parameter} via WSST...")

for val in test_range:
    # Configure the ssqueezepy Wavelet Object dynamically
    if parameter == "wavelet_bandwidth":
        wavelet_label = f"cmor{val}-{default_C}"
        print('success')
    else:
        wavelet_label = f"cmor{default_B}-{val}" 
        print('success')
    
    try:

        # Reconstruct Bands via SST
        recon_b = SST_v2_processing.reconstruct_band_cwt(t, signal, band_min_base, band_max_base, wavelet_label)
        recon_h = SST_v2_processing.reconstruct_band_cwt(t, signal, band_min_harmonic, band_max_harmonic, wavelet_label)
        print('success2')

        # # Calculate Error handling potential length mismatches safely
        min_len_b = min(len(recon_b), len(gt_base))
        min_len_h = min(len(recon_h), len(gt_harmonic))
        
        print('success3')

        sum_b = np.sum(np.abs(recon_b[:min_len_b] - gt_base[:min_len_b]))
        sum_h = np.sum(np.abs(recon_h[:min_len_h] - gt_harmonic[:min_len_h]))
        sum_t = sum_b + sum_h
        
        results['par'].append(val)
        results['h_err'].append(sum_h)
        results['b_err'].append(sum_b)
        results['t_err'].append(sum_t)
        print(f"  Tested {wavelet_label} -> Total Error: {sum_t:.4f}")
        
    except Exception as e:
        print(f"  ✗ Error testing {wavelet_label}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

plt.figure(figsize=(10, 6))
plt.plot(results['par'], results['h_err'], 'o-', color="blue", label="Harmonic Error")
plt.plot(results['par'], results['b_err'], 'o-', color="red", label="Base Error")
plt.plot(results['par'], results['t_err'], 'o--', color="green", label="Total Error", alpha=0.7)

plt.xlabel(xlabel)
plt.ylabel("Summed Absolute Error (nm)")
plt.title(f"WSST Reconstruction Error vs {parameter}")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/wavelet_sst_error_{parameter}_NEW.png", dpi=300)
print(f"\nDone — Plot saved to plots/wavelet_sst_error_{parameter}_NEW.png")