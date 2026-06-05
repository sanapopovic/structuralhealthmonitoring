import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, issq_cwt, Wavelet

import preprocess

# Make sure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
band_min_base = 1_100_000 
band_max_base = 1_500_000
band_min_harmonic = 2_300_000
band_max_harmonic = 2_900_000

# --- Wavelet Hyperparameters to Test ---------------------------------------
# options: "wavelet_bandwidth" or "wavelet_center_freq"
parameter = "wavelet_center_freq" 

# Range for Morse Beta (Proxy for Bandwidth)
b_start, b_end, b_step = 0.5, 20, 0.5
# Range for Morlet Mu (Proxy for Center Frequency)
c_start, c_end, c_step = 2.0, 15.0, 0.5 

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
#  SST RECONSTRUCTION HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_band_sst(Tx, freqs, W, band_min, band_max):
    """Reconstructs signal using only the Synchrosqueezed bins within the band."""
    mask = (freqs >= band_min) & (freqs <= band_max)
    Tx_masked = np.zeros_like(Tx)
    Tx_masked[mask, :] = Tx[mask, :]
    return issq_cwt(Tx_masked, W)

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
        W_obj = Wavelet(('gmw', {'beta': val, 'gamma': 3}))
        wavelet_label = f"Morse(β={val})"
    else:
        W_obj = Wavelet(('morlet', {'mu': val}))
        wavelet_label = f"Morlet(μ={val})"
    
    try:
        # Compute SST
        Tx, Wx, freqs, scales, *_ = ssq_cwt(signal, W_obj, fs=fs)
        
        # Reconstruct Bands via SST
        recon_b = reconstruct_band_sst(Tx, freqs, W_obj, band_min_base, band_max_base)
        recon_h = reconstruct_band_sst(Tx, freqs, W_obj, band_min_harmonic, band_max_harmonic)
        
        # Calculate Error handling potential length mismatches safely
        min_len_b = min(len(recon_b), len(gt_base))
        min_len_h = min(len(recon_h), len(gt_harmonic))
        
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
plt.savefig(f"plots/wavelet_sst_error_{parameter}.png", dpi=300)
print(f"\nDone — Plot saved to plots/wavelet_sst_error_{parameter}.png")

# ═══════════════════════════════════════════════════════════════════════════
#  WAVELET TOURNAMENT (STAYS FOR COMPARISON)
# ═══════════════════════════════════════════════════════════════════════════

wavelet_configs = [
    ('gmw', {'beta': 3}),   
    ('gmw', {'beta': 10}),  
    ('morlet', {}),         
    ('bump', {}),           
]

wavelet_labels = ["Morse (β=3)", "Morse (β=10)", "Morlet", "Bump"]
final_errors = []

print(f"\n[4] Comparing {len(wavelet_configs)} different mother wavelets...")

for i, (name, params) in enumerate(wavelet_configs):
    try:
        W_obj = Wavelet((name, params))
        Tx, Wx, freqs, scales, *_ = ssq_cwt(signal, W_obj, fs=fs)
        
        r_base = reconstruct_band_sst(Tx, freqs, W_obj, band_min_base, band_max_base)
        r_harm = reconstruct_band_sst(Tx, freqs, W_obj, band_min_harmonic, band_max_harmonic)
        
        min_len = min(len(r_base), len(gt_base))
        err = np.sum(np.abs(r_base[:min_len] - gt_base[:min_len])) + \
              np.sum(np.abs(r_harm[:min_len] - gt_harmonic[:min_len]))
        
        final_errors.append(err)
        print(f"  ✓ {wavelet_labels[i]}: Total Error = {err:.4f}")
    except Exception as e:
        print(f"  ✗ Error testing {wavelet_labels[i]}: {e}")
        final_errors.append(np.nan)

plt.figure(figsize=(10, 5))
bars = plt.bar(wavelet_labels, final_errors, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

for bar in bars:
    yval = bar.get_height()
    if not np.isnan(yval):
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center')

plt.ylabel("Total Summed Absolute Error (nm)")
plt.title("Mother Wavelet Performance Comparison (via WSST)")
plt.tight_layout()
plt.savefig("plots/wavelet_family_comparison.png", dpi=300)

print("\nTournament complete! Check 'plots/wavelet_family_comparison.png'")