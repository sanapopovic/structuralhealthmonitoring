import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, issq_cwt, Wavelet

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pywt

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
beta = 6
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
b_start, b_end, b_step = 0.5, 5.0, 0.5
# Range for Center Frequency (C) in cmorB-C
c_start, c_end, c_step = 0.5, 2.0, 0.2

# Defaults when not being swept
default_B = 1.5
default_C = 1.0

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & BUILD SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n[0] Loading data …")
data_base = preprocess.get_data(dataset_base)
data_harmonic = preprocess.get_data(dataset_harmonic)

print("[1] Building composite signal …")
t, signal, second_scale = preprocess.create_signal(
    data_base, data_harmonic, beta, noise_level,
    A1_mode, A2_mode, modes_base, modes_harmonic,
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
    recon_b = wavelet_processing.reconstruct_frequency_band(
        t, signal, band_min_base, band_max_base,
        wavelet=current_wavelet, fmin=f_min_analyse, fmax=f_max_analyse, n_freqs=n_freqs
    )
    
    # Reconstruct Harmonic
    recon_h = wavelet_processing.reconstruct_frequency_band(
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
print(f"\nDone — Plot saved to plots/wavelet_error_{parameter}.png")


# ═══════════════════════════════════════════════════════════════════════════
#  WAVELET TOURNAMENT
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, issq_cwt, Wavelet

# ─── 1. DEFINE THE HELPER FUNCTIONS FIRST ──────────────────────────────────

def reconstruct_band_sst(Tx, freqs, W, band_min, band_max):
    """
    Reconstructs signal using only the Synchrosqueezed bins within the band.
    """
    # Create mask for the frequency band
    mask = (freqs >= band_min) & (freqs <= band_max)
    Tx_masked = np.zeros_like(Tx)
    Tx_masked[mask, :] = Tx[mask, :]
    
    # Invert SST
    recon = issq_cwt(Tx_masked, W)
    return recon

# ─── 2. RUN THE TOURNAMENT ────────────────────────────────────────────────

wavelet_configs = [
    ('gmw', {'beta': 3}),   # Morse: standard
    ('gmw', {'beta': 10}),  # Morse: higher freq resolution
    ('morlet', {}),         # Standard Morlet
    ('bump', {}),           # Bump wavelet: very sharp frequency separation
]

wavelet_labels = ["Morse (β=3)", "Morse (β=10)", "Morlet", "Bump"]
final_errors = []

print(f"[4] Comparing {len(wavelet_configs)} different mother wavelets...")

# Calculate your sampling frequency once
fs = 1 / (np.mean(np.diff(t)) * 1e-6)

for i, (name, params) in enumerate(wavelet_configs):
    try:
        # Initialize Wavelet object
        W_obj = Wavelet((name, params))
        
        # Compute SST
        # Tx: Synchrosqueezed plane, Wx: original CWT
        Tx, Wx, freqs, scales, *_ = ssq_cwt(signal, W_obj, fs=fs)
        
        # 3. RECONSTRUCT (This is where the error was happening!)
        r_base = reconstruct_band_sst(Tx, freqs, W_obj, band_min_base, band_max_base)
        r_harm = reconstruct_band_sst(Tx, freqs, W_obj, band_min_harmonic, band_max_harmonic)
        
        # Calculate Error
        min_len = min(len(r_base), len(gt_base))
        err = np.sum(np.abs(r_base[:min_len] - gt_base[:min_len])) + \
              np.sum(np.abs(r_harm[:min_len] - gt_harmonic[:min_len]))
        
        final_errors.append(err)
        print(f"  ✓ {wavelet_labels[i]}: Total Error = {err:.4f}")
        
    except Exception as e:
        print(f"  ✗ Error testing {wavelet_labels[i]}: {e}")
        final_errors.append(np.nan)

# ─── 3. PLOT THE WINNER ───────────────────────────────────────────────────

plt.figure(figsize=(10, 5))
bars = plt.bar(wavelet_labels, final_errors, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

for bar in bars:
    yval = bar.get_height()
    if not np.isnan(yval):
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center')

plt.ylabel("Total Summed Absolute Error (nm)")
plt.title("Mother Wavelet Performance Comparison (via WSST)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("plots/wavelet_family_comparison.png", dpi=300)

print("\nComparison complete! Check 'plots/wavelet_family_comparison.png' to see which one won.")
