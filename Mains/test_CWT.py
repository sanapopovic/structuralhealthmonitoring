"""
test_cwt_reconstruction.py
──────────────────────────
Focused test to get a correct CWT band reconstruction.
Tries multiple reconstruction formulas and prints diagnostics
so we can see which one matches the ground truth best.

Run from project root.
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess

os.makedirs("plots", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════════

noise_level = 0.0    # clean signal so we can judge reconstruction purely
beta        = 9

A1_mode = "S2 Propagated signal (nm)"
A2_mode = "S4 Propagated signal (nm)"

dataset_base     = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harmonic = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"

modes_base = [
    "S0 Propagated signal (nm)", "S1 Propagated signal (nm)",
    "S2 Propagated signal (nm)", "S3 Propagated signal (nm)",
    "A0 Propagated signal (nm)", "A1 Propagated signal (nm)",
    "A2 Propagated signal (nm)", "A3 Propagated signal (nm)",
    "A4 Propagated signal (nm)",
]
modes_harmonic = [
    "S0 Propagated signal (nm)", "S1 Propagated signal (nm)",
    "S2 Propagated signal (nm)", "S3 Propagated signal (nm)",
    "S4 Propagated signal (nm)", "S5 Propagated signal (nm)",
    "S6 Propagated signal (nm)", "S7 Propagated signal (nm)",
    "S8 Propagated signal (nm)", "A0 Propagated signal (nm)",
    "A1 Propagated signal (nm)", "A2 Propagated signal (nm)",
    "A3 Propagated signal (nm)", "A4 Propagated signal (nm)",
    "A5 Propagated signal (nm)", "A7 Propagated signal (nm)",
]

wavelet      = "cmor3.0-1.0"
fmin         = 1.0e6
fmax         = 4.5e6
n_freqs      = 400

band_min_base     = 1_100_000
band_max_base     = 1_500_000
band_min_harmonic = 2_300_000
band_max_harmonic = 2_900_000


# ═══════════════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════════

print("[0] Loading …")
data_base     = preprocess.get_data(dataset_base)
data_harmonic = preprocess.get_data(dataset_harmonic)

t, signal, second_scale = preprocess.create_signal(
    data_base, data_harmonic,
    beta, noise_level,
    A1_mode, A2_mode,
    modes_base, modes_harmonic,
)

# ground truth components
t_base = data_base["Propagation time (micsec)"].to_numpy()
t_harm = data_harmonic["Propagation time (micsec)"].to_numpy()

gt_base = np.zeros(len(t))
for mode in modes_base:
    gt_base += np.interp(t, t_base, data_base[mode].to_numpy())

gt_harmonic = np.zeros(len(t))
for mode in modes_harmonic:
    gt_harmonic += second_scale * np.interp(t, t_harm, data_harmonic[mode].to_numpy())

print(f"  signal max    : {np.abs(signal).max():.6f} nm")
print(f"  gt_base max   : {np.abs(gt_base).max():.6f} nm")
print(f"  gt_harmonic max: {np.abs(gt_harmonic).max():.6f} nm")


# ═══════════════════════════════════════════════════════════════════
#  CWT
# ═══════════════════════════════════════════════════════════════════

print("\n[1] Computing CWT …")

t_s  = t * 1e-6                          # µs → s
dt   = float(np.mean(np.diff(t_s)))
fs   = 1.0 / dt

fc     = pywt.central_frequency(wavelet)
freqs_target = np.linspace(fmin, fmax, n_freqs)
scales = fc / (freqs_target * dt)

cwtmatr, freqs_out = pywt.cwt(signal, scales, wavelet, sampling_period=dt)

print(f"  fc            : {fc}")
print(f"  dt            : {dt:.4e} s    fs: {fs/1e6:.3f} MHz")
print(f"  scales        : {scales.min():.3f} → {scales.max():.3f}")
print(f"  freqs_out     : {freqs_out.min()/1e6:.3f} → {freqs_out.max()/1e6:.3f} MHz")
print(f"  cwtmatr max   : {np.abs(cwtmatr).max():.6f}")

# band mask
mask_base = (freqs_out >= band_min_base) & (freqs_out <= band_max_base)
mask_harm = (freqs_out >= band_min_harmonic) & (freqs_out <= band_max_harmonic)

cwt_base = cwtmatr[mask_base, :]
cwt_harm = cwtmatr[mask_harm, :]
scales_base = scales[mask_base]
scales_harm = scales[mask_harm]

print(f"\n  base band     : {freqs_out[mask_base].min()/1e6:.3f}–{freqs_out[mask_base].max()/1e6:.3f} MHz  "
      f"({mask_base.sum()} bins)")
print(f"  harmonic band : {freqs_out[mask_harm].min()/1e6:.3f}–{freqs_out[mask_harm].max()/1e6:.3f} MHz  "
      f"({mask_harm.sum()} bins)")
print(f"  cwt_base max  : {np.abs(cwt_base).max():.6f}")
print(f"  cwt_harm max  : {np.abs(cwt_harm).max():.6f}")


# ═══════════════════════════════════════════════════════════════════
#  RECONSTRUCTION — try 4 different formulas
# ═══════════════════════════════════════════════════════════════════

def recon_v1(cwt_b, sc):
    """Original wavelet_processing formula: 1/a² × d_log_s"""
    r = np.real(np.sum(cwt_b / sc[:, None]**2, axis=0))
    return r * np.mean(np.diff(np.log(sc)))

def recon_v2(cwt_b, sc):
    """1/a × d_log_s  (analytic wavelet fix)"""
    r = np.real(np.sum(cwt_b / sc[:, None], axis=0))
    return r * np.mean(np.diff(np.log(sc)))

def recon_v3(cwt_b, sc):
    """1/sqrt(a) × d_log_s  (energy-normalised CWT convention)"""
    r = np.real(np.sum(cwt_b / np.sqrt(sc[:, None]), axis=0))
    return r * np.mean(np.diff(np.log(sc)))

def recon_v4(cwt_b, sc):
    """1/a × d_a/a  (Torrence & Compo 1998 — most cited geophysics formula)"""
    da = np.abs(np.diff(sc))
    da = np.append(da, da[-1])           # pad to same length
    r  = np.real(np.sum(cwt_b * da[:, None] / sc[:, None], axis=0))
    return r

print("\n[2] Testing reconstruction formulas …\n")
print(f"  {'Formula':<10}  {'base max':>12}  {'harm max':>12}  "
      f"{'base SNR':>10}  {'harm SNR':>10}")
print("  " + "─"*58)

results = {}
for name, fn in [("1/a²×dls", recon_v1),
                 ("1/a×dls",  recon_v2),
                 ("1/√a×dls", recon_v3),
                 ("T&C 1998", recon_v4)]:

    rb = fn(cwt_base, scales_base)
    rh = fn(cwt_harm, scales_harm)

    def snr(ref, rec):
        n = ref - rec
        return 10 * np.log10(np.var(ref) / (np.var(n) + 1e-30))

    print(f"  {name:<10}  {np.abs(rb).max():>12.6f}  {np.abs(rh).max():>12.6f}  "
          f"  {snr(gt_base, rb):>+8.1f}dB  {snr(gt_harmonic, rh):>+8.1f}dB")

    results[name] = (rb, rh)


# ═══════════════════════════════════════════════════════════════════
#  PICK BEST AND PLOT
# ═══════════════════════════════════════════════════════════════════

# Use the formula with best base SNR
def snr(ref, rec):
    n = ref - rec
    return 10 * np.log10(np.var(ref) / (np.var(n) + 1e-30))

best_name = max(results, key=lambda k: snr(gt_base, results[k][0]))
print(f"\n  Best formula: {best_name}")

rb_best, rh_best = results[best_name]

fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True)

axes[0].plot(t, signal, color="steelblue", lw=0.8)
axes[0].set_title("Composite signal (f + 2f)", fontsize=11)
axes[0].set_ylabel("nm"); axes[0].grid(True, alpha=0.25)

axes[1].plot(t, gt_base, color="seagreen", lw=0.9)
axes[1].set_title("GT fundamental", fontsize=11)
axes[1].set_ylabel("nm"); axes[1].grid(True, alpha=0.25)

axes[2].plot(t, gt_base,  color="seagreen",  lw=0.9, alpha=0.5, label="GT")
axes[2].plot(t, rb_best,  color="tomato",    lw=1.0, label=f"CWT recon ({best_name})")
axes[2].set_title(f"Base reconstruction vs GT  [SNR={snr(gt_base, rb_best):+.1f} dB]", fontsize=11)
axes[2].set_ylabel("nm"); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.25)

axes[3].plot(t, gt_harmonic, color="mediumpurple", lw=0.9)
axes[3].set_title("GT 2nd harmonic", fontsize=11)
axes[3].set_ylabel("nm"); axes[3].grid(True, alpha=0.25)

axes[4].plot(t, gt_harmonic, color="mediumpurple", lw=0.9, alpha=0.5, label="GT")
axes[4].plot(t, rh_best,     color="darkorange",   lw=1.0, label=f"CWT recon ({best_name})")
axes[4].set_title(f"Harmonic reconstruction vs GT  [SNR={snr(gt_harmonic, rh_best):+.1f} dB]", fontsize=11)
axes[4].set_ylabel("nm"); axes[4].set_xlabel("Time [µs]")
axes[4].legend(fontsize=8); axes[4].grid(True, alpha=0.25)

plt.suptitle(f"CWT Band Reconstruction — {wavelet}  (best: {best_name})", fontsize=13)
plt.tight_layout()
path = "plots/cwt_reconstruction_test.png"
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {path}")
print("\nDone.")