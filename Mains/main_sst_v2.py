"""
sst_main.py
===========
Visual verification script for sst_processing.py.

Produces three figure groups:
  1.  Spectrograms  – raw CWT, CWT-SST, raw STFT, STFT-SST (2 × 2 grid)
  2.  Reconstructions – base band and harmonic band for every method
      (CWT-plain, CWT-SST, STFT-SST) vs. the individual ground-truth modes
  3.  Overlay comparison – all reconstructions on one plot per band

Run:
    python sst_main.py

Output folder:  ./sst_results/
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt
from scipy.signal import stft

from transforms import sst_processing_v2 as sst

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — tweak these to match despair.py settings
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE   = "In-plane_TemporalResponse_7_9866MHzmm_350mm.xlsx"
OUT_FOLDER  = "./sst_results"

WAVELET     = "cmor3.0-1.0"
F_MIN       = 1.0e6    # Hz — analysis window
F_MAX       = 4.5e6    # Hz
N_FREQS     = 300      # frequency bins (lower = faster for preview)
NPERSEG     = 256      # STFT window length

# Base harmonic band
BAND_MIN_BASE     = 1.1e6   # Hz
BAND_MAX_BASE     = 1.5e6   # Hz

# Second harmonic band
BAND_MIN_HARMONIC = 2.3e6
BAND_MAX_HARMONIC = 2.9e6

GAMMA = 1e-6           # SST noise threshold

# Time window to focus plots (µs) – set to None for full signal
T_FOCUS = (70, 350)   # µs

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_FOLDER, exist_ok=True)

def savefig(name: str, dpi: int = 200):
    path = os.path.join(OUT_FOLDER, name + ".png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data …")
df  = pd.read_excel(DATA_FILE)
t   = df["Propagation time (micsec)"].values
sig = df["Sum Propagated signal (nm)"].values

# Individual ground-truth modes available in this file (base harmonics only)
mode_cols = [c for c in df.columns if "Propagated signal" in c and "Sum" not in c]
modes = {c.split(" ")[0]: df[c].values for c in mode_cols}

dt   = float(np.mean(np.diff(t)))
fs   = 1.0 / (dt * 1e-6)   # Hz

print(f"  n={len(sig)}  dt={dt:.4f} µs  fs={fs/1e6:.3f} MHz")
print(f"  Modes available: {list(modes.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing transforms …")

# ── 2a. Raw CWT (no SST) ─────────────────────────────────────────────────────
print("  CWT …")
t_s        = t * 1e-6
fc         = pywt.central_frequency(WAVELET)
freqs_cwt  = np.linspace(F_MIN, F_MAX, N_FREQS)
scales_cwt = fc / (freqs_cwt * dt * 1e-6)
cwtmatr, _ = pywt.cwt(sig, scales_cwt, WAVELET, sampling_period=dt * 1e-6)
cwt_amp    = np.abs(cwtmatr)

# ── 2b. CWT-SST ──────────────────────────────────────────────────────────────
print("  CWT-SST …")
_, freqs_sst, Ts_cwt = sst.cwt_sst(
    t, sig, wavelet=WAVELET, fmin=F_MIN, fmax=F_MAX,
    n_freqs=N_FREQS, gamma=GAMMA, plot=False
)

# ── 2c. Raw STFT ─────────────────────────────────────────────────────────────
print("  STFT …")
noverlap       = NPERSEG - 1
f_stft, t_stft, Zxx = stft(sig, fs=fs, nperseg=NPERSEG, noverlap=noverlap, window="hann")
stft_amp       = np.abs(Zxx)
f_mask         = (f_stft >= F_MIN) & (f_stft <= F_MAX)
t_stft_us      = t_stft * 1e6

# ── 2d. STFT-SST ─────────────────────────────────────────────────────────────
print("  STFT-SST …")
t_stft_us2, freqs_stft_sst, Ts_stft = sst.stft_sst(
    t, sig, fmin=F_MIN, fmax=F_MAX, nperseg=NPERSEG, gamma=GAMMA, plot=False
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURE 1 — 2×2 SPECTROGRAM COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print("\nPlotting spectrograms …")

def _tlim(t_arr):
    if T_FOCUS is None:
        return slice(None)
    return (t_arr >= T_FOCUS[0]) & (t_arr <= T_FOCUS[1])

fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
fig.suptitle("Spectrogram comparison — CWT vs STFT, with and without SST", fontsize=13)

# Panel labels
labels = [
    ("CWT — raw",   t,           freqs_cwt / 1e6,           cwt_amp),
    ("CWT-SST",     t,           freqs_sst / 1e6,           Ts_cwt),
    ("STFT — raw",  t_stft_us,   f_stft[f_mask] / 1e6,      stft_amp[f_mask, :]),
    ("STFT-SST",    t_stft_us2,  freqs_stft_sst / 1e6,      Ts_stft),
]

for ax, (title, t_ax, f_ax, Z) in zip(axes.flat, labels):
    tslice = _tlim(t_ax)
    t_plot = t_ax[tslice]
    Z_plot = Z[:, tslice]

    pcm = ax.pcolormesh(t_plot, f_ax, Z_plot, shading="gouraud",
                        cmap="inferno")
    fig.colorbar(pcm, ax=ax, label="Amplitude")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Frequency [MHz]")
    ax.set_ylim(F_MIN / 1e6, F_MAX / 1e6)

    # Mark band boundaries
    for bmin, bmax, col in [
        (BAND_MIN_BASE,     BAND_MAX_BASE,     "cyan"),
        (BAND_MIN_HARMONIC, BAND_MAX_HARMONIC, "lime"),
    ]:
        ax.axhline(bmin / 1e6, color=col, lw=0.8, ls="--", alpha=0.7)
        ax.axhline(bmax / 1e6, color=col, lw=0.8, ls="--", alpha=0.7)

# Legend for band markers
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], color="cyan", ls="--", label=f"Base band  {BAND_MIN_BASE/1e6:.1f}–{BAND_MAX_BASE/1e6:.1f} MHz"),
    Line2D([0], [0], color="lime", ls="--", label=f"2nd harmonic {BAND_MIN_HARMONIC/1e6:.1f}–{BAND_MAX_HARMONIC/1e6:.1f} MHz"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

savefig("fig1_spectrogram_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPUTE RECONSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing reconstructions …")

# Plain CWT reconstruction (wavelet_processing-style, no SST)
def cwt_plain_reconstruct(t_us, signal, band_min, band_max):
    t_s    = t_us * 1e-6
    dt_loc = float(np.mean(np.diff(t_s)))
    fc_loc = pywt.central_frequency(WAVELET)
    freqs  = np.linspace(F_MIN, F_MAX, N_FREQS)
    scales = fc_loc / (freqs * dt_loc)
    C, _   = pywt.cwt(signal, scales, WAVELET, sampling_period=dt_loc)
    mask   = (freqs >= band_min) & (freqs <= band_max)
    C_band = C[mask, :]
    s_band = scales[mask]
    rec    = np.real(np.sum(C_band / s_band[:, None] ** 2, axis=0))
    rec   *= np.mean(np.diff(np.log(s_band)))
    return rec

print("  CWT plain base …")
recon_base_cwt_plain = cwt_plain_reconstruct(t, sig, BAND_MIN_BASE, BAND_MAX_BASE)

print("  CWT-SST base …")
recon_base_cwt_sst = sst.reconstruct_band_cwt_sst(
    t, sig, BAND_MIN_BASE, BAND_MAX_BASE,
    wavelet=WAVELET, fmin=F_MIN, fmax=F_MAX, n_freqs=N_FREQS, gamma=GAMMA
)

print("  STFT-SST base …")
recon_base_stft_sst = sst.reconstruct_band_stft_sst(
    t, sig, BAND_MIN_BASE, BAND_MAX_BASE,
    fmin=F_MIN, fmax=F_MAX, nperseg=NPERSEG, gamma=GAMMA
)

print("  CWT plain harmonic …")
recon_harm_cwt_plain = cwt_plain_reconstruct(t, sig, BAND_MIN_HARMONIC, BAND_MAX_HARMONIC)

print("  CWT-SST harmonic …")
recon_harm_cwt_sst = sst.reconstruct_band_cwt_sst(
    t, sig, BAND_MIN_HARMONIC, BAND_MAX_HARMONIC,
    wavelet=WAVELET, fmin=F_MIN, fmax=F_MAX, n_freqs=N_FREQS, gamma=GAMMA
)

print("  STFT-SST harmonic …")
recon_harm_stft_sst = sst.reconstruct_band_stft_sst(
    t, sig, BAND_MIN_HARMONIC, BAND_MAX_HARMONIC,
    fmin=F_MIN, fmax=F_MAX, nperseg=NPERSEG, gamma=GAMMA
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 2 — RECONSTRUCTION DETAIL (3 rows × 2 cols)
#    Left: base band   Right: harmonic band
#    Row 1: CWT plain  Row 2: CWT-SST  Row 3: STFT-SST
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting reconstructions …")

tslice = _tlim(t)

# Ground-truth sums per band (sum of individual modes that fall in each band)
# Base band modes (S1, S2 rough) — use whatever is in file
def band_gt(modes_dict, t_arr):
    """Sum of all individual mode signals (serves as a rough ground truth)."""
    return sum(v for v in modes_dict.values())

gt_all = band_gt(modes, t)

fig2, axes2 = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
fig2.suptitle("Band reconstructions — base (left) vs harmonic (right)", fontsize=13)

recon_pairs = [
    ("CWT — plain (no SST)", recon_base_cwt_plain,  recon_harm_cwt_plain),
    ("CWT-SST",               recon_base_cwt_sst,    recon_harm_cwt_sst),
    ("STFT-SST",              recon_base_stft_sst,   recon_harm_stft_sst),
]

for row_idx, (method, r_base, r_harm) in enumerate(recon_pairs):
    for col_idx, (recon, band_label, bmin, bmax) in enumerate([
        (r_base, f"Base band {BAND_MIN_BASE/1e6:.1f}–{BAND_MAX_BASE/1e6:.1f} MHz",
         BAND_MIN_BASE, BAND_MAX_BASE),
        (r_harm, f"Harmonic band {BAND_MIN_HARMONIC/1e6:.1f}–{BAND_MAX_HARMONIC/1e6:.1f} MHz",
         BAND_MIN_HARMONIC, BAND_MAX_HARMONIC),
    ]):
        ax = axes2[row_idx, col_idx]

        t_p  = t[tslice]
        r_p  = recon[tslice]

        # Normalise for visual comparison
        norm = np.abs(r_p).max() or 1.0

        ax.plot(t_p, r_p / norm, color="steelblue", lw=0.9, label=method)
        ax.plot(t_p, sig[tslice] / (np.abs(sig[tslice]).max() or 1.0),
                color="gray", lw=0.5, alpha=0.4, label="Full signal (norm)")

        ax.set_title(f"{method}  |  {band_label}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Time [µs]")
        ax.set_ylabel("Norm. amplitude")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(T_FOCUS or (t[0], t[-1]))

savefig("fig2_reconstructions")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE 3 — OVERLAY: all methods on one plot per band
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting overlay comparison …")

fig3, (ax_b, ax_h) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
fig3.suptitle("All reconstructions overlaid", fontsize=13)

t_p = t[tslice]

for ax, recons, title in [
    (ax_b,
     [("CWT plain",  recon_base_cwt_plain,  "tab:blue"),
      ("CWT-SST",    recon_base_cwt_sst,    "tab:orange"),
      ("STFT-SST",   recon_base_stft_sst,   "tab:green")],
     f"Base band  {BAND_MIN_BASE/1e6:.1f}–{BAND_MAX_BASE/1e6:.1f} MHz"),
    (ax_h,
     [("CWT plain",  recon_harm_cwt_plain,  "tab:blue"),
      ("CWT-SST",    recon_harm_cwt_sst,    "tab:orange"),
      ("STFT-SST",   recon_harm_stft_sst,   "tab:green")],
     f"Harmonic band  {BAND_MIN_HARMONIC/1e6:.1f}–{BAND_MAX_HARMONIC/1e6:.1f} MHz"),
]:
    ax.plot(t_p, sig[tslice] / (np.abs(sig[tslice]).max() or 1.0),
            color="lightgray", lw=0.7, zorder=0, label="Full signal (norm)")

    for label, recon, col in recons:
        r_p  = recon[tslice]
        norm = np.abs(r_p).max() or 1.0
        ax.plot(t_p, r_p / norm, color=col, lw=1.1, label=label)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Norm. amplitude")
    ax.legend(fontsize=8)
    ax.set_xlim(T_FOCUS or (t[0], t[-1]))

savefig("fig3_overlay_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE 4 — SINGLE-MODE GROUND TRUTH vs CWT-SST reconstruction
#    Compares what each SST method recovers against individual known modes
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting per-mode ground truth comparison …")

# Pick the two most energetic modes from the file as reference
sorted_modes = sorted(modes.items(), key=lambda kv: np.abs(kv[1]).max(), reverse=True)
top_modes = sorted_modes[:4]

fig4, axes4 = plt.subplots(len(top_modes), 1, figsize=(14, 3.5 * len(top_modes)),
                            constrained_layout=True)
fig4.suptitle("Individual mode ground truths vs CWT-SST band reconstruction", fontsize=12)

for ax, (mode_name, mode_sig) in zip(np.atleast_1d(axes4), top_modes):
    t_p = t[tslice]
    gt  = mode_sig[tslice]
    gt_n = gt / (np.abs(gt).max() or 1.0)

    # Which band does this mode belong to? Check its peak frequency via FFT
    n = len(mode_sig)
    f_fft = np.fft.rfftfreq(n, d=dt * 1e-6)
    S_fft = np.abs(np.fft.rfft(mode_sig))
    peak_f = f_fft[np.argmax(S_fft)]

    if BAND_MIN_BASE <= peak_f <= BAND_MAX_BASE:
        recon = recon_base_cwt_sst
        band_label = f"CWT-SST base {BAND_MIN_BASE/1e6:.1f}–{BAND_MAX_BASE/1e6:.1f} MHz"
    elif BAND_MIN_HARMONIC <= peak_f <= BAND_MAX_HARMONIC:
        recon = recon_harm_cwt_sst
        band_label = f"CWT-SST harmonic {BAND_MIN_HARMONIC/1e6:.1f}–{BAND_MAX_HARMONIC/1e6:.1f} MHz"
    else:
        recon = recon_base_cwt_sst   # fallback
        band_label = "CWT-SST base (fallback)"

    r_p  = recon[tslice]
    r_n  = r_p / (np.abs(r_p).max() or 1.0)

    ax.plot(t_p, gt_n,  lw=1.1, color="tab:blue",   label=f"GT: {mode_name} (peak {peak_f/1e6:.2f} MHz)")
    ax.plot(t_p, r_n,   lw=1.0, color="tab:orange", ls="--", label=band_label, alpha=0.85)
    ax.set_ylabel("Norm. amplitude")
    ax.set_xlabel("Time [µs]")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(T_FOCUS or (t[0], t[-1]))

savefig("fig4_mode_groundtruth_vs_sst")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nDone.  All figures saved to: {os.path.abspath(OUT_FOLDER)}/")
print("  fig1_spectrogram_comparison.png  — 2×2 CWT/STFT ± SST spectrograms")
print("  fig2_reconstructions.png         — 3×2 reconstruction grid per method")
print("  fig3_overlay_comparison.png      — all methods overlaid per band")
print("  fig4_mode_groundtruth_vs_sst.png — individual modes vs SST output")