"""
═══════════════════════════════════════════════════════════════════════════════
  test_sst.py  — standalone SST evaluation script
  ─────────────────────────────────────────────────
  Run from the project root (same folder as preprocess.py / decomp.py).

  What this script produces
  ─────────────────────────
  plots/sst_stft_comparison.png   — STFT original vs SST side-by-side
  plots/sst_cwt_comparison.png    — CWT  original vs SST side-by-side
  plots/sst_stft_reconstruction.png — base + harmonic band recon (STFT)
  plots/sst_cwt_reconstruction.png  — base + harmonic band recon (CWT)

  All parameters are grouped at the top — edit freely.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import sys
import os

# ── make sure the project root is on the path ──────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocess
from transforms import SST_v2_processing          


# ═══════════════════════════════════════════════════════════════════════════
#  USER PARAMETERS  — edit here
# ═══════════════════════════════════════════════════════════════════════════

# --- signal construction (mirrors despair.py) -------------------------------
noise_level = 0.0          # 0 = clean, 1.5 = 150% noise
beta        = 9            # non-linearity parameter

A1_mode = "S2 Propagated signal (nm)"   # base harmonic mode for β
A2_mode = "S4 Propagated signal (nm)"   # second harmonic mode for β

dataset_base      = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harmonic   = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"

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

# --- time-frequency analysis -----------------------------------------------
f_min_analyse = 1.0e6      # Hz — lower bound for TF display
f_max_analyse = 4.5e6      # Hz — upper bound for TF display
n_freq        = 400        # frequency bins (CWT)

band_min_base      = 1_100_000   # Hz
band_max_base      = 1_500_000
band_min_harmonic  = 2_300_000
band_max_harmonic  = 2_900_000

# --- CWT wavelet -----------------------------------------------------------
wavelet = "cmor3.0-1.0"

# --- STFT parameters (from blind-decomp stage 1) ---------------------------
stft_win_len = 128     # samples
stft_hop_len = 2
stft_n_fft   = 512

# --- SST thresholds --------------------------------------------------------
stft_gamma = 1e-6      # STFT bins weaker than this are not reassigned
cwt_gamma  = 1e-8      # CWT  coefficients weaker than this are not reassigned

# --- plot flags ------------------------------------------------------------
log_scale = True       # dB colour scale in TF plots


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & BUILD SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n[0] Loading data …")
data_base      = preprocess.get_data(dataset_base)
data_harmonic  = preprocess.get_data(dataset_harmonic)

print("[1] Building composite signal …")
t, signal, second_scale = preprocess.create_signal(
    data_base, data_harmonic,
    beta, noise_level,
    A1_mode, A2_mode,
    modes_base, modes_harmonic,
)
print(f"    samples={len(t)}   t=[{t[0]:.2f}, {t[-1]:.2f}] µs   "
      f"second_scale={second_scale:.4f}")

# ── Ground truth components (Option A — no changes to preprocess.py) ────────
# Reconstruct exactly what create_signal summed, but keep the two parts separate.
# This mirrors the internal logic of preprocess.create_signal.

t_base = data_base["Propagation time (micsec)"].to_numpy()
t_harm = data_harmonic["Propagation time (micsec)"].to_numpy()

gt_base = np.zeros(len(t))
for mode in modes_base:
    gt_base += np.interp(t, t_base, data_base[mode].to_numpy())

gt_harmonic = np.zeros(len(t))
for mode in modes_harmonic:
    gt_harmonic += second_scale * np.interp(t, t_harm, data_harmonic[mode].to_numpy())

print(f"    GT base peak    : {np.max(np.abs(gt_base)):.4f} nm")
print(f"    GT harmonic peak: {np.max(np.abs(gt_harmonic)):.4f} nm")


# ═══════════════════════════════════════════════════════════════════════════
#  STFT-SST
# ═══════════════════════════════════════════════════════════════════════════

print("\n[2] STFT-SST …")
S_orig, S_sst, f_stft, t_stft = SST_v2_processing.stft_sst(
    t, signal,
    fmin=f_min_analyse,
    fmax=f_max_analyse,
    win_len=stft_win_len,
    hop_len=stft_hop_len,
    n_fft=stft_n_fft,
    gamma=stft_gamma,
)
print(f"    STFT shape: {S_orig.shape}   SST shape: {S_sst.shape}")

# side-by-side spectrogram comparison
SST_v2_processing.plot_comparison(
    t_stft, f_stft,
    S_orig, S_sst,
    method="STFT",
    name="sst_stft_comparison",
    log_scale=log_scale,
    fmin=f_min_analyse,
    fmax=f_max_analyse,
)

# band reconstructions
print("    Reconstructing base band (STFT) …")
recon_base_stft = SST_v2_processing.reconstruct_band_stft(
    t, signal,
    band_min=band_min_base,
    band_max=band_max_base,
    fmin=f_min_analyse,
    fmax=f_max_analyse,
    win_len=stft_win_len,
    hop_len=stft_hop_len,
    n_fft=stft_n_fft,
)

print("    Reconstructing harmonic band (STFT) …")
recon_harmonic_stft = SST_v2_processing.reconstruct_band_stft(
    t, signal,
    band_min=band_min_harmonic,
    band_max=band_max_harmonic,
    fmin=f_min_analyse,
    fmax=f_max_analyse,
    win_len=stft_win_len,
    hop_len=stft_hop_len,
    n_fft=stft_n_fft,
)

SST_v2_processing.plot_reconstruction(
    t, signal,
    recon_base_stft,
    recon_harmonic_stft,
    gt_base=gt_base,
    gt_harmonic=gt_harmonic,
    method="STFT",
    name="sst_stft_reconstruction",
)


# ═══════════════════════════════════════════════════════════════════════════
#  CWT-SST
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3] CWT-SST …  (this takes a moment)")
C_orig, C_sst, f_cwt, t_cwt = SST_v2_processing.cwt_sst(
    t, signal,
    wavelet=wavelet,
    fmin=f_min_analyse,
    fmax=f_max_analyse,
    n_freqs=n_freq,
    gamma=cwt_gamma,
)
print(f"    CWT shape: {C_orig.shape}   SST shape: {C_sst.shape}")

# side-by-side scalogram comparison
SST_v2_processing.plot_comparison(
    t_cwt, f_cwt,
    C_orig, C_sst,
    method="CWT",
    name="sst_cwt_comparison",
    log_scale=log_scale,
    fmin=f_min_analyse,
    fmax=f_max_analyse,
)

# band reconstructions
print("    Reconstructing base band (CWT) …")
recon_base_cwt = SST_v2_processing.reconstruct_band_cwt(
    t, signal,
    band_min=band_min_base,
    band_max=band_max_base,
    wavelet=wavelet,
    fmin=f_min_analyse,
    fmax=f_max_analyse,
    n_freqs=n_freq,
)

print("    Reconstructing harmonic band (CWT) …")
recon_harmonic_cwt = SST_v2_processing.reconstruct_band_cwt(
    t, signal,
    band_min=band_min_harmonic,
    band_max=band_max_harmonic,
    wavelet=wavelet,
    fmin=f_min_analyse,
    fmax=f_max_analyse,
    n_freqs=n_freq,
)

SST_v2_processing.plot_reconstruction(
    t, signal,
    recon_base_cwt,
    recon_harmonic_cwt,
    gt_base=gt_base,
    gt_harmonic=gt_harmonic,
    method="CWT",
    name="sst_cwt_reconstruction",
)

# --- DEBUG ---
import pywt as _pywt
_t = t * 1e-6
_dt = float(np.mean(np.diff(_t)))
_fc = _pywt.central_frequency(wavelet)
_freqs = np.linspace(f_min_analyse, f_max_analyse, n_freq)
_scales = _fc / (_freqs * _dt)
_cwtmatr, _freqs_out = _pywt.cwt(signal, _scales, wavelet, sampling_period=_dt)

_mask = (_freqs_out >= band_min_base) & (_freqs_out <= band_max_base)
print("scales range:", _scales.min(), _scales.max())
print("band scales:", _scales[_mask].min(), _scales[_mask].max(), "count:", _mask.sum())
print("d_log_s_full:", np.mean(np.diff(np.log(_scales))))
print("d_log_s_band:", np.mean(np.diff(np.log(_scales[_mask]))))
print("CWT max:", np.abs(_cwtmatr).max())
print("CWT band max:", np.abs(_cwtmatr[_mask]).max())
print("recon_base_cwt max:", np.abs(recon_base_cwt).max())
print("gt_base max:", np.abs(gt_base).max())
# --- END DEBUG ---
# ═══════════════════════════════════════════════════════════════════════════
#  QUICK SANITY CHECK  — print peak SNR of reconstructions
# ═══════════════════════════════════════════════════════════════════════════

def _peak_snr(reference, recon):
    noise = reference - recon
    with np.errstate(divide="ignore"):
        snr = 10 * np.log10(np.var(reference) / (np.var(noise) + 1e-30))
    return snr

print("\n── Reconstruction quality (compared against GT components) ──────────")
print(f"  STFT base band SNR     : {_peak_snr(gt_base,     recon_base_stft):+.1f} dB")
print(f"  STFT harmonic SNR      : {_peak_snr(gt_harmonic, recon_harmonic_stft):+.1f} dB")
print(f"  CWT  base band SNR     : {_peak_snr(gt_base,     recon_base_cwt):+.1f} dB")
print(f"  CWT  harmonic SNR      : {_peak_snr(gt_harmonic, recon_harmonic_cwt):+.1f} dB")

print("\nDone — all plots saved to plots/")