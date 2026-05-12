"""
═══════════════════════════════════════════════════════════════════════════════
  LAMB WAVE BLIND MODE DECOMPOSITION
  ───────────────────────────────────
  INPUT  : Sum Propagated signal only — as in real experiments
  OUTPUT : Which modes are present, their reconstructed waveforms, amplitudes

  METHOD — 3-stage pipeline
  ─────────────────────────
  STAGE 1 — SPECTROGRAM (SHORT-WINDOW STFT)
    Lamb wave modes share a similar frequency content but travel at different
    GROUP VELOCITIES, arriving at different times. A short-window STFT gives
    time resolution to reveal each modal packet as a localised burst of energy.
    The spectrogram amplitude averaged over the active frequency band shows
    distinct peaks — one per mode (or mode group if two share the same group
    velocity).

  STAGE 2 — MODAL PACKET DETECTION & SOFT-GATE EXTRACTION
    Peaks in the band-averaged envelope = modal arrivals. Around each peak
    we apply a Gaussian soft gate to the original signal to extract that
    mode's waveform m_n(t). Soft gating minimises spectral leakage between
    overlapping adjacent packets compared to a hard rectangular window.

        m_n(t) = s(t) · exp(−(t − t_peak_n)² / 2σ²)

  STAGE 3 — TIKHONOV LEAST SQUARES
    Stack gated waveforms as columns of modal matrix M and solve:

        s(t) = M · a   →   a = (MᵀM + λI)⁻¹ Mᵀ s

    a_n = scalar amplitude of mode n.  Reconstructed contribution = a_n · m_n(t)

  VERIFICATION (simulation only — needs GT columns in Excel):
    Cross-correlate each recovered waveform with GT mode signals.
    Report amplitude recovery error per mode.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
from scipy.signal import stft, hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS  — tune these for your signal
# ─────────────────────────────────────────────────────────────────────────────

class Params:
    f_min            = 1.0    # MHz — lower edge of active frequency band
    f_max            = 5.0    # MHz — upper edge
    win_len          = 128    # STFT window samples (~2.4 μs at fs=53 MHz)
    hop_len          = 2      # STFT hop samples
    n_fft            = 512    # FFT size (zero-pad)
    peak_threshold   = 0.08   # fraction of max envelope to call a peak
    peak_distance_us = 4.0    # minimum μs between detected peaks
    gate_width_us    = 6.0    # Gaussian gate σ in μs
    regularization   = 1e-4   # Tikhonov λ

P = Params()


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 0 — LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load(filepath):
    df  = pd.read_excel(filepath)
    t   = df["Propagation time (micsec)"].to_numpy()
    s   = df["Sum Propagated signal (nm)"].to_numpy()
    dt  = float(np.mean(np.diff(t)))
    fs  = 1.0 / dt

    gt_cols = [c for c in df.columns if "Propagated" in c and "Sum" not in c]
    gt      = {c.split(" Propagated")[0]: df[c].to_numpy() for c in gt_cols}

    print(f"  Samples : {len(t)}   dt={dt:.4f} μs   fs={fs:.2f} MHz")
    print(f"  Duration: 0 – {t[-1]:.1f} μs")
    print(f"  GT modes available: {list(gt.keys())}")
    return t, s, fs, dt, gt


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — SPECTROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectrogram(s, fs, win_len, hop_len, n_fft, f_min, f_max):
    window = np.hanning(win_len)
    f_hz, t_s, Vx = stft(s, fs=fs, window=window,
                          nperseg=win_len, noverlap=win_len - hop_len,
                          nfft=n_fft, return_onesided=True)
    Amp      = np.abs(Vx)
    f_mask   = (f_hz >= f_min) & (f_hz <= f_max)
    Amp_band = Amp[f_mask, :]
    envelope = Amp_band.mean(axis=0)

    print(f"  STFT : {Vx.shape}  t-res={win_len/fs:.2f}μs  "
          f"f-res={f_hz[1]-f_hz[0]:.3f}MHz")
    print(f"  Band : {f_hz[f_mask][0]:.2f}–{f_hz[f_mask][-1]:.2f} MHz")
    return f_hz, t_s, Vx, Amp, f_mask, envelope


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — DETECTION + EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_packets(envelope, t_s, peak_threshold, peak_distance_us):
    env_smooth      = gaussian_filter1d(envelope, sigma=5)
    fs_stft         = 1.0 / (t_s[1] - t_s[0])
    min_dist_frames = max(1, int(peak_distance_us * fs_stft))
    height_thresh   = peak_threshold * env_smooth.max()
    peaks, _        = find_peaks(env_smooth,
                                  height=height_thresh,
                                  distance=min_dist_frames)
    return peaks, env_smooth


def extract_modal_waveforms(s, t, peaks, t_s, gate_width_us):
    """
    Soft Gaussian gate extraction.
    Each modal waveform = s(t) * Gaussian centred on detected arrival time.
    Soft gating avoids Gibbs ringing that hard windows introduce.
    """
    modal_waveforms, gate_times = [], []
    for p in peaks:
        t_peak = t_s[p]
        gate   = np.exp(-0.5 * ((t - t_peak) / gate_width_us) ** 2)
        modal_waveforms.append(s * gate)
        gate_times.append(t_peak)
    return modal_waveforms, gate_times


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — LEAST SQUARES
# ─────────────────────────────────────────────────────────────────────────────

def solve_amplitudes(s, modal_waveforms, regularization):
    """
    Tikhonov least squares:  a = (MᵀM + λI)⁻¹ Mᵀ s
    Condition number > 1e6 → increase regularization.
    """
    M     = np.column_stack(modal_waveforms)
    cond  = np.linalg.cond(M)
    a     = np.linalg.solve(M.T @ M + regularization * np.eye(M.shape[1]),
                             M.T @ s)
    resid = np.linalg.norm(s - M @ a) / (np.linalg.norm(s) + 1e-12)
    return a, M, cond, resid


# ─────────────────────────────────────────────────────────────────────────────
#  VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def compare_ground_truth(t, modal_waveforms, amplitudes, gate_times, gt):
    print("\n── Ground Truth Comparison ──────────────────────────────────────────")
    print(f"  {'Mode':<5}  {'GT arrival':>11}  {'GT peak(nm)':>12}  "
          f"{'Best R':>7}  {'R arrival':>10}  {'Rec peak(nm)':>13}  {'Err%':>7}")
    print("  " + "─"*72)

    results = {}
    for mode_name, gt_sig in sorted(gt.items()):
        gt_peak = float(np.max(np.abs(gt_sig)))
        if gt_peak < 1e-4:
            continue
        gt_env_t = float(t[np.argmax(np.abs(hilbert(gt_sig)))])

        best_corr, best_i = -1, -1
        for i, mw in enumerate(modal_waveforms):
            n    = min(len(gt_sig), len(mw))
            corr = float(np.abs(np.corrcoef(gt_sig[:n], mw[:n])[0, 1]))
            if corr > best_corr:
                best_corr, best_i = corr, i

        rec_peak  = abs(amplitudes[best_i]) * np.max(np.abs(modal_waveforms[best_i]))
        err_pct   = abs(rec_peak - gt_peak) / (gt_peak + 1e-12) * 100
        ridge_t   = gate_times[best_i]

        results[mode_name] = dict(gt_peak=gt_peak, gt_t=gt_env_t,
                                   rec_peak=float(rec_peak), ridge_t=ridge_t,
                                   ridge_idx=best_i, err_pct=err_pct)
        print(f"  {mode_name:<5}  {gt_env_t:>11.2f}  {gt_peak:>12.5f}  "
              f"R{best_i:>5}  {ridge_t:>10.2f}  {rec_peak:>13.5f}  {err_pct:>6.1f}%")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(t, s, t_s, Amp, f_hz, f_mask, envelope, env_smooth,
             peaks, modal_waveforms, amplitudes, gate_times, gt_comp, P):

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(peaks), 1)))

    # ── Fig 1: Overview ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(t, s, color="steelblue", linewidth=0.8)
    axes[0].set_title("Measured signal — Sum Propagated (input only)")
    axes[0].set_xlabel("Time (μs)"); axes[0].set_ylabel("Amplitude (nm)")
    axes[0].grid(True, alpha=0.3)

    Amp_db = 20 * np.log10(Amp[f_mask, :] + 1e-12)
    Amp_db -= Amp_db.max()
    axes[1].pcolormesh(t_s, f_hz[f_mask], Amp_db,
                       vmin=-40, vmax=0, cmap="inferno", shading="gouraud")
    axes[1].set_title(f"Spectrogram  ({P.f_min}–{P.f_max} MHz)")
    axes[1].set_xlabel("Time (μs)"); axes[1].set_ylabel("Frequency (MHz)")
    for i, p in enumerate(peaks):
        axes[1].axvline(t_s[p], color=colors[i], lw=1.5, ls="--", alpha=0.9)

    axes[2].plot(t_s, envelope,   color="lightgrey", lw=0.8, label="Raw")
    axes[2].plot(t_s, env_smooth, color="steelblue",  lw=1.2, label="Smoothed")
    for i, p in enumerate(peaks):
        axes[2].axvline(t_s[p], color=colors[i], lw=1.5, ls="--",
                        label=f"R{i} @ {t_s[p]:.1f}μs")
    axes[2].set_title("Band-averaged envelope + detected modal arrivals")
    axes[2].set_xlabel("Time (μs)"); axes[2].set_ylabel("Mean amplitude")
    axes[2].legend(fontsize=7, ncol=5); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/01_overview.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/01_overview.png")
    plt.show()

    # ── Fig 2: Reconstructions ────────────────────────────────────────────────
    n  = len(modal_waveforms)
    nc = 3
    nr = int(np.ceil(n / nc)) + 1
    fig = plt.figure(figsize=(16, 3.2 * nr))

    M   = np.column_stack(modal_waveforms)
    rec = M @ amplitudes
    ax0 = fig.add_subplot(nr, 1, 1)
    ax0.plot(t, s,   color="steelblue", lw=1, alpha=0.7, label="Measured")
    ax0.plot(t, rec, color="tomato",    lw=1.3, ls="--",
             label=f"Reconstruction  (residual={np.linalg.norm(s-rec)/np.linalg.norm(s):.3f})")
    ax0.set_title("Measured vs Reconstructed"); ax0.legend()
    ax0.set_xlabel("Time (μs)"); ax0.set_ylabel("nm"); ax0.grid(True, alpha=0.3)

    for i, (mw, amp, gt_) in enumerate(zip(modal_waveforms, amplitudes, gate_times)):
        ax = fig.add_subplot(nr, nc, nc + 1 + i)
        ax.plot(t, amp * mw, color=colors[i], lw=1)
        ax.axvline(gt_, color="black", lw=0.8, ls=":")
        ax.set_title(f"R{i}  t={gt_:.1f}μs  |a|={abs(amp):.4f}", fontsize=9)
        ax.set_xlabel("Time (μs)", fontsize=7); ax.set_ylabel("nm", fontsize=7)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/02_modal_reconstructions.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/02_modal_reconstructions.png")
    plt.show()

    # ── Fig 3: Amplitude bars + GT comparison ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    labels = [f"R{i}\n{gate_times[i]:.1f}μs" for i in range(n)]
    vals   = [abs(a) * np.max(np.abs(mw))
              for a, mw in zip(amplitudes, modal_waveforms)]
    bars   = axes[0].bar(labels, vals, color=colors[:n], edgecolor="white")
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, v*1.02,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
    axes[0].set_title("Recovered modal peak amplitudes")
    axes[0].set_ylabel("|a_n|·max|m_n(t)|  (nm)")
    axes[0].set_xlabel("Ridge (arrival time)")
    axes[0].grid(True, alpha=0.3, axis="y")

    if gt_comp:
        gn  = list(gt_comp.keys())
        gv  = [gt_comp[m]["gt_peak"]  for m in gn]
        rv  = [gt_comp[m]["rec_peak"] for m in gn]
        x   = np.arange(len(gn)); w = 0.38
        axes[1].bar(x-w/2, gv, w, label="GT peak",     color="steelblue", alpha=0.85)
        axes[1].bar(x+w/2, rv, w, label="Recovered",   color="tomato",    alpha=0.85)
        axes[1].set_xticks(x); axes[1].set_xticklabels(gn, rotation=45, fontsize=8)
        axes[1].set_title("GT peak vs Recovered amplitude per mode")
        axes[1].set_ylabel("|Amplitude| (nm)"); axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("plots/03_amplitudes.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/03_amplitudes.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    FILEPATH = r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@350mm.xlsx"
    # FILEPATH = r"/mnt/user-data/uploads/In-plane_A2_TemporalResponse_15_963MHzmm_200mm.xlsx"

    print("\n[0] Loading...")
    t, s, fs, dt, gt = load(FILEPATH)

    print("\n[1] Spectrogram...")
    f_hz, t_s, Vx, Amp, f_mask, envelope = compute_spectrogram(
        s, fs, P.win_len, P.hop_len, P.n_fft, P.f_min, P.f_max)

    print("\n[2] Detecting modal packets...")
    peaks, env_smooth = detect_packets(
        envelope, t_s, P.peak_threshold, P.peak_distance_us)
    print(f"  {len(peaks)} packets detected:")
    for i, p in enumerate(peaks):
        print(f"    R{i}: {t_s[p]:.2f} μs")

    print("\n[2b] Extracting modal waveforms (Gaussian gate)...")
    modal_waveforms, gate_times = extract_modal_waveforms(
        s, t, peaks, t_s, P.gate_width_us)

    print("\n[3] Least squares solve...")
    amplitudes, M, cond, resid = solve_amplitudes(
        s, modal_waveforms, P.regularization)

    print(f"  Condition number : {cond:.3e}")
    print(f"  Relative residual: {resid:.4f}")
    if cond > 1e6:
        print("  ⚠  Increase P.regularization")

    print("\n── Mode Summary ─────────────────────────────────────────────────────")
    print(f"  {'R':<4}  {'Arrival(μs)':>12}  {'|a_n|':>10}  {'Peak amp(nm)':>13}")
    print("  " + "─"*44)
    for i, (amp, gt_) in enumerate(zip(amplitudes, gate_times)):
        pk = abs(amp) * np.max(np.abs(modal_waveforms[i]))
        print(f"  R{i:<3}  {gt_:>12.2f}  {abs(amp):>10.5f}  {pk:>13.5f}")

    print("\n[Verification]...")
    gt_comp = compare_ground_truth(
        t, modal_waveforms, amplitudes, gate_times, gt)

    print("\n[Plotting]...")
    plot_all(t, s, t_s, Amp, f_hz, f_mask, envelope, env_smooth,
             peaks, modal_waveforms, amplitudes, gate_times, gt_comp, P)

    print("\nDone.")