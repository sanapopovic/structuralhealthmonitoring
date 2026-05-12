"""
═══════════════════════════════════════════════════════════════════════════════
  decomposition_twostage.py
  ─────────────────────────────────────────────────────────────────────────────

  TWO-STAGE LAMB WAVE AMPLITUDE EXTRACTION
  ─────────────────────────────────────────

  STAGE 1 — LASSO mode identification
    Run LASSO on all possible modes to identify which ones are active.
    You do not need to know this ahead of time — LASSO figures it out
    from the data automatically.

  STAGE 2 — Weighted least squares (WLS) amplitude extraction
    Solve restricted to (LASSO active set) ∪ (forced modes of interest).
    Weighting focuses accuracy on the time windows where your modes
    of interest are predicted to arrive, rather than minimising global
    reconstruction error equally across the whole signal.

  WHY NOT BAYESIAN FOR STAGE 2
  ──────────────────────────────
  Your research question is about repeatability across measurements,
  not posterior uncertainty from a single measurement. The most direct
  and honest way to quantify reliability is to run the decomposition
  on each repetition and compute statistics across repetitions.
  This measures actual experimental variability, which is what matters
  for Objective 2.3.2.

  WHY WEIGHTED LEAST SQUARES OVER PLAIN TIKHONOV
  ────────────────────────────────────────────────
  Standard least squares minimises ||s - M·a||² equally across all
  time samples. This means large-amplitude modes dominate the fit
  because they contribute most to the total error. Your modes of
  interest (S1, S2, S4) may have smaller amplitudes and get poorly
  estimated as a result.

  WLS instead minimises ||W·(s - M·a)||² where W is a diagonal
  weight matrix that is large around the predicted arrival time of
  your modes of interest and smaller elsewhere. This forces the solver
  to prioritise fitting the signal correctly where your modes live.

  RELIABILITY QUANTIFICATION
  ───────────────────────────
  Run process_file() on each repetition file and collect results.
  The reliability metrics are then computed across repetitions:
    mean(a_n)  →  best amplitude estimate
    std(a_n)   →  measurement precision
    CV(a_n)    →  std/mean × 100%  — your primary reliability metric
  Plot CV vs propagation distance for each mode of interest.

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert, butter, filtfilt, windows
from scipy.interpolate import interp1d
from sklearn.linear_model import LassoCV
import os, warnings
warnings.filterwarnings("ignore")

from decomposition_v9_1 import (
    load_file, load_dispersion, build_dictionary,
    envelope_peaks, verify_against_gt, normalise_matrix,
    FUNDAMENTAL_FILE, SECOND_HARM_FILE, DISP_FILES, PROPAGATION_MM
)

os.makedirs("plots", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION — modes of interest per band
# ─────────────────────────────────────────────────────────────────────────────

# These modes are ALWAYS included in Stage 2 regardless of what LASSO finds.
# They are your research targets — you need their amplitudes even if they
# are small and LASSO would have zeroed them out.
FORCED_FUNDAMENTAL   = ["S1", "A1", "A2"]   # at f  (1.33 MHz)
FORCED_SECOND_HARM   = ["S2", "S4"]          # at 2f (2.66 MHz)

# Weight applied to the arrival window of modes of interest vs elsewhere.
# Higher = more focus on those windows, less on the rest of the signal.
# 10.0 is a good starting point — increase if your modes of interest
# have much smaller amplitude than other modes in the signal.
WINDOW_WEIGHT = 10.0

# Half-width of the arrival window around each predicted mode arrival (μs).
# Should be roughly half the expected packet duration.
# Wider = more signal included, narrower = more focused but may clip packet.
WINDOW_HALF_WIDTH_US = 20

# Tikhonov regularisation for Stage 2 WLS solve.
# Smaller than standard because we've already reduced the mode set in Stage 1.
REGULARIZATION_WLS = 1e-5


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — LASSO MODE IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def stage1_lasso(s, M, mode_names, n_cv_folds=5):
    """
    Run LASSO with cross-validated λ to identify which modes are active.

    We are not trusting the LASSO amplitudes here — only the sparsity
    pattern (which modes are non-zero). The actual amplitudes come from
    Stage 2 where we use a more targeted solver.

    Returns
    -------
    active_modes : list of mode names LASSO identified as non-zero
    lambda_opt   : the λ selected by cross-validation (diagnostic)
    """
    s0, M_norm, scales = normalise_matrix(s, M)

    lasso_cv = LassoCV(
        cv            = n_cv_folds,
        max_iter      = 10000,
        n_alphas      = 100,
        fit_intercept = False,
        random_state  = 42,
    )
    lasso_cv.fit(M_norm, s0)

    lambda_opt   = float(lasso_cv.alpha_)
    a_sc         = lasso_cv.coef_
    active_modes = [m for m, coef in zip(mode_names, a_sc)
                    if abs(coef) > 1e-12]

    print(f"  Stage 1 LASSO — λ = {lambda_opt:.6f}")
    print(f"  Active modes identified : {active_modes}")
    print(f"  Zeroed modes            : "
          f"{[m for m in mode_names if m not in active_modes]}")

    return active_modes, lambda_opt


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — WEIGHTED LEAST SQUARES
# ─────────────────────────────────────────────────────────────────────────────

def build_weight_matrix(t, info, modes_of_interest,
                        window_weight=WINDOW_WEIGHT,
                        half_width=WINDOW_HALF_WIDTH_US):
    """
    Build diagonal weight matrix W that emphasises the arrival windows
    of the modes of interest.

    For each mode of interest:
      - Find its predicted arrival time t_pred from the dispersion data
      - Create a Hann window of width 2*half_width centred on t_pred
      - Add it to the weight array

    WHY HANN WINDOW (not rectangular):
      A rectangular window creates sharp edges that introduce Gibbs
      ringing when multiplied with the signal. The Hann window tapers
      smoothly to zero, avoiding this. It also means the weighting
      transitions gradually rather than abruptly, which is more
      physically appropriate since mode packets have gradual onsets.

    Regions outside all mode windows get weight 1.0 (not zero) so the
    solver still uses all the data — it just prioritises the windows.
    """
    N = len(t)
    dt = float(np.mean(np.diff(t)))
    w  = np.ones(N)  # baseline weight = 1.0 everywhere

    for mode in modes_of_interest:
        if mode not in info:
            continue
        t_pred = info[mode].get("t_pred", np.nan)
        if np.isnan(t_pred):
            continue

        # Build Hann window around predicted arrival
        half_samples = int(half_width / dt)
        win_len      = 2 * half_samples + 1
        hann         = windows.hann(win_len)  #changed to hamming to see difference

        # Find centre index
        centre_idx = int(np.argmin(np.abs(t - t_pred)))
        i_start    = max(0, centre_idx - half_samples)
        i_end      = min(N, centre_idx + half_samples + 1)

        # Trim window if near signal edges
        hann_start = half_samples - (centre_idx - i_start)
        hann_end   = hann_start + (i_end - i_start)
        hann_trim  = hann[hann_start:hann_end]

        # Add weighted contribution — peaks at window_weight, tapers to 1
        w[i_start:i_end] += (window_weight - 1.0) * hann_trim

    return w


def stage2_wls(s, M, mode_names, t, info, forced_modes,
               regularization=REGULARIZATION_WLS):
    """
    Weighted Least Squares solve focused on modes of interest.

    Minimises: ||W·(s - M·a)||²  +  λ·||a||²

    where W upweights the time windows around forced_modes arrivals.

    This is equivalent to standard Tikhonov but on the weighted system:
      W·s = W·M·a  →  (WM)'(WM)·a + λI·a = (WM)'(Ws)

    Returns
    -------
    amplitudes : dict {mode: float}  — physical amplitude scalars
    residual   : relative reconstruction error
    condition  : condition number of weighted normalised matrix
    """
    # Build weight matrix focused on forced modes
    w = build_weight_matrix(t, info, forced_modes)
    W = w  # diagonal — apply as elementwise multiply

    # Apply weighting
    s_w  = W * s
    M_w  = W[:, np.newaxis] * M

    # Normalise weighted system
    s0_w, M_norm_w, scales = normalise_matrix(s_w, M_w)

    cond = np.linalg.cond(M_norm_w)
    lam  = regularization

    a_sc = np.linalg.solve(
        M_norm_w.T @ M_norm_w + lam * np.eye(len(mode_names)),
        M_norm_w.T @ s0_w)
    a = a_sc / scales

    # Residual on UNWEIGHTED signal — true reconstruction quality
    rec   = M @ a
    resid = float(np.linalg.norm(s - rec) / (np.linalg.norm(s) + 1e-12))

    amplitudes = {m: float(a[i]) for i, m in enumerate(mode_names)}
    return amplitudes, resid, cond


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE FOR ONE FILE
# ─────────────────────────────────────────────────────────────────────────────

def process_file(filepath, disp, forced_modes, label="", verbose=True):
    """
    Run the full two-stage pipeline on one signal file.

    Parameters
    ----------
    filepath     : path to Excel signal file
    disp         : dispersion dict from load_dispersion()
    forced_modes : list of mode names always included in Stage 2
                   (your modes of interest)
    label        : string label for printing and plots

    Returns
    -------
    dict with amplitudes, envelope peaks, verification results,
    and all arrays needed for plotting and reliability analysis.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    t, s, exc, fs, gt, f_centre = load_file(filepath)
    N = len(t)

    # Build full modal dictionary
    modal_waveforms, info, M_full, mode_names_full = build_dictionary(
        disp, exc, fs, N, f_centre, PROPAGATION_MM)

    # ── Stage 1: LASSO identifies active modes ────────────────────────────
    if verbose:
        print(f"\n  [Stage 1] LASSO mode identification...")
    active_lasso, lambda_opt = stage1_lasso(s, M_full, mode_names_full)

    # ── Combine LASSO active + forced modes of interest ───────────────────
    # Forced modes are always included even if LASSO zeroed them.
    # This ensures we always get an amplitude estimate for S1, S2, S4 etc.
    # even when their amplitude is small enough that LASSO suppressed them.
    stage2_modes = sorted(set(active_lasso) |
                          set(m for m in forced_modes
                              if m in modal_waveforms))

    forced_added = [m for m in forced_modes
                    if m in modal_waveforms and m not in active_lasso]
    if forced_added and verbose:
        print(f"  Forced modes added (not in LASSO active set): {forced_added}")
        print(f"    → These had small amplitude — LASSO suppressed them")
        print(f"    → Included anyway because they are your research targets")

    if verbose:
        print(f"  Stage 2 mode set: {stage2_modes}")

    M_stage2 = np.column_stack([modal_waveforms[m] for m in stage2_modes])

    # ── Stage 2: Weighted least squares ───────────────────────────────────
    if verbose:
        print(f"\n  [Stage 2] Weighted least squares (focusing on {forced_modes})...")

    amplitudes, resid, cond = stage2_wls(
        s, M_stage2, stage2_modes, t, info, forced_modes)

    if verbose:
        print(f"  Condition number : {cond:.3e}"
              f"  {'(OK)' if cond < 1e6 else '⚠ ill-conditioned'}")
        print(f"  Relative residual: {resid:.4f}")

    # Reconstruction
    rec = sum(amplitudes[m] * modal_waveforms[m] for m in stage2_modes)

    # Envelope peaks — the physical amplitude at receiver
    peaks = envelope_peaks(amplitudes, modal_waveforms)

    # Print amplitude table
    if verbose:
        print(f"\n  ── Amplitudes ──────────────────────────────────────────")
        print(f"  {'Mode':<8} {'a_n':>12} {'Peak (nm)':>12} "
              f"{'t_pred (μs)':>13} {'Forced':>8}")
        print("  " + "─" * 58)
        for mode in stage2_modes:
            t_pred = info[mode].get("t_pred", np.nan)
            t_str  = f"{t_pred:.2f}" if not np.isnan(t_pred) else "—"
            flag   = "  ◄" if mode in forced_modes else ""
            peak   = peaks.get(mode, 0.0)
            print(f"  {mode:<8} {amplitudes[mode]:>12.6f} "
                  f"{peak:>12.6f} {t_str:>13}{flag}")

    # Verify against ground truth (simulation data only)
    verify_results = verify_against_gt(amplitudes, modal_waveforms, gt, label)

    return {
        "t": t, "s": s, "rec": rec,
        "amplitudes": amplitudes, "peaks": peaks,
        "modal_waveforms": modal_waveforms, "info": info,
        "stage2_modes": stage2_modes, "forced_modes": forced_modes,
        "active_lasso": active_lasso,
        "resid": resid, "cond": cond,
        "f_centre": f_centre, "label": label,
        "gt": gt, "verify": verify_results,
        "lambda_opt": lambda_opt,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RELIABILITY ANALYSIS ACROSS REPETITIONS / DISTANCES
# ─────────────────────────────────────────────────────────────────────────────

def reliability_analysis(all_results, modes_of_interest, label=""):
    """
    Compute reliability metrics across multiple runs (repetitions or distances).

    For each mode of interest computes:
      mean   : best amplitude estimate
      std    : measurement precision (spread across repetitions)
      CV     : coefficient of variation = std/mean × 100%
               This is your primary reliability metric.
               CV < 5%  → excellent repeatability
               CV < 10% → acceptable
               CV > 20% → poor — method unreliable for this mode

    Parameters
    ----------
    all_results      : list of result dicts from process_file()
    modes_of_interest: list of mode names to analyse
    label            : string label for printing

    Returns
    -------
    stats : dict {mode: {mean, std, cv, peaks_list}}
    """
    print(f"\n  ── Reliability Analysis — {label} {'─'*(35-len(label))}")
    print(f"  {'Mode':<8} {'Mean (nm)':>12} {'Std (nm)':>12} "
          f"{'CV (%)':>9} {'N':>5} {'Reliable?':>12}")
    print("  " + "─" * 62)

    stats = {}
    for mode in modes_of_interest:
        peaks_list = [r["peaks"].get(mode, np.nan) for r in all_results]
        peaks_arr  = np.array([p for p in peaks_list if not np.isnan(p)])

        if len(peaks_arr) < 2:
            print(f"  {mode:<8} {'—':>12} {'—':>12} {'—':>9} "
                  f"{len(peaks_arr):>5} {'insufficient data':>12}")
            stats[mode] = {"mean": np.nan, "std": np.nan, "cv": np.nan,
                           "peaks": peaks_list}
            continue

        mean = float(np.mean(peaks_arr))
        std  = float(np.std(peaks_arr))
        cv   = std / (mean + 1e-12) * 100

        reliable = ("excellent" if cv < 5 else
                    "acceptable" if cv < 10 else
                    "poor" if cv < 20 else "unreliable")

        stats[mode] = {"mean": mean, "std": std, "cv": cv,
                       "peaks": peaks_list}

        print(f"  {mode:<8} {mean:>12.6f} {std:>12.6f} "
              f"{cv:>9.2f} {len(peaks_arr):>5} {reliable:>12}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_result(result):
    """
    Four-panel diagnostic plot for one file:
    1. Signal vs reconstruction
    2. Individual mode contributions with arrival windows highlighted
    3. Amplitude bar chart (forced modes highlighted)
    4. GT verification (if available)
    """
    t      = result["t"]
    s      = result["s"]
    rec    = result["rec"]
    label  = result["label"]
    forced = result["forced_modes"]
    modes  = result["stage2_modes"]
    info   = result["info"]

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.5)
    fig.suptitle(f"Two-Stage Decomposition — {label}  |  "
                 f"d = {PROPAGATION_MM} mm", fontsize=12)

    # ── Panel 1: signal vs reconstruction ────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    resid = result["resid"]
    ax0.plot(t, s,   color="steelblue", lw=0.8, alpha=0.8, label="Measured s(t)")
    ax0.plot(t, rec, color="tomato",    lw=1.2, ls="--",
             label=f"Stage 2 reconstruction  (resid = {resid:.4f})")
    ax0.set_ylabel("nm"); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
    ax0.set_title("Signal vs Reconstruction")

    # ── Panel 2: mode contributions with arrival windows ─────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, s, color="black", lw=0.4, alpha=0.1, label="Measured")
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(modes), 1)))

    for i, mode in enumerate(modes):
        contrib = result["amplitudes"][mode] * result["modal_waveforms"][mode]
        env     = np.abs(hilbert(contrib))
        t_pred  = info.get(mode, {}).get("t_pred", np.nan)
        lw      = 2.0 if mode in forced else 0.8
        lbl     = (f"{mode} ◄ ({t_pred:.1f}μs)" if mode in forced
                   else f"{mode} ({t_pred:.1f}μs)"
                   if not np.isnan(t_pred) else mode)
        ax1.plot(t, contrib, color=cmap[i], lw=lw, label=lbl)
        ax1.plot(t, env,     color=cmap[i], lw=0.7, ls=":", alpha=0.6)
        # Show arrival window for forced modes
        if mode in forced and not np.isnan(t_pred):
            ax1.axvspan(t_pred - WINDOW_HALF_WIDTH_US,
                        t_pred + WINDOW_HALF_WIDTH_US,
                        alpha=0.06, color=cmap[i])

    ax1.set_ylabel("nm"); ax1.legend(ncol=3, fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Mode contributions  (◄ = forced modes of interest, "
                  "shaded = weighted window)")

    # ── Panel 3: amplitude bar chart ─────────────────────────────────────
    ax2    = fig.add_subplot(gs[2])
    peaks  = result["peaks"]
    vals   = [peaks.get(m, 0.0) for m in modes]
    colors = ["tomato" if m in forced else "steelblue" for m in modes]
    bars   = ax2.bar(modes, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 1e-8:
            ax2.text(bar.get_x() + bar.get_width()/2, v * 1.02,
                     f"{v:.5f}", ha="center", va="bottom",
                     fontsize=7, rotation=45)
    ax2.set_title("Envelope peak amplitudes  "
                  "(red = forced modes of interest, blue = LASSO-identified)")
    ax2.set_ylabel("|A| (nm)"); ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=45)

    # ── Panel 4: verification ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    vr  = result["verify"]
    if vr:
        vm    = sorted(vr.keys())
        gt_v  = [vr[m]["gt_peak"]  for m in vm]
        rec_v = [vr[m]["rec_peak"] for m in vm]
        err_v = [vr[m]["err_pct"]  for m in vm]
        x     = np.arange(len(vm)); w = 0.35
        ax3.bar(x - w/2, gt_v,  w, label="GT",        color="steelblue", alpha=0.85)
        ax3.bar(x + w/2, rec_v, w, label="Stage 2 WLS", color="tomato",  alpha=0.85)
        ax3.set_xticks(x); ax3.set_xticklabels(vm, rotation=45, fontsize=8)

        # Colour x-tick labels red for forced modes
        for tick, m in zip(ax3.get_xticklabels(), vm):
            if m in forced:
                tick.set_color("tomato"); tick.set_fontweight("bold")

        ax3.set_ylabel("nm"); ax3.legend(fontsize=8)
        ax3.set_title("GT vs Recovered  (bold red labels = modes of interest)")
        ax3.grid(True, alpha=0.3, axis="y")

        # Annotate error % on bars
        for xi, (gv, rv, ev) in enumerate(zip(gt_v, rec_v, err_v)):
            color = "tomato" if ev > 50 else "orange" if ev > 20 else "green"
            ax3.text(xi + w/2, rv * 1.02, f"{ev:.0f}%",
                     ha="center", va="bottom", fontsize=7, color=color)
    else:
        ax3.text(0.5, 0.5, "No GT available\n(experimental data)",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=11)

    fname = f"plots/twostage_{label.replace(' ', '_').replace('/', '')}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.show()


def plot_reliability(stats_f, stats_2f,
                     forced_f=FORCED_FUNDAMENTAL,
                     forced_2f=FORCED_SECOND_HARM):
    """
    Reliability summary plot — mean ± std and CV% for all modes of interest.
    This is your key results figure for Objective 2.3.2.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Amplitude Reliability — Modes of Interest  |  "
                 f"d = {PROPAGATION_MM} mm", fontsize=12)

    for row, (stats, forced, band) in enumerate([
        (stats_f,  forced_f,  "Fundamental (f)"),
        (stats_2f, forced_2f, "2nd Harmonic (2f)"),
    ]):
        modes  = [m for m in forced if m in stats]
        means  = [stats[m]["mean"] for m in modes]
        stds   = [stats[m]["std"]  for m in modes]
        cvs    = [stats[m]["cv"]   for m in modes]

        # Mean ± std
        ax = axes[row, 0]
        ax.bar(modes, means, yerr=stds, color="steelblue",
               capsize=8, edgecolor="white", error_kw={"elinewidth": 2})
        ax.set_title(f"{band} — Mean ± std across repetitions")
        ax.set_ylabel("|A| (nm)"); ax.grid(True, alpha=0.3, axis="y")

        # CV%
        ax = axes[row, 1]
        bar_colors = ["green" if cv < 5 else
                      "orange" if cv < 10 else
                      "tomato" if cv < 20 else "darkred"
                      for cv in cvs]
        ax.bar(modes, cvs, color=bar_colors, edgecolor="white")
        ax.axhline(5,  color="green",  lw=1.5, ls="--", label="5%  — excellent")
        ax.axhline(10, color="orange", lw=1.5, ls="--", label="10% — acceptable")
        ax.axhline(20, color="tomato", lw=1.5, ls="--", label="20% — poor")
        ax.set_title(f"{band} — Coefficient of Variation (%)")
        ax.set_ylabel("CV (%)"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        for ax in axes[row]:
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/reliability_summary.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/reliability_summary.png")
    plt.show()


def plot_cv_vs_distance(stats_per_distance, modes_of_interest, distances_mm):
    """
    Plot CV% vs propagation distance for each mode of interest.
    Shows how measurement reliability degrades (or holds) with distance.
    This directly supports Objective 2.3.2 — how precision varies with distance.

    Parameters
    ----------
    stats_per_distance : list of stats dicts, one per distance
    modes_of_interest  : list of mode names to plot
    distances_mm       : list of propagation distances in mm
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Measurement Reliability vs Propagation Distance", fontsize=12)

    cmap = plt.cm.tab10(np.linspace(0, 1, len(modes_of_interest)))

    for ax_idx, (stats_list, label) in enumerate([
        (stats_per_distance, "All modes of interest")
    ]):
        ax = axes[0]
        for i, mode in enumerate(modes_of_interest):
            cvs   = [s.get(mode, {}).get("cv", np.nan) for s in stats_list]
            means = [s.get(mode, {}).get("mean", np.nan) for s in stats_list]
            ax.plot(distances_mm, cvs, "o-", color=cmap[i],
                    lw=2, ms=7, label=mode)
        ax.axhline(5,  color="green",  lw=1, ls="--", alpha=0.7)
        ax.axhline(10, color="orange", lw=1, ls="--", alpha=0.7)
        ax.axhline(20, color="tomato", lw=1, ls="--", alpha=0.7)
        ax.set_xlabel("Propagation distance (mm)")
        ax.set_ylabel("CV (%)")
        ax.set_title("Coefficient of Variation vs Distance\n"
                     "(lower = more reliable)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1]
        for i, mode in enumerate(modes_of_interest):
            means = [s.get(mode, {}).get("mean", np.nan) for s in stats_list]
            stds  = [s.get(mode, {}).get("std",  np.nan) for s in stats_list]
            ax.errorbar(distances_mm, means, yerr=stds,
                        fmt="o-", color=cmap[i], lw=2, ms=7,
                        capsize=5, label=mode)
        ax.set_xlabel("Propagation distance (mm)")
        ax.set_ylabel("|A| (nm)")
        ax.set_title("Mean amplitude ± std vs Distance")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/cv_vs_distance.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/cv_vs_distance.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  TWO-STAGE LAMB WAVE DECOMPOSITION")
    print("="*60)

    # ── Load dispersion ────────────────────────────────────────────────────
    print("\n[0] Loading dispersion curves...")
    disp = load_dispersion(DISP_FILES)

    # ── Single file run (200mm simulation data) ────────────────────────────
    print("\n[1] Processing fundamental file (f = 1.33 MHz)...")
    result_f = process_file(
        FUNDAMENTAL_FILE, disp,
        forced_modes = FORCED_FUNDAMENTAL,
        label        = "Fundamental 200mm")

    print("\n[2] Processing second harmonic file (2f = 2.66 MHz)...")
    result_2f = process_file(
        SECOND_HARM_FILE, disp,
        forced_modes = FORCED_SECOND_HARM,
        label        = "2nd Harmonic 200mm")

    # ── Plots for single run ───────────────────────────────────────────────
    print("\n[3] Plotting single-run results...")
    plot_single_result(result_f)
    plot_single_result(result_2f)

    # ── Reliability across repetitions ─────────────────────────────────────
    # When you have multiple repetition files, replace the lists below.
    # For now we simulate with a single result to show the structure.
    #
    # USAGE WITH REAL REPETITIONS:
    #   rep_files_f  = ["rep1_f.xlsx", "rep2_f.xlsx", ..., "rep10_f.xlsx"]
    #   rep_files_2f = ["rep1_2f.xlsx", ..., "rep10_2f.xlsx"]
    #   results_f    = [process_file(f, disp, FORCED_FUNDAMENTAL,
    #                                label=f"Rep {i+1} f",
    #                                verbose=False)
    #                   for i, f in enumerate(rep_files_f)]
    #   results_2f   = [process_file(f, disp, FORCED_SECOND_HARM,
    #                                label=f"Rep {i+1} 2f",
    #                                verbose=False)
    #                   for i, f in enumerate(rep_files_2f)]

    # Single result wrapped in list for demonstration
    results_f  = [result_f]
    results_2f = [result_2f]

    print("\n[4] Reliability analysis...")
    stats_f  = reliability_analysis(
        results_f,  FORCED_FUNDAMENTAL, label="Fundamental")
    stats_2f = reliability_analysis(
        results_2f, FORCED_SECOND_HARM, label="2nd Harmonic")

    plot_reliability(stats_f, stats_2f)

    # ── Multi-distance example ─────────────────────────────────────────────
    # When you add the 250mm data, extend this:
    #
    # distances = [200, 250]
    # stats_per_dist_f = []
    # for dist, f_file, f2_file in zip(distances, fund_files, harm_files):
    #     res = process_file(f_file, disp, FORCED_FUNDAMENTAL, verbose=False)
    #     st  = reliability_analysis([res], FORCED_FUNDAMENTAL, label="")
    #     stats_per_dist_f.append(st)
    #
    # plot_cv_vs_distance(stats_per_dist_f,
    #                     FORCED_FUNDAMENTAL + FORCED_SECOND_HARM,
    #                     distances)

    print("\n" + "="*60 + "\n  DONE\n" + "="*60)