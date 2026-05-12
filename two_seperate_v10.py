"""
═══════════════════════════════════════════════════════════════════════════════
  decomposition_twostage.py
  ─────────────────────────────────────────────────────────────────────────────

  THREE-STAGE LAMB WAVE AMPLITUDE EXTRACTION
  ───────────────────────────────────────────
  Designed to maximise amplitude accuracy specifically for:
    At f  : S1, A1, A2
    At 2f : S2, S4

  STAGE 1 — LASSO mode identification
    Identifies which modes are active without prior knowledge.
    Output: sparse active mode set.

  STAGE 2 — Sequential subtraction (2f only)
    Dominant modes A0/S0 overlap temporally with S2/S4 and are much
    larger. They are solved on the full signal then subtracted, leaving
    a residual where S2/S4 are the dominant features.

  STAGE 3 — Matched filter with Gram matrix correction
    Replaces WLS. Directly maximises amplitude estimation accuracy
    for the forced modes of interest.

    WHY MATCHED FILTER OVER WLS
    ────────────────────────────
    WLS minimises total reconstruction error equally across all modes.
    It has no special preference for getting S1 or S2 right — it just
    minimises the sum. If S0 is large and S2 is small, the solver
    concentrates its effort on S0.

    Matched filter computes the amplitude of each target mode as the
    inner product of the signal with that mode's predicted waveform:

        a_n_raw = <s, m_n> / <m_n, m_n>

    This is the optimal linear estimator for a_n when the other modes
    are absent. The problem is that other modes are present and their
    waveforms are not perfectly orthogonal to m_n — they leak into the
    matched filter output and bias a_n_raw.

    The Gram matrix correction removes this cross-talk bias:

        G_ij = <m_i, m_j> / (||m_i|| ||m_j||)   [cosine similarity]

        a_corrected = G^-1 · a_raw

    G^-1 redistributes the amplitude estimates to cancel the cross-talk
    between modes. The result is the amplitude vector that, given the
    observed inner products with each mode's waveform, is most consistent
    with all modes being present simultaneously.

    WHY THIS MAXIMISES ACCURACY FOR TARGET MODES
    ──────────────────────────────────────────────
    The matched filter is the maximum likelihood estimator for each
    amplitude individually under white noise. The Gram matrix correction
    makes it unbiased under correlated (multi-mode) noise. Together they
    form the minimum variance unbiased estimator (MVUE) for each a_n —
    meaning no other linear estimator can achieve lower error on the
    target mode amplitudes given the same data.

    In practice: if S1 and A1 have overlapping waveforms, WLS will
    trade off accuracy between them. Matched filter + Gram correction
    gives each its own optimal estimate without that tradeoff.

    WHEN IT CAN FAIL
    ─────────────────
    If two modes have very similar waveforms (cosine similarity > 0.95),
    G becomes nearly singular and the correction amplifies noise.
    This is diagnosed by the condition number of G — if cond(G) > 1e4
    the two modes cannot be reliably separated and should be merged or
    one dropped. The code reports this automatically.

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert, windows
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
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

FORCED_FUNDAMENTAL   = ["S1", "S2", "A2"]
FORCED_SECOND_HARM   = ["S2", "S4"]
DOMINANT_SECOND_HARM = ["A0", "S0"]

# Gram matrix regularisation — prevents instability when two modes are
# nearly identical in waveform shape. Increase if cond(G) > 1e4.
GRAM_REGULARIZATION  = 1e-3

# Tikhonov regularisation for Stage 2 full solve only.
REGULARIZATION_WLS   = 1e-5

# Condition number threshold above which a warning is printed.
# If exceeded, two or more forced modes are too similar to separate reliably.
GRAM_COND_WARN       = 1e4


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — LASSO MODE IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def stage1_lasso(s, M, mode_names, n_cv_folds=5):
    """
    LASSO with cross-validated lambda. Returns sparsity pattern only.
    Amplitudes come from Stage 3.
    """
    s0, M_norm, scales = normalise_matrix(s, M)
    lasso_cv = LassoCV(cv=n_cv_folds, max_iter=10000,
                       n_alphas=100, fit_intercept=False, random_state=42)
    lasso_cv.fit(M_norm, s0)

    lambda_opt   = float(lasso_cv.alpha_)
    active_modes = [m for m, c in zip(mode_names, lasso_cv.coef_)
                    if abs(c) > 1e-12]

    print(f"  Stage 1 LASSO  lambda={lambda_opt:.6f}")
    print(f"  Active : {active_modes}")
    print(f"  Zeroed : {[m for m in mode_names if m not in active_modes]}")
    return active_modes, lambda_opt


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — SEQUENTIAL SUBTRACTION  (2f only)
# ─────────────────────────────────────────────────────────────────────────────

def solve_wls(s, M, mode_names, regularization=REGULARIZATION_WLS):
    """Standard Tikhonov WLS. Used in Stage 2 to estimate dominant modes."""
    s0, M_norm, scales = normalise_matrix(s, M)
    cond = np.linalg.cond(M_norm)
    a_sc = np.linalg.solve(
        M_norm.T @ M_norm + regularization * np.eye(len(mode_names)),
        M_norm.T @ s0)
    a     = a_sc / scales
    resid = float(np.linalg.norm(s - M @ a) / (np.linalg.norm(s) + 1e-12))
    return {m: float(a[i]) for i, m in enumerate(mode_names)}, resid, cond


def stage2_subtract(s, modal_waveforms, dominant_modes,
                    all_stage2_modes, verbose=True):
    """
    Solve all Stage 2 modes on full signal, subtract dominant contributions.

    WHY SOLVE ALL MODES BEFORE SUBTRACTING:
    Solving dominant modes alone would absorb energy from the forced modes
    (S2, S4) into the dominant mode estimates, causing over-subtraction.
    Solving all modes together gives unbiased dominant mode estimates
    before subtracting.
    """
    avail = [m for m in all_stage2_modes if m in modal_waveforms]
    M_all = np.column_stack([modal_waveforms[m] for m in avail])
    amps_all, resid_full, cond_full = solve_wls(s, M_all, avail)

    if verbose:
        print(f"  Full solve  cond={cond_full:.2e}  resid={resid_full:.4f}")

    s_residual    = s.copy()
    amps_dominant = {}
    for mode in dominant_modes:
        if mode not in amps_all or mode not in modal_waveforms:
            continue
        s_residual        -= amps_all[mode] * modal_waveforms[mode]
        amps_dominant[mode] = amps_all[mode]

    rel = float(np.linalg.norm(s_residual) / (np.linalg.norm(s) + 1e-12))
    if verbose:
        print(f"  Subtracted: {list(amps_dominant.keys())}")
        print(f"  Residual / original norm: {rel:.4f}"
              + ("  (OK)" if rel < 1.0 else
                 "  WARNING: >1.0 — subtraction degraded signal, "
                 "check DOMINANT_SECOND_HARM"))
    return s_residual, amps_dominant


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — MATCHED FILTER WITH GRAM MATRIX CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def stage3_matched_filter(s, modal_waveforms, forced_modes,
                           context_modes=None,
                           gram_reg=GRAM_REGULARIZATION,
                           verbose=True):
    """
    Matched filter decomposition with Gram matrix cross-talk correction.
    Maximises amplitude estimation accuracy for forced_modes specifically.

    Parameters
    ----------
    s             : signal (may be residual after Stage 2 subtraction)
    modal_waveforms: dict of all synthesized mode waveforms
    forced_modes  : modes to extract with maximum accuracy
    context_modes : additional modes from LASSO active set to include in
                    the Gram correction (improves accuracy if other modes
                    are still present in s after subtraction)
    gram_reg      : regularisation added to Gram matrix diagonal to
                    stabilise inversion when modes are similar

    Returns
    -------
    amplitudes    : dict {mode: amplitude}  — optimally estimated
    gram_cond     : condition number of Gram matrix (diagnostic)
    cross_talk    : dict showing how much each pair of modes leaks into
                    each other — useful for your report

    STEP BY STEP
    ─────────────
    1. Collect all modes: forced ∪ context
    2. Normalise each waveform to unit norm (required for matched filter)
    3. Compute raw matched filter outputs:
         a_raw_n = <s, m_n_normalised>
       This is the projection of s onto each mode's direction.
       It equals the true amplitude only if modes are orthogonal.
    4. Build Gram matrix G:
         G_ij = <m_i_normalised, m_j_normalised>
       G_ii = 1 always. G_ij measures overlap between modes i and j.
       If G_ij = 0, modes are orthogonal — no cross-talk.
       If G_ij = 1, modes are identical — cannot be separated.
    5. Solve G · a_corrected = a_raw  (regularised)
       This inverts the cross-talk to get unbiased amplitude estimates.
    6. Rescale from normalised back to physical units.
    """
    # Assemble solve set
    all_modes = sorted(set(forced_modes) |
                       set(context_modes or []))
    avail     = [m for m in all_modes if m in modal_waveforms]

    if not avail:
        print("  Stage 3: no modes available.")
        return {}, np.inf, {}

    # Normalise waveforms to unit norm
    # Store norms for rescaling back to physical amplitudes
    norms = {}
    m_hat = {}   # unit-norm waveforms
    for mode in avail:
        mw         = modal_waveforms[mode]
        n          = float(np.linalg.norm(mw))
        norms[mode] = n if n > 1e-12 else 1.0
        m_hat[mode] = mw / norms[mode]

    # Step 3: raw matched filter outputs
    # a_raw_n = <s, m_hat_n>  = projection of signal onto mode direction
    a_raw = np.array([float(np.dot(s, m_hat[m])) for m in avail])

    # Step 4: Gram matrix
    # G_ij = <m_hat_i, m_hat_j>  = cosine similarity between modes
    n_modes = len(avail)
    G = np.zeros((n_modes, n_modes))
    for i, mi in enumerate(avail):
        for j, mj in enumerate(avail):
            G[i, j] = float(np.dot(m_hat[mi], m_hat[mj]))

    gram_cond = float(np.linalg.cond(G))

    if verbose:
        print(f"  Gram matrix condition number: {gram_cond:.3e}")
        if gram_cond > GRAM_COND_WARN:
            print(f"  WARNING: cond(G) > {GRAM_COND_WARN:.0e} — some forced modes "
                  f"have very similar waveforms and cannot be reliably separated.")
            print(f"  Consider removing one of the nearly-identical modes.")
        # Print cross-talk table
        print(f"\n  Cross-talk matrix (G_ij = cosine similarity):")
        print(f"  {'':>6}", end="")
        for m in avail:
            print(f"  {m:>6}", end="")
        print()
        for i, mi in enumerate(avail):
            print(f"  {mi:>6}", end="")
            for j in range(n_modes):
                val = G[i, j]
                marker = " *" if i != j and abs(val) > 0.3 else "  "
                print(f"  {val:>5.3f}{marker}"[:8], end="")
            print()
        print(f"  (* = |cross-talk| > 0.3 — significant overlap)")

    # Step 5: solve G · a_corrected = a_raw  with regularisation
    G_reg = G + gram_reg * np.eye(n_modes)
    try:
        a_corrected_norm = np.linalg.solve(G_reg, a_raw)
    except np.linalg.LinAlgError:
        print("  WARNING: Gram matrix solve failed. Falling back to raw estimates.")
        a_corrected_norm = a_raw

    # Step 6: rescale to physical units
    # a_corrected_norm is in units of normalised waveform
    # physical amplitude = a_corrected_norm / norm(m_n)
    amplitudes   = {}
    cross_talk   = {}
    for i, mode in enumerate(avail):
        amplitudes[mode] = float(a_corrected_norm[i]) / norms[mode]

    # Cross-talk report: how much does each OTHER mode contaminate each target?
    for i, mi in enumerate(avail):
        if mi not in forced_modes:
            continue
        cross_talk[mi] = {}
        for j, mj in enumerate(avail):
            if i != j:
                cross_talk[mi][mj] = float(G[i, j])

    if verbose:
        print(f"\n  {'Mode':<8} {'Raw a_n':>12} {'Corrected a_n':>14} "
              f"{'Peak (nm)':>12} {'Forced':>8}")
        print("  " + "─" * 58)
        for i, mode in enumerate(avail):
            raw_phys = float(a_raw[i]) / norms[mode]
            cor_phys = amplitudes[mode]
            env      = np.abs(hilbert(amplitudes[mode] * modal_waveforms[mode]))
            peak     = float(np.max(env))
            flag     = "  ◄" if mode in forced_modes else ""
            print(f"  {mode:<8} {raw_phys:>12.6f} {cor_phys:>14.6f} "
                  f"{peak:>12.6f}{flag}")

    return amplitudes, gram_cond, cross_talk


# ────────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE — FUNDAMENTAL FILE
# ────────────────────────────────────────────────────────────────────────────

def process_fundamental(filepath, disp,
                        forced_modes=FORCED_FUNDAMENTAL,
                        label="Fundamental", verbose=True):
    """
    Stage 1 (LASSO) + Stage 3 (matched filter).
    No subtraction needed at f — modes are temporally separable enough.
    LASSO active modes are included as context in the Gram correction
    so their cross-talk with forced modes is accounted for.
    """
    if verbose:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")

    t, s, exc, fs, gt, f_centre = load_file(filepath)
    N = len(t)
    modal_waveforms, info, M_full, mode_names_full = build_dictionary(
        disp, exc, fs, N, f_centre, PROPAGATION_MM)

    if verbose:
        print(f"\n  [Stage 1] LASSO...")
    active_lasso, lambda_opt = stage1_lasso(s, M_full, mode_names_full)

    forced_in_dict  = [m for m in forced_modes if m in modal_waveforms]
    forced_added    = [m for m in forced_in_dict if m not in active_lasso]
    context_modes   = [m for m in active_lasso if m not in forced_modes
                       and m in modal_waveforms]

    if forced_added and verbose:
        print(f"  Forced modes added (LASSO zeroed): {forced_added}")
    if verbose:
        print(f"  Context modes for Gram correction: {context_modes}")
        print(f"\n  [Stage 3] Matched filter with Gram correction...")

    amplitudes, gram_cond, cross_talk = stage3_matched_filter(
        s, modal_waveforms, forced_in_dict,
        context_modes=context_modes, verbose=verbose)

    all_modes = sorted(amplitudes.keys())
    rec       = sum(amplitudes[m] * modal_waveforms[m]
                    for m in all_modes if m in modal_waveforms)
    resid     = float(np.linalg.norm(s - rec) / (np.linalg.norm(s) + 1e-12))
    peaks     = envelope_peaks(amplitudes, modal_waveforms)

    if verbose:
        print(f"\n  Overall reconstruction residual: {resid:.4f}")

    verify_results = verify_against_gt(amplitudes, modal_waveforms, gt, label)

    return dict(t=t, s=s, rec=rec, amplitudes=amplitudes, peaks=peaks,
                modal_waveforms=modal_waveforms, info=info,
                stage2_modes=all_modes, forced_modes=forced_in_dict,
                active_lasso=active_lasso, context_modes=context_modes,
                resid=resid, gram_cond=gram_cond, cross_talk=cross_talk,
                f_centre=f_centre, label=label, gt=gt,
                verify=verify_results, lambda_opt=lambda_opt,
                s_residual=None, amps_dominant={})


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE — SECOND HARMONIC FILE
# ─────────────────────────────────────────────────────────────────────────────

def process_second_harmonic(filepath, disp,
                             forced_modes=FORCED_SECOND_HARM,
                             dominant_modes=DOMINANT_SECOND_HARM,
                             label="2nd Harmonic", verbose=True):
    """
    Stage 1 (LASSO) + Stage 2 (subtraction) + Stage 3 (matched filter).

    Sequential subtraction removes A0/S0 first so that the matched
    filter in Stage 3 operates on a signal where S2/S4 are dominant.
    The matched filter then extracts their amplitudes with maximum
    accuracy, with Gram correction for any remaining cross-talk.
    """
    if verbose:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")

    t, s, exc, fs, gt, f_centre = load_file(filepath)
    N = len(t)
    modal_waveforms, info, M_full, mode_names_full = build_dictionary(
        disp, exc, fs, N, f_centre, PROPAGATION_MM)

    if verbose:
        print(f"\n  [Stage 1] LASSO...")
    active_lasso, lambda_opt = stage1_lasso(s, M_full, mode_names_full)

    # Stage 2 solve set
    stage2_set = sorted(
        set(active_lasso) |
        set(m for m in forced_modes   if m in modal_waveforms) |
        set(m for m in dominant_modes if m in modal_waveforms))

    if verbose:
        print(f"\n  [Stage 2] Sequential subtraction of {dominant_modes}...")
    s_residual, amps_dominant = stage2_subtract(
        s, modal_waveforms, dominant_modes, stage2_set, verbose)

    # Context for Gram correction = LASSO active modes still present
    # in residual (i.e. not subtracted)
    forced_in_dict = [m for m in forced_modes  if m in modal_waveforms]
    forced_added   = [m for m in forced_in_dict if m not in active_lasso]
    context_modes  = [m for m in active_lasso
                      if m not in forced_modes
                      and m not in dominant_modes
                      and m in modal_waveforms]

    if forced_added and verbose:
        print(f"  Forced modes added (LASSO zeroed): {forced_added}")
    if verbose:
        print(f"  Context modes for Gram correction: {context_modes}")
        print(f"\n  [Stage 3] Matched filter on residual...")

    amps_stage3, gram_cond, cross_talk = stage3_matched_filter(
        s_residual, modal_waveforms, forced_in_dict,
        context_modes=context_modes, verbose=verbose)

    # Combine
    amplitudes = {**amps_dominant, **amps_stage3}
    all_modes  = sorted(amplitudes.keys())
    rec        = sum(amplitudes[m] * modal_waveforms[m]
                     for m in amplitudes if m in modal_waveforms)
    resid_full = float(np.linalg.norm(s - rec) / (np.linalg.norm(s) + 1e-12))
    peaks      = envelope_peaks(amplitudes, modal_waveforms)

    if verbose:
        print(f"\n  Overall reconstruction residual (full signal): {resid_full:.4f}")

    verify_results = verify_against_gt(amplitudes, modal_waveforms, gt, label)

    return dict(t=t, s=s, rec=rec, s_residual=s_residual,
                amplitudes=amplitudes, peaks=peaks,
                amps_dominant=amps_dominant,
                modal_waveforms=modal_waveforms, info=info,
                stage2_modes=all_modes, forced_modes=forced_in_dict,
                active_lasso=active_lasso, context_modes=context_modes,
                resid=resid_full, gram_cond=gram_cond, cross_talk=cross_talk,
                f_centre=f_centre, label=label, gt=gt,
                verify=verify_results, lambda_opt=lambda_opt)


# ─────────────────────────────────────────────────────────────────────────────
#  RELIABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def reliability_analysis(all_results, modes_of_interest, label=""):
    """
    Mean, std, CV across multiple runs (repetitions or distances).
    CV = std/mean * 100% is your primary reliability metric.
      < 5%  : excellent
      < 10% : acceptable
      < 20% : poor
      > 20% : unreliable
    """
    print(f"\n  ── Reliability — {label} {'─'*(38-len(label))}")
    print(f"  {'Mode':<8} {'Mean (nm)':>12} {'Std (nm)':>12} "
          f"{'CV (%)':>9} {'N':>4} {'Quality':>12}")
    print("  " + "─" * 62)

    stats = {}
    for mode in modes_of_interest:
        vals = np.array([r["peaks"].get(mode, np.nan) for r in all_results])
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            stats[mode] = {"mean": np.nan, "std": np.nan,
                           "cv": np.nan, "peaks": []}
            print(f"  {mode:<8} {'—':>12} {'—':>12} {'—':>9} "
                  f"{len(vals):>4} {'no data':>12}")
            continue
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        cv   = std / (mean + 1e-12) * 100
        tag  = ("excellent"  if cv < 5  else
                "acceptable" if cv < 10 else
                "poor"       if cv < 20 else "unreliable")
        stats[mode] = {"mean": mean, "std": std, "cv": cv,
                       "peaks": vals.tolist()}
        print(f"  {mode:<8} {mean:>12.6f} {std:>12.6f} "
              f"{cv:>9.2f} {len(vals):>4} {tag:>12}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_result(result, show_residual=False):
    """Five-panel diagnostic plot."""
    t      = result["t"]
    s      = result["s"]
    rec    = result["rec"]
    label  = result["label"]
    forced = result["forced_modes"]
    modes  = result["stage2_modes"]
    info   = result["info"]

    n_panels = 5 if (show_residual and
                     result.get("s_residual") is not None) else 4
    fig = plt.figure(figsize=(14, 3.8 * n_panels))
    gs  = gridspec.GridSpec(n_panels, 1, figure=fig, hspace=0.52)
    fig.suptitle(
        f"Three-Stage Matched Filter Decomposition — {label}\n"
        f"d = {PROPAGATION_MM} mm  |  "
        f"Gram cond = {result.get('gram_cond', float('nan')):.2e}",
        fontsize=11)

    p = 0

    # Signal vs reconstruction
    ax = fig.add_subplot(gs[p]); p += 1
    ax.plot(t, s,   color="steelblue", lw=0.8, alpha=0.8,
            label="Measured s(t)")
    ax.plot(t, rec, color="tomato",    lw=1.2, ls="--",
            label=f"Reconstruction  resid={result['resid']:.4f}")
    ax.set_ylabel("nm"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Signal vs Reconstruction")

    # Cleaned residual (2f only)
    if show_residual and result.get("s_residual") is not None:
        ax = fig.add_subplot(gs[p]); p += 1
        ax.plot(t, result["s_residual"], color="darkorange", lw=0.8,
                label="After Stage 2 subtraction")
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
        ax.set_ylabel("nm"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Cleaned Signal — S2/S4 should be dominant here")

    # Mode contributions
    ax = fig.add_subplot(gs[p]); p += 1
    ax.plot(t, s, color="black", lw=0.4, alpha=0.1, label="Measured")
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(modes), 1)))
    for i, mode in enumerate(modes):
        if mode not in result["amplitudes"]:
            continue
        contrib = result["amplitudes"][mode] * result["modal_waveforms"][mode]
        env     = np.abs(hilbert(contrib))
        t_pred  = info.get(mode, {}).get("t_pred", np.nan)
        lw      = 2.0 if mode in forced else 0.8
        lbl     = (f"{mode} ◄ ({t_pred:.1f}us)" if mode in forced
                   else f"{mode} ({t_pred:.1f}us)"
                   if not np.isnan(t_pred) else mode)
        ax.plot(t, contrib, color=cmap[i], lw=lw, label=lbl)
        ax.plot(t, env,     color=cmap[i], lw=0.7, ls=":", alpha=0.5)
    ax.set_ylabel("nm"); ax.legend(ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Mode contributions  "
                 "(◄ = forced modes, matched filter optimised)")

    # Amplitude bar chart
    ax     = fig.add_subplot(gs[p]); p += 1
    peaks  = result["peaks"]
    vals   = [peaks.get(m, 0.0) for m in modes]
    colors = ["tomato"    if m in forced else
              "gold"      if m in result.get("amps_dominant", {}) else
              "steelblue" for m in modes]
    bars   = ax.bar(modes, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 1e-8:
            ax.text(bar.get_x() + bar.get_width()/2, v * 1.02,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=6, rotation=45)
    ax.set_title("Envelope peak amplitudes  "
                 "(red=forced/matched-filter, gold=subtracted, blue=context)")
    ax.set_ylabel("|A| (nm)"); ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=45)

    # Verification
    ax = fig.add_subplot(gs[p]); p += 1
    vr = result["verify"]
    if vr:
        vm    = sorted(vr.keys())
        gt_v  = [vr[m]["gt_peak"]  for m in vm]
        rec_v = [vr[m]["rec_peak"] for m in vm]
        err_v = [vr[m]["err_pct"]  for m in vm]
        x = np.arange(len(vm)); bw = 0.35
        ax.bar(x - bw/2, gt_v,  bw, color="steelblue",
               alpha=0.85, label="GT")
        ax.bar(x + bw/2, rec_v, bw, color="tomato",
               alpha=0.85, label="Matched filter")
        ax.set_xticks(x)
        ax.set_xticklabels(vm, rotation=45, fontsize=8)
        for tick, m in zip(ax.get_xticklabels(), vm):
            if m in forced:
                tick.set_color("tomato")
                tick.set_fontweight("bold")
        for xi, (gv, rv, ev) in enumerate(zip(gt_v, rec_v, err_v)):
            col = "green" if ev < 20 else "orange" if ev < 50 else "tomato"
            ax.text(xi + bw/2, rv * 1.02, f"{ev:.0f}%",
                    ha="center", va="bottom", fontsize=7, color=col)
        ax.set_ylabel("nm"); ax.legend(fontsize=8)
        ax.set_title("GT vs Recovered  (bold red = forced modes)")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No GT available\n(experimental data)",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11)

    fname = (f"plots/matched_filter_"
             f"{label.replace(' ','_').replace('/','').replace('.','')}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.show()


def plot_cross_talk(result):
    """
    Visualise the Gram matrix as a heatmap.
    Shows how much each pair of modes leaks into each other.
    Off-diagonal values close to 0 = good separation.
    Off-diagonal values close to 1 = modes are indistinguishable.
    This is a key methodological figure for your report.
    """
    forced = result["forced_modes"]
    ctx    = result.get("context_modes", [])
    modes  = sorted(set(forced) | set(ctx),
                    key=lambda m: (m not in forced, m))

    avail = [m for m in modes if m in result["modal_waveforms"]]
    if len(avail) < 2:
        return

    # Rebuild Gram matrix for plotting
    mws   = result["modal_waveforms"]
    n     = len(avail)
    G     = np.zeros((n, n))
    norms = [float(np.linalg.norm(mws[m])) for m in avail]
    for i, mi in enumerate(avail):
        for j, mj in enumerate(avail):
            G[i, j] = float(np.dot(mws[mi], mws[mj])) / (norms[i] * norms[j])

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n-1)))
    im = ax.imshow(np.abs(G), vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(avail, rotation=45)
    ax.set_yticks(range(n)); ax.set_yticklabels(avail)

    for i in range(n):
        for j in range(n):
            col = "white" if abs(G[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=col)

    # Highlight forced modes
    for i, m in enumerate(avail):
        if m in forced:
            for spine_pos in [i - 0.5, i + 0.5]:
                ax.axhline(spine_pos, color="tomato", lw=2)
                ax.axvline(spine_pos, color="tomato", lw=2)

    plt.colorbar(im, ax=ax, label="|cosine similarity|")
    ax.set_title(
        f"Gram Matrix — {result['label']}\n"
        f"Red borders = forced modes  |  "
        f"Off-diagonal ≈ 0: well separated  |  ≈ 1: indistinguishable\n"
        f"Condition number: {result['gram_cond']:.2e}")

    fname = (f"plots/gram_matrix_"
             f"{result['label'].replace(' ','_').replace('/','').replace('.','')}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.show()


def plot_reliability(stats_f, stats_2f):
    """Mean ± std and CV% for all forced modes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Amplitude Reliability — Matched Filter Decomposition  |  "
                 f"d = {PROPAGATION_MM} mm", fontsize=12)

    for row, (stats, forced, band) in enumerate([
        (stats_f,  FORCED_FUNDAMENTAL, "Fundamental (f = 1.33 MHz)"),
        (stats_2f, FORCED_SECOND_HARM, "2nd Harmonic (2f = 2.66 MHz)"),
    ]):
        modes = [m for m in forced
                 if m in stats and not np.isnan(stats[m]["mean"])]
        means = [stats[m]["mean"] for m in modes]
        stds  = [stats[m]["std"]  for m in modes]
        cvs   = [stats[m]["cv"]   for m in modes]

        ax = axes[row, 0]
        ax.bar(modes, means, yerr=stds, color="steelblue",
               capsize=8, edgecolor="white", error_kw={"elinewidth": 2})
        ax.set_title(f"{band} — Mean ± std")
        ax.set_ylabel("|A| (nm)"); ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

        ax = axes[row, 1]
        bar_colors = ["green"  if cv < 5  else
                      "orange" if cv < 10 else
                      "tomato" if cv < 20 else "darkred" for cv in cvs]
        ax.bar(modes, cvs, color=bar_colors, edgecolor="white")
        ax.axhline(5,  color="green",  lw=1.5, ls="--", label="5%  excellent")
        ax.axhline(10, color="orange", lw=1.5, ls="--", label="10% acceptable")
        ax.axhline(20, color="tomato", lw=1.5, ls="--", label="20% poor")
        ax.set_title(f"{band} — CV (%)")
        ax.set_ylabel("CV (%)"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/reliability_summary.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/reliability_summary.png")
    plt.show()


def plot_cv_vs_distance(stats_per_dist_f, stats_per_dist_2f, distances_mm):
    """CV% and amplitude vs propagation distance for forced modes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Reliability vs Propagation Distance — "
                 "Matched Filter", fontsize=12)
    cmap = plt.cm.tab10

    for col, (stats_list, forced, band) in enumerate([
        (stats_per_dist_f,  FORCED_FUNDAMENTAL, "Fundamental (f)"),
        (stats_per_dist_2f, FORCED_SECOND_HARM, "2nd Harmonic (2f)"),
    ]):
        ax = axes[0, col]
        for i, mode in enumerate(forced):
            cvs = [s.get(mode, {}).get("cv", np.nan) for s in stats_list]
            ax.plot(distances_mm, cvs, "o-", color=cmap(i),
                    lw=2, ms=7, label=mode)
        ax.axhline(5,  color="green",  lw=1, ls="--", alpha=0.6)
        ax.axhline(10, color="orange", lw=1, ls="--", alpha=0.6)
        ax.axhline(20, color="tomato", lw=1, ls="--", alpha=0.6)
        ax.set_xlabel("Distance (mm)"); ax.set_ylabel("CV (%)")
        ax.set_title(f"{band} — CV vs Distance")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        for i, mode in enumerate(forced):
            means = [s.get(mode, {}).get("mean", np.nan) for s in stats_list]
            stds  = [s.get(mode, {}).get("std",  np.nan) for s in stats_list]
            ax.errorbar(distances_mm, means, yerr=stds, fmt="o-",
                        color=cmap(i), lw=2, ms=7, capsize=5, label=mode)
        ax.set_xlabel("Distance (mm)"); ax.set_ylabel("|A| (nm)")
        ax.set_title(f"{band} — Amplitude vs Distance")
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
    print("  THREE-STAGE MATCHED FILTER DECOMPOSITION")
    print("="*60)

    print("\n[0] Loading dispersion curves...")
    disp = load_dispersion(DISP_FILES)

    print("\n[1] Processing fundamental file (f = 1.33 MHz)...")
    result_f = process_fundamental(
        FUNDAMENTAL_FILE, disp,
        forced_modes=FORCED_FUNDAMENTAL,
        label=f"Fundamental {PROPAGATION_MM} mm")

    print("\n[2] Processing second harmonic file (2f = 2.66 MHz)...")
    result_2f = process_second_harmonic(
        SECOND_HARM_FILE, disp,
        forced_modes=FORCED_SECOND_HARM,
        dominant_modes=DOMINANT_SECOND_HARM,
        label=f"2nd Harmonic {PROPAGATION_MM} mm")

    print("\n[3] Plotting results...")
    plot_result(result_f,  show_residual=False)
    plot_result(result_2f, show_residual=True)

    ''' 
    print("\n[4] Plotting Gram matrices (mode separability)...")
    plot_cross_talk(result_f)
    plot_cross_talk(result_2f)

    print("\n[5] Reliability analysis (single file — extend with repetitions)...")
    stats_f  = reliability_analysis(
        [result_f],  FORCED_FUNDAMENTAL, label="Fundamental")
    stats_2f = reliability_analysis(
        [result_2f], FORCED_SECOND_HARM, label="2nd Harmonic")
    plot_reliability(stats_f, stats_2f)
    ''' 
    '''
    # ── Extend with repetition files ──────────────────────────────────────
    
    fund_files = ['In-plane_A2_TemporalResponse@7.9866MHzmm@200mm.xlsx'] #["rep01_f.xlsx",  ..., "rep10_f.xlsx"]
    harm_files = ['In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx'] #["rep01_2f.xlsx", ..., "rep10_2f.xlsx"]
    
    results_f  = [process_fundamental(f, disp, verbose=False)
                  for f in fund_files]
    results_2f = [process_second_harmonic(f, disp, verbose=False)
                  for f in harm_files]
    
    stats_f  = reliability_analysis(results_f,  FORCED_FUNDAMENTAL)
    stats_2f = reliability_analysis(results_2f, FORCED_SECOND_HARM)
    plot_reliability(stats_f, stats_2f)
    
    # ── Extend with 250mm data ────────────────────────────────────────────
    #
    # distances = [200, 250]
    # stats_f_per_dist  = [
    #     reliability_analysis([process_fundamental(
    #         f, disp, verbose=False)], FORCED_FUNDAMENTAL)
    #     for f in [FUNDAMENTAL_FILE, FUND_FILE_250]]
    # stats_2f_per_dist = [
    #     reliability_analysis([process_second_harmonic(
    #         f, disp, verbose=False)], FORCED_SECOND_HARM)
    #     for f in [SECOND_HARM_FILE, HARM_FILE_250]]
    # plot_cv_vs_distance(stats_f_per_dist, stats_2f_per_dist, distances)
    '''
    print("\n" + "="*60 + "\n  DONE\n" + "="*60)