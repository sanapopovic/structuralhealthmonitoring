"""
═══════════════════════════════════════════════════════════════════════════════
  RELIABILITY MODULE FOR LAMB WAVE MODE DECOMPOSITION
  ─────────────────────────────────────────────────────────────────────────────

  This module answers the question:
    "I don't know which modes are present — how do I find out, and how
     do I know whether to trust the amplitudes I recover?"

  FOUR DIAGNOSTICS ARE PROVIDED
  ──────────────────────────────

  1. LASSO SPARSE RECOVERY
     Replaces Tikhonov (L2) with L1 regularisation. LASSO drives absent-mode
     amplitudes to exactly zero — it automatically selects which modes are
     present without you specifying them. The nonzero amplitudes are the
     solver's answer to "which modes exist in this signal?"

  2. CONDITION NUMBER SWEEP (mode selection diagnostic)
     Adds modes to M one by one (sorted by how much energy they explain)
     and tracks cond(M). A sudden jump in condition number means the newly
     added mode is nearly collinear with existing ones — it cannot be
     reliably separated. This tells you the maximum number of modes you
     can reliably decompose with this data.

  3. LEAVE-ONE-OUT AMPLITUDE STABILITY
     Solves the system N_modes times, each time removing one mode from M.
     If removing mode k causes large changes in the amplitudes of other
     modes, those modes are interfering and their amplitudes are unreliable.
     Stable amplitudes (small change when a mode is dropped) are trustworthy.

  4. BOOTSTRAP CONFIDENCE INTERVALS
     Adds small Gaussian noise to s(t) many times and solves each time.
     The standard deviation of each amplitude across bootstrap runs is your
     uncertainty estimate. A narrow interval = reliable. Wide = unreliable.

  HOW TO USE
  ──────────
  Import this alongside your main decomposition script. Call run_all_diagnostics()
  after your main solve() to get a full reliability report.

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  1. LASSO — AUTOMATIC MODE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def lasso_mode_selection(s, modal_waveforms, mode_names, alphas=None):
    """
    Use LASSO (L1 regularisation) to automatically determine which modes
    are present without specifying them in advance.

    LASSO solves:
        min  ||s - M*a||²  +  alpha * ||a||₁

    The L1 penalty drives small (noise-level) amplitudes to exactly zero,
    leaving only the modes that genuinely contribute to the signal.

    Parameters
    ----------
    s              : measured signal [Nt]
    modal_waveforms: dict of synthesized modal waveforms
    mode_names     : list of mode names to test (all candidate modes)
    alphas         : list of regularisation strengths to sweep over
                     (if None, auto-selected from data scale)

    Returns
    -------
    results : dict with keys:
        'selected_modes'   : modes with nonzero amplitude at best alpha
        'amplitudes'       : {mode: amplitude} at best alpha
        'alpha_path'       : array of alpha values tested
        'n_modes_path'     : number of nonzero modes at each alpha
        'residual_path'    : residual at each alpha
        'best_alpha'       : chosen alpha (elbow of n_modes vs residual)
    """
    try:
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import normalize
    except ImportError:
        print("  sklearn not installed. Run: pip install scikit-learn")
        return None

    M_raw  = np.column_stack([modal_waveforms[m] for m in mode_names])
    s0     = s - np.mean(s)
    M0     = M_raw - np.mean(M_raw, axis=0)
    scales = np.linalg.norm(M0, axis=0)
    scales[scales == 0] = 1.0
    M_norm = M0 / scales

    if alphas is None:
        # Auto-range: from nearly-OLS to very sparse
        alpha_max = np.max(np.abs(M_norm.T @ s0)) / len(s)
        alphas = np.logspace(np.log10(alpha_max * 1e-4),
                             np.log10(alpha_max), 60)[::-1]

    alpha_path    = []
    n_modes_path  = []
    residual_path = []
    amplitude_path = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, fit_intercept=False,
                      max_iter=10000, tol=1e-6)
        lasso.fit(M_norm, s0)
        a_scaled = lasso.coef_
        a        = a_scaled / scales
        rec      = M_raw @ a
        resid    = np.linalg.norm(s - rec) / (np.linalg.norm(s) + 1e-12)
        n_nz     = int(np.sum(np.abs(a_scaled) > 1e-10 * np.max(np.abs(a_scaled))))

        alpha_path.append(alpha)
        n_modes_path.append(n_nz)
        residual_path.append(resid)
        amplitude_path.append(a.copy())

    alpha_path    = np.array(alpha_path)
    n_modes_path  = np.array(n_modes_path)
    residual_path = np.array(residual_path)

    # Find elbow: largest alpha where residual is still below 2x its minimum
    resid_thresh = residual_path.min() * 2.0
    candidates   = np.where(residual_path <= resid_thresh)[0]
    # Among candidates, pick the one with fewest modes (most parsimonious)
    best_idx = candidates[np.argmin(n_modes_path[candidates])]
    best_alpha = alpha_path[best_idx]
    best_a     = amplitude_path[best_idx]

    thresh = 1e-10 * np.max(np.abs(best_a)) if np.max(np.abs(best_a)) > 0 else 1e-10
    selected = [m for m, amp in zip(mode_names, best_a) if abs(amp) > thresh]
    amplitudes = {m: float(amp) for m, amp in zip(mode_names, best_a)}

    return {
        "selected_modes":  selected,
        "amplitudes":      amplitudes,
        "alpha_path":      alpha_path,
        "n_modes_path":    n_modes_path,
        "residual_path":   residual_path,
        "best_alpha":      best_alpha,
        "best_idx":        best_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. CONDITION NUMBER SWEEP — HOW MANY MODES CAN WE RELIABLY SEPARATE?
# ─────────────────────────────────────────────────────────────────────────────

def condition_number_sweep(s, modal_waveforms, mode_names):
    """
    Add modes one by one to M, tracking cond(M) at each step.

    Modes are added in order of their individual correlation with s
    (greedily — most-explaining mode first). This mimics how you would
    build the model if you had to choose modes sequentially.

    A sudden jump in condition number (> 10x) when adding a mode means
    that mode is nearly collinear with existing ones. The reliable
    decomposition depth is the number of modes before that jump.

    Returns
    -------
    sweep : list of dicts, one per step:
        'mode'        : mode added at this step
        'cond'        : condition number of M after adding this mode
        'corr_with_s' : correlation of this mode alone with s
        'cumul_resid' : residual of M*a fit after adding this mode
    """
    # Sort modes by correlation with s
    order = []
    for mode in mode_names:
        m_n  = modal_waveforms[mode]
        norm = np.linalg.norm(m_n) * np.linalg.norm(s) + 1e-12
        corr = float(np.abs(np.dot(m_n, s)) / norm)
        order.append((mode, corr))
    order.sort(key=lambda x: -x[1])

    sweep       = []
    cols        = []
    lam         = 1e-6  # tiny regularisation just for numerical stability

    for mode, corr in order:
        cols.append(modal_waveforms[mode])
        M    = np.column_stack(cols)
        M0   = M - np.mean(M, axis=0)
        sc   = np.linalg.norm(M0, axis=0)
        sc[sc == 0] = 1.0
        Mn   = M0 / sc
        cond = float(np.linalg.cond(Mn))
        s0   = s - np.mean(s)
        a_sc = np.linalg.solve(Mn.T @ Mn + lam * np.eye(Mn.shape[1]), Mn.T @ s0)
        a    = a_sc / sc
        resid = float(np.linalg.norm(s - M @ a) / (np.linalg.norm(s) + 1e-12))

        sweep.append({"mode": mode, "cond": cond,
                      "corr_with_s": corr, "cumul_resid": resid})

    return sweep


# ─────────────────────────────────────────────────────────────────────────────
#  3. LEAVE-ONE-OUT STABILITY
# ─────────────────────────────────────────────────────────────────────────────

def leave_one_out_stability(s, modal_waveforms, mode_names, regularization=1e-5):
    """
    Solve the system N times, each time removing one mode.

    For each removed mode k, compute:
        delta_n = |a_n^{-k} - a_n^{full}| / |a_n^{full}|

    where a_n^{full} is the amplitude when all modes are included and
    a_n^{-k} is the amplitude when mode k is removed.

    Large delta means mode k and mode n are interfering — removing k
    significantly changes n's amplitude. Small delta means the estimate
    of mode n's amplitude is stable and does not depend on whether k is
    included.

    Returns
    -------
    stability : dict
        'full_amplitudes'  : {mode: amplitude} with all modes
        'loo_amplitudes'   : {removed_mode: {remaining_mode: amplitude}}
        'instability'      : {mode: max relative change when any other mode removed}
        'reliable'         : {mode: True if instability < 0.1}
    """
    M_raw  = np.column_stack([modal_waveforms[m] for m in mode_names])
    s0     = s - np.mean(s)
    M0     = M_raw - np.mean(M_raw, axis=0)
    sc     = np.linalg.norm(M0, axis=0); sc[sc == 0] = 1.0
    Mn     = M0 / sc
    lam    = regularization

    # Full solve
    a_sc_full = np.linalg.solve(Mn.T @ Mn + lam * np.eye(len(mode_names)), Mn.T @ s0)
    a_full    = a_sc_full / sc
    full_amp  = {m: float(a_full[i]) for i, m in enumerate(mode_names)}

    loo_amps    = {}
    instability = {m: 0.0 for m in mode_names}

    for k, removed in enumerate(mode_names):
        remaining = [m for j, m in enumerate(mode_names) if j != k]
        if not remaining:
            continue

        M_loo  = np.column_stack([modal_waveforms[m] for m in remaining])
        M0_loo = M_loo - np.mean(M_loo, axis=0)
        sc_loo = np.linalg.norm(M0_loo, axis=0); sc_loo[sc_loo == 0] = 1.0
        Mn_loo = M0_loo / sc_loo
        a_sc_loo = np.linalg.solve(
            Mn_loo.T @ Mn_loo + lam * np.eye(len(remaining)),
            Mn_loo.T @ s0)
        a_loo = a_sc_loo / sc_loo
        loo_dict = {m: float(a_loo[j]) for j, m in enumerate(remaining)}
        loo_amps[removed] = loo_dict

        # Instability of remaining modes
        for m in remaining:
            ref = abs(full_amp[m])
            if ref < 1e-12:
                continue
            delta = abs(loo_dict[m] - full_amp[m]) / ref
            instability[m] = max(instability[m], delta)

    reliable = {m: instability[m] < 0.10 for m in mode_names}
    return {"full_amplitudes": full_amp, "loo_amplitudes": loo_amps,
            "instability": instability, "reliable": reliable}


# ─────────────────────────────────────────────────────────────────────────────
#  4. BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_confidence(s, modal_waveforms, mode_names,
                         regularization=1e-5, n_bootstrap=200, noise_fraction=0.05):
    """
    Estimate amplitude uncertainty by solving with perturbed signals.

    For each bootstrap run:
      1. Add Gaussian noise with std = noise_fraction * std(s)
      2. Solve the system
      3. Record all amplitudes

    Returns mean and std of each amplitude across runs.
    The 95% confidence interval is approximately mean ± 2*std.

    noise_fraction : fraction of signal std used as noise std
                     0.05 = 5% noise (typical for low-SNR experiments)
                     0.01 = 1% noise (clean data)
                     0.10 = 10% noise (noisy data)
    """
    M_raw  = np.column_stack([modal_waveforms[m] for m in mode_names])
    s0_base = s - np.mean(s)
    M0     = M_raw - np.mean(M_raw, axis=0)
    sc     = np.linalg.norm(M0, axis=0); sc[sc == 0] = 1.0
    Mn     = M0 / sc
    lam    = regularization
    MtM    = Mn.T @ Mn + lam * np.eye(len(mode_names))
    noise_std = noise_fraction * np.std(s)

    all_amps = np.zeros((n_bootstrap, len(mode_names)))
    rng = np.random.default_rng(42)

    for i in range(n_bootstrap):
        noise   = rng.normal(0, noise_std, size=len(s))
        s_pert  = (s + noise) - np.mean(s + noise)
        a_sc    = np.linalg.solve(MtM, Mn.T @ s_pert)
        all_amps[i] = a_sc / sc

    mean_amp = np.mean(all_amps, axis=0)
    std_amp  = np.std(all_amps, axis=0)
    cv       = np.abs(std_amp / (mean_amp + 1e-30))  # coefficient of variation

    return {
        "mean":  {m: float(mean_amp[i]) for i, m in enumerate(mode_names)},
        "std":   {m: float(std_amp[i])  for i, m in enumerate(mode_names)},
        "cv":    {m: float(cv[i])       for i, m in enumerate(mode_names)},
        "ci95_lo": {m: float(mean_amp[i] - 2*std_amp[i]) for i, m in enumerate(mode_names)},
        "ci95_hi": {m: float(mean_amp[i] + 2*std_amp[i]) for i, m in enumerate(mode_names)},
        "all_amps": all_amps,
        "reliable": {m: bool(cv[i] < 0.20) for i, m in enumerate(mode_names)},
    }


# ─────────────────────────────────────────────────────────────────────────────
#  COMBINED RELIABILITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def reliability_score(lasso_result, loo_result, bootstrap_result, mode_names):
    """
    Combine all diagnostics into a single reliability assessment per mode.

    A mode is reliable if ALL of the following are true:
      1. LASSO selected it as nonzero (mode is actually present)
      2. LOO instability < 10% (amplitude does not depend on other modes)
      3. Bootstrap CV < 20% (amplitude is stable under noise)

    Score interpretation:
      3/3 checks passed : HIGH confidence
      2/3               : MODERATE confidence
      1/3               : LOW confidence — treat with caution
      0/3               : UNRELIABLE — likely absent or indistinguishable
    """
    print("\n── Reliability Summary ──────────────────────────────────────────────")
    print(f"  {'Mode':<8} {'LASSO':>8} {'LOO stab':>10} {'Bootstrap':>11} "
          f"{'Score':>8} {'Assessment':>14}")
    print("  " + "─"*62)

    scores = {}
    for mode in mode_names:
        lasso_ok = (lasso_result is not None and
                    mode in lasso_result.get("selected_modes", []))
        loo_ok   = (loo_result is not None and
                    loo_result["reliable"].get(mode, False))
        boot_ok  = (bootstrap_result is not None and
                    bootstrap_result["reliable"].get(mode, False))

        score = int(lasso_ok) + int(loo_ok) + int(boot_ok)
        assess = {3: "HIGH",2: "MODERATE",1: "LOW",0: "UNRELIABLE"}[score]

        lasso_str = "present" if lasso_ok else "absent"
        loo_val   = loo_result["instability"].get(mode, np.nan) if loo_result else np.nan
        cv_val    = bootstrap_result["cv"].get(mode, np.nan) if bootstrap_result else np.nan
        loo_str   = f"{loo_val*100:.0f}%" if not np.isnan(loo_val) else "—"
        cv_str    = f"{cv_val*100:.0f}%" if not np.isnan(cv_val) else "—"

        scores[mode] = {"score": score, "assessment": assess,
                        "lasso_present": lasso_ok,
                        "loo_instability": loo_val, "bootstrap_cv": cv_val}
        print(f"  {mode:<8} {lasso_str:>8} {loo_str:>10} {cv_str:>11} "
              f"{score:>8}/3 {assess:>14}")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_lasso_path(lasso_result, mode_names):
    if lasso_result is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Number of modes vs alpha
    axes[0].semilogx(lasso_result["alpha_path"], lasso_result["n_modes_path"],
                     "o-", color="steelblue", lw=1.5, ms=4)
    axes[0].axvline(lasso_result["best_alpha"], color="tomato", lw=2,
                    ls="--", label=f"Best α = {lasso_result['best_alpha']:.2e}")
    axes[0].set_xlabel("Regularisation α")
    axes[0].set_ylabel("Number of nonzero modes")
    axes[0].set_title("LASSO: Mode sparsity vs regularisation")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Residual vs alpha
    axes[1].semilogx(lasso_result["alpha_path"], lasso_result["residual_path"],
                     "o-", color="steelblue", lw=1.5, ms=4)
    axes[1].axvline(lasso_result["best_alpha"], color="tomato", lw=2,
                    ls="--", label=f"Best α = {lasso_result['best_alpha']:.2e}")
    axes[1].set_xlabel("Regularisation α")
    axes[1].set_ylabel("Relative residual")
    axes[1].set_title("LASSO: Residual vs regularisation\n"
                      "(corner = best trade-off between fit and sparsity)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/lasso_path.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/lasso_path.png")
    plt.show()


def plot_condition_sweep(sweep):
    modes  = [s["mode"]      for s in sweep]
    conds  = [s["cond"]      for s in sweep]
    resids = [s["cumul_resid"] for s in sweep]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(range(1, len(modes)+1), conds, "o-",
                     color="steelblue", lw=1.5, ms=6)
    # Mark jumps
    for i in range(1, len(conds)):
        if conds[i] > conds[i-1] * 10:
            axes[0].axvline(i+1, color="tomato", lw=1.5, ls="--", alpha=0.7)
    axes[0].set_xticks(range(1, len(modes)+1))
    axes[0].set_xticklabels(modes, rotation=45, fontsize=8)
    axes[0].set_xlabel("Mode added (in order of signal correlation)")
    axes[0].set_ylabel("Condition number (log scale)")
    axes[0].set_title("Condition number as modes are added\n"
                      "(red dashed = >10x jump, modes beyond are ill-conditioned)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, len(modes)+1), resids, "o-",
                 color="steelblue", lw=1.5, ms=6)
    axes[1].set_xticks(range(1, len(modes)+1))
    axes[1].set_xticklabels(modes, rotation=45, fontsize=8)
    axes[1].set_xlabel("Mode added")
    axes[1].set_ylabel("Cumulative residual")
    axes[1].set_title("Residual as modes are added\n"
                      "(flattening = additional modes not improving fit)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/condition_sweep.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/condition_sweep.png")
    plt.show()


def plot_bootstrap(bootstrap_result, mode_names):
    if bootstrap_result is None:
        return
    means  = [bootstrap_result["mean"][m] for m in mode_names]
    stds   = [bootstrap_result["std"][m]  for m in mode_names]
    colors = ["tomato" if not bootstrap_result["reliable"][m] else "steelblue"
              for m in mode_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(mode_names))
    axes[0].bar(x, np.abs(means), color=colors, edgecolor="white", alpha=0.85)
    axes[0].errorbar(x, np.abs(means), yerr=2*np.array(stds),
                     fmt="none", color="black", capsize=4, lw=1.5,
                     label="95% CI (±2σ)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(mode_names, rotation=45)
    axes[0].set_title("Bootstrap amplitude estimates with 95% confidence intervals\n"
                      "(red = CV > 20%, unreliable)")
    axes[0].set_ylabel("|a_n| (nm)"); axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    cvs = [bootstrap_result["cv"][m]*100 for m in mode_names]
    bc  = ["tomato" if cv > 20 else "steelblue" for cv in cvs]
    axes[1].bar(x, cvs, color=bc, edgecolor="white")
    axes[1].axhline(20, color="tomato", lw=1.5, ls="--", label="20% threshold")
    axes[1].axhline(10, color="gold",   lw=1.5, ls="--", label="10% threshold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(mode_names, rotation=45)
    axes[1].set_title("Coefficient of variation per mode\n"
                      "(lower = more reliable amplitude estimate)")
    axes[1].set_ylabel("CV (%)"); axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("plots/bootstrap_confidence.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/bootstrap_confidence.png")
    plt.show()


def plot_loo_stability(loo_result, mode_names):
    if loo_result is None:
        return
    instab = [loo_result["instability"][m]*100 for m in mode_names]
    colors = ["tomato" if v > 10 else "steelblue" for v in instab]

    plt.figure(figsize=(10, 4))
    x = np.arange(len(mode_names))
    plt.bar(x, instab, color=colors, edgecolor="white")
    plt.axhline(10, color="tomato", lw=1.5, ls="--", label="10% threshold")
    plt.xticks(x, mode_names, rotation=45)
    plt.title("Leave-One-Out amplitude instability per mode\n"
              "(red = amplitude changes >10% when any other mode is removed)")
    plt.ylabel("Max relative amplitude change (%)")
    plt.legend(); plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("plots/loo_stability.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/loo_stability.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_all_diagnostics(s, modal_waveforms, mode_names,
                        regularization=1e-5,
                        n_bootstrap=200,
                        noise_fraction=0.05):
    """
    Run all four reliability diagnostics and print a combined summary.

    Parameters
    ----------
    s               : measured signal
    modal_waveforms : dict of synthesized modal waveforms (from build_dictionary)
    mode_names      : list of mode names to assess
    regularization  : Tikhonov lambda used in main solve
    n_bootstrap     : number of bootstrap iterations (200 is usually enough)
    noise_fraction  : noise level as fraction of signal std (0.05 = 5%)

    Returns
    -------
    scores : dict {mode: {'score': int, 'assessment': str, ...}}
    """
    print("\n" + "="*70)
    print("  RELIABILITY DIAGNOSTICS")
    print("="*70)

    # 1. LASSO
    print("\n[R1] LASSO sparse mode selection...")
    lasso = lasso_mode_selection(s, modal_waveforms, mode_names)
    if lasso:
        print(f"  LASSO selected modes: {lasso['selected_modes']}")
        print(f"  Best alpha: {lasso['best_alpha']:.2e}")

    # 2. Condition number sweep
    print("\n[R2] Condition number sweep...")
    sweep = condition_number_sweep(s, modal_waveforms, mode_names)
    conds = [step["cond"] for step in sweep]
    # Find first big jump
    reliable_depth = len(conds)
    for i in range(1, len(conds)):
        if conds[i] > conds[i-1] * 10:
            reliable_depth = i
            break
    print(f"  Reliable decomposition depth: {reliable_depth} modes")
    print(f"  (condition number jumps >10x after mode "
          f"'{sweep[reliable_depth-1]['mode']}' if applicable)")

    # 3. Leave-one-out
    print("\n[R3] Leave-one-out stability...")
    loo = leave_one_out_stability(s, modal_waveforms, mode_names, regularization)
    reliable_loo = [m for m in mode_names if loo["reliable"][m]]
    print(f"  Stable modes (instability < 10%): {reliable_loo}")

    # 4. Bootstrap
    print(f"\n[R4] Bootstrap confidence intervals ({n_bootstrap} runs, "
          f"noise={noise_fraction*100:.0f}% of signal std)...")
    boot = bootstrap_confidence(s, modal_waveforms, mode_names,
                                regularization, n_bootstrap, noise_fraction)
    reliable_boot = [m for m in mode_names if boot["reliable"][m]]
    print(f"  Reliable modes (CV < 20%): {reliable_boot}")

    # Combined score
    scores = reliability_score(lasso, loo, boot, mode_names)

    # Plots
    print("\n[R5] Plotting diagnostics...")
    plot_lasso_path(lasso, mode_names)
    plot_condition_sweep(sweep)
    plot_bootstrap(boot, mode_names)
    plot_loo_stability(loo, mode_names)

    # Final recommendation
    high_conf = [m for m, v in scores.items() if v["score"] == 3]
    mod_conf  = [m for m, v in scores.items() if v["score"] == 2]
    low_conf  = [m for m, v in scores.items() if v["score"] <= 1]

    print("\n── Final Recommendation ─────────────────────────────────────────────")
    if high_conf:
        print(f"  HIGH confidence  : {high_conf}")
        print(f"    → Amplitudes are reliable. Report these results.")
    if mod_conf:
        print(f"  MODERATE         : {mod_conf}")
        print(f"    → Amplitudes are indicative but cross-check with other methods.")
    if low_conf:
        print(f"  LOW / UNRELIABLE : {low_conf}")
        print(f"    → Do not report amplitudes for these modes.")
        print(f"    → Either the mode is absent or it cannot be separated from others.")
    print()

    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Add this to your existing script after the main solve():
    #
    # from reliability import run_all_diagnostics
    #
    # scores = run_all_diagnostics(
    #     s                = s,
    #     modal_waveforms  = modal_waveforms,
    #     mode_names       = mode_names,
    #     regularization   = C.regularization,
    #     n_bootstrap      = 300,
    #     noise_fraction   = 0.05,   # adjust to your expected SNR
    # )
    #
    # The scores dict tells you which modes to trust.
    print("Import this module and call run_all_diagnostics() from your main script.")
    print("See the usage example in the docstring above.")