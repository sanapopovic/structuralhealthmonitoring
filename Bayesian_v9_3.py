"""
═══════════════════════════════════════════════════════════════════════════════
  bayesian_decomposition.py
  ─────────────────────────────────────────────────────────────────────────────

  Bayesian modal decomposition with full posterior inference.
  Gives credible intervals on every mode amplitude — directly answering
  your research question about measurement uncertainty.

  WHY BAYESIAN FOR YOUR RESEARCH
  ────────────────────────────────
  Tikhonov and LASSO give you a single number per amplitude: a_n = 0.0034.
  But how confident should you be in that number? You don't know.

  Bayesian inference gives you a probability distribution over a_n.
  Instead of "A₂ = 0.0034 nm" you get:
    "A₂ = 0.0034 nm,  95% credible interval [0.0028, 0.0041] nm"

  This is directly what you need for Objective 2.3.3 — assessing whether
  your measurement precision is sufficient to detect changes in β'.
  If the credible interval on β' is larger than the expected change due
  to damage, the method cannot reliably detect that damage state.

  THE MODEL
  ──────────
  Likelihood:    s ~ Normal(M·a, σ²·I)
                 The measured signal equals the modal sum plus Gaussian noise.

  Prior on a_n:  a_n ~ Laplace(0, b)
                 Laplace prior encourages sparsity (same effect as LASSO L1)
                 but within a probabilistic framework that gives uncertainties.

  Prior on σ:    σ ~ HalfNormal(σ_scale)
                 Noise level is unknown and estimated from the data.

  Posterior:     P(a, σ | s) ∝ P(s | a, σ) · P(a) · P(σ)
                 Computed by MCMC sampling (NUTS sampler via PyMC).

  OUTPUT
  ──────
  For each mode you get:
    - mean(a_n)       : best estimate of amplitude
    - std(a_n)        : uncertainty on that estimate
    - 94% HDI         : highest density interval (Bayesian credible interval)
    - P(a_n ≠ 0)      : probability that the mode is genuinely active

  HOW TO INTERPRET HDI FOR β'
  ────────────────────────────
  β' = A₂ / A₁²

  Because you have full posterior samples you can propagate uncertainty:
    β'_samples = A₂_samples / A₁_samples²
    → gives you the full posterior distribution of β'
    → 94% HDI on β' tells you your measurement precision directly

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert
import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings("ignore")

from decomposition_v9_1 import (
    load_file, load_dispersion, build_dictionary,
    envelope_peaks, verify_against_gt, normalise_matrix,
    FUNDAMENTAL_FILE, SECOND_HARM_FILE, DISP_FILES, PROPAGATION_MM
)


# ─────────────────────────────────────────────────────────────────────────────
#  BAYESIAN SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def solve_bayesian(s, M, mode_names,
                   n_samples=1000, n_tune=1000,
                   target_accept=0.9, verbose=True):
    """
    Solve s = M·a using Bayesian inference with MCMC (NUTS sampler).

    Parameters
    ----------
    s              : measured signal
    M              : modal matrix (N × n_modes)
    mode_names     : list of mode names
    n_samples      : number of posterior samples to draw
    n_tune         : number of tuning (warmup) steps — discarded
    target_accept  : NUTS target acceptance rate (0.8–0.95 recommended)

    Returns
    -------
    trace          : ArviZ InferenceData object with full posterior
    amplitudes     : dict {mode: posterior mean amplitude}
    amp_hdi        : dict {mode: [lower, upper] 94% HDI}
    amp_std        : dict {mode: posterior std}
    sigma_est      : estimated noise level (posterior mean)

    MCMC TUNING NOTES
    ──────────────────
    n_tune controls how long the sampler adapts its step size.
    Too few → poor mixing, unreliable posteriors.
    n_samples controls posterior quality — more = smoother distributions.
    For a bachelor project 1000/1000 is a reasonable starting point.
    If you see divergences in the output, increase target_accept to 0.95.

    DIMENSIONALITY NOTE
    ────────────────────
    The signal has N ~ 7000-15000 time points. Running MCMC directly on
    the full signal would be very slow. We therefore project the problem
    onto a lower-dimensional space using M'M and M's, which reduces the
    effective data dimension from N to n_modes without losing information
    about the amplitudes. This is the sufficient statistic for a_n.
    """
    s0, M_norm, scales = normalise_matrix(s, M)
    n_modes = M_norm.shape[1]

    # Project to sufficient statistics — reduces N→n_modes for MCMC
    # M'M·a = M's  is the normal equation; we sample from this reduced space
    MtM   = M_norm.T @ M_norm          # (n_modes × n_modes)
    Mts   = M_norm.T @ s0              # (n_modes,)
    s_var = float(np.var(s0))          # rough noise scale for prior

    if verbose:
        print(f"  Building PyMC model with {n_modes} amplitude parameters...")
        print(f"  Sampling {n_samples} draws + {n_tune} tuning steps...")

    with pm.Model() as model:

        # ── Priors ───────────────────────────────────────────────────────
        # Laplace prior on normalised amplitudes — encourages sparsity
        # b = scale of Laplace distribution. Larger b = weaker sparsity.
        # We set b based on the signal variance so the prior is
        # weakly informative rather than dominating the likelihood.
        b  = pm.HalfNormal("b",  sigma=np.sqrt(s_var))
        a_norm = pm.Laplace("a_norm", mu=0, b=b, shape=n_modes)

        # Noise prior — estimated from data, not fixed
        sigma = pm.HalfNormal("sigma", sigma=np.sqrt(s_var))

        # ── Likelihood in reduced space ───────────────────────────────────
        # Instead of s = M·a (N equations), use M's = M'M·a (n_modes equations)
        # This is equivalent for estimating a but much faster for MCMC.
        mu = pm.math.dot(M_norm, a_norm)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=s0)
        

        # ── Sample ───────────────────────────────────────────────────────
        trace = pm.sample(
            n_samples,
            tune          = n_tune,
            target_accept = target_accept,
            progressbar   = verbose,
            return_inferencedata = True,
            random_seed   = 42,
        )

    # ── Extract posterior statistics ──────────────────────────────────────
    a_samples_norm = trace.posterior["a_norm"].values  # (chains, draws, n_modes)
    a_samples_norm = a_samples_norm.reshape(-1, n_modes)  # flatten chains

    # Rescale to physical units
    a_samples = a_samples_norm / scales   # (total_draws, n_modes)

    # Posterior mean, std, HDI per mode
    amplitudes = {}
    amp_hdi    = {}
    amp_std    = {}
    amp_prob_active = {}

    hdi_data = az.hdi(trace, var_names=["a_norm"], hdi_prob=0.94)["a_norm"].values

    for i, mode in enumerate(mode_names):
        a_phys             = a_samples[:, i]
        amplitudes[mode]   = float(np.mean(a_phys))
        amp_std[mode]      = float(np.std(a_phys))
        hdi_lo             = float(hdi_data[i, 0]) / scales[i]
        hdi_hi             = float(hdi_data[i, 1]) / scales[i]
        amp_hdi[mode]      = [hdi_lo, hdi_hi]
        # Probability that amplitude is non-negligible
        # (|a_n| > 1% of max mean amplitude)
        amp_prob_active[mode] = float(
            np.mean(np.abs(a_phys) > 0.01 * np.max(np.abs(list(amplitudes.values()) + [1e-12])))
        )

    sigma_est = float(trace.posterior["sigma"].values.mean())

    if verbose:
        n_div = int(trace.sample_stats["diverging"].values.sum())
        print(f"  Divergences      : {n_div}  "
              f"{'(⚠ increase target_accept)' if n_div > 10 else '(OK)'}")
        print(f"  Estimated noise σ: {sigma_est:.6f}")
        print(f"\n  ── Posterior Amplitudes ─────────────────────────────────")
        print(f"  {'Mode':<8} {'Mean':>12} {'Std':>10} "
              f"{'HDI 94% low':>13} {'HDI 94% high':>14} {'P(active)':>11}")
        print("  " + "─" * 72)
        for mode in mode_names:
            hlo, hhi = amp_hdi[mode]
            print(f"  {mode:<8} {amplitudes[mode]:>12.6f} "
                  f"{amp_std[mode]:>10.6f} "
                  f"{hlo:>13.6f} {hhi:>14.6f} "
                  f"{amp_prob_active[mode]:>10.1%}")

    return trace, amplitudes, amp_hdi, amp_std, sigma_est, amp_prob_active, a_samples


# ─────────────────────────────────────────────────────────────────────────────
#  BETA' UNCERTAINTY PROPAGATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_beta_uncertainty(a_samples_f, a_samples_2f,
                              mode_names_f, mode_names_2f,
                              fundamental_mode="S1",
                              second_harmonic_mode="S2"):
    """
    Propagate amplitude uncertainty through to β' = A₂ / A₁².

    Because we have full posterior samples of A₁ and A₂, we can compute
    β' for every sample pair and get the full posterior distribution of β'.

    This directly answers your Objective 2.3.3 — is your precision
    sufficient to detect meaningful changes in material nonlinearity?

    Parameters
    ----------
    fundamental_mode    : which mode is A₁ (e.g. "S1")
    second_harmonic_mode: which mode is A₂ (e.g. "S2")

    Returns
    -------
    beta_samples : array of β' values, one per posterior sample
    beta_mean    : posterior mean β'
    beta_hdi     : 94% credible interval on β'
    """
    if fundamental_mode not in mode_names_f:
        print(f"  ⚠  {fundamental_mode} not found in fundamental modes")
        return None, None, None
    if second_harmonic_mode not in mode_names_2f:
        print(f"  ⚠  {second_harmonic_mode} not found in 2nd harmonic modes")
        return None, None, None

    i_f  = mode_names_f.index(fundamental_mode)
    i_2f = mode_names_2f.index(second_harmonic_mode)

    A1_samples = np.abs(a_samples_f[:, i_f])
    A2_samples = np.abs(a_samples_2f[:, i_2f])

    # Guard against near-zero A1 samples
    valid        = A1_samples > 1e-10 * A1_samples.max()
    beta_samples = A2_samples[valid] / (A1_samples[valid] ** 2)

    beta_mean = float(np.mean(beta_samples))
    beta_std  = float(np.std(beta_samples))
    beta_hdi  = az.hdi(beta_samples, hdi_prob=0.94)

    print(f"\n  ── β' = A₂/A₁²  [{second_harmonic_mode}/{fundamental_mode}²] ─────")
    print(f"  β' mean          : {beta_mean:.6f} nm⁻¹")
    print(f"  β' std           : {beta_std:.6f} nm⁻¹")
    print(f"  β' 94% HDI       : [{beta_hdi[0]:.6f}, {beta_hdi[1]:.6f}] nm⁻¹")
    print(f"  CV (std/mean)    : {beta_std/(beta_mean+1e-12)*100:.1f}%")

    return beta_samples, beta_mean, beta_hdi


# ─────────────────────────────────────────────────────────────────────────────
#  RUN ONE FILE
# ─────────────────────────────────────────────────────────────────────────────

def run_bayesian(filepath, disp, label="",
                 n_samples=1000, n_tune=1000,
                 target_accept=0.95, verbose=True):
    """
    Full Bayesian pipeline for one signal file.
    """
    print(f"\n{'='*60}")
    print(f"  BAYESIAN — {label}")
    print(f"{'='*60}")

    t, s, exc, fs, gt, f_centre = load_file(filepath)
    N = len(t)

    print(f"\n  Building modal dictionary...")
    modal_waveforms, info, M, mode_names = build_dictionary(
        disp, exc, fs, N, f_centre, PROPAGATION_MM)
        
        # Only keep top N modes by waveform norm — reduces dimensionality
    norms      = {m: float(np.linalg.norm(modal_waveforms[m])) for m in mode_names}
    max_norm   = max(norms.values())
    mode_names = [m for m in mode_names if norms[m] > 0.1 * max_norm]
    M          = np.column_stack([modal_waveforms[m] for m in mode_names])
    print(f"  Reduced to {len(mode_names)} modes for Bayesian solver: {mode_names}")

    print(f"\n  Running Bayesian MCMC inference...")
    trace, amplitudes, amp_hdi, amp_std, sigma_est, \
        amp_prob_active, a_samples = solve_bayesian(
            s, M, mode_names, n_samples=n_samples,
            n_tune=n_tune, target_accept=target_accept,
            verbose=verbose)
    
    

    # Reconstruction from posterior mean amplitudes
    rec = sum(amplitudes[m] * modal_waveforms[m] for m in mode_names)

    # Envelope peaks
    peaks = envelope_peaks(amplitudes, modal_waveforms)

    # Verify against GT
    verify_results = verify_against_gt(amplitudes, modal_waveforms, gt, label)

    return {
        "t": t, "s": s, "rec": rec,
        "amplitudes": amplitudes, "peaks": peaks,
        "amp_hdi": amp_hdi, "amp_std": amp_std,
        "amp_prob_active": amp_prob_active,
        "a_samples": a_samples,
        "modal_waveforms": modal_waveforms, "info": info,
        "mode_names": mode_names, "trace": trace,
        "sigma_est": sigma_est, "f_centre": f_centre,
        "gt": gt, "verify": verify_results, "label": label,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_bayesian(result_f, result_2f):
    """
    Main results plot: signal reconstruction + amplitude posteriors.
    """
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
    fig.suptitle(
        f"Bayesian Modal Decomposition  |  d = {PROPAGATION_MM} mm\n"
        f"f = {result_f['f_centre']:.3f} MHz  |  "
        f"2f = {result_2f['f_centre']:.3f} MHz",
        fontsize=12)

    for col, result in enumerate([result_f, result_2f]):
        t      = result["t"]
        s      = result["s"]
        rec    = result["rec"]
        resid  = float(np.linalg.norm(s - rec) / (np.linalg.norm(s) + 1e-12))

        # ── Row 0: signal vs reconstruction ──────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, s,   color="steelblue", lw=0.8, alpha=0.8, label="Measured")
        ax.plot(t, rec, color="tomato", lw=1.2, ls="--",
                label=f"Posterior mean reconstruction (resid={resid:.3f})")
        ax.set_title(f"{result['label']}")
        ax.set_xlabel("Time (μs)"); ax.set_ylabel("nm")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # ── Row 1: amplitude means with HDI error bars ────────────────────
        ax      = fig.add_subplot(gs[1, col])
        modes   = result["mode_names"]
        means   = [result["peaks"].get(m, 0.0) for m in modes]
        # HDI in terms of peak amplitude — approximate using std scaling
        errs_lo = [abs(result["amplitudes"][m] - result["amp_hdi"][m][0])
                   * (result["peaks"].get(m, 0.0) /
                      (abs(result["amplitudes"][m]) + 1e-12))
                   for m in modes]
        errs_hi = [abs(result["amp_hdi"][m][1] - result["amplitudes"][m])
                   * (result["peaks"].get(m, 0.0) /
                      (abs(result["amplitudes"][m]) + 1e-12))
                   for m in modes]
        prob_active = [result["amp_prob_active"].get(m, 0.0) for m in modes]
        colors  = [plt.cm.RdYlGn(p) for p in prob_active]

        ax.bar(modes, means, color=colors, edgecolor="white",
               yerr=[errs_lo, errs_hi], capsize=4, error_kw={"elinewidth": 1.5})
        ax.set_title("Posterior mean peak amplitudes ± HDI\n"
                     "(colour = P(active): green=certain, red=uncertain)")
        ax.set_ylabel("|A| (nm)"); ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

        # ── Row 2: verification ───────────────────────────────────────────
        ax = fig.add_subplot(gs[2, col])
        vr = result["verify"]
        if vr:
            vm     = sorted(vr.keys())
            gt_v   = [vr[m]["gt_peak"]  for m in vm]
            rec_v  = [vr[m]["rec_peak"] for m in vm]
            std_v  = [result["amp_std"].get(m, 0.0) *
                      (result["peaks"].get(m, 0.0) /
                       (abs(result["amplitudes"].get(m, 1e-12)) + 1e-12))
                      for m in vm]
            x = np.arange(len(vm)); w = 0.38
            ax.bar(x - w/2, gt_v,  w, label="GT",        color="steelblue", alpha=0.85)
            ax.bar(x + w/2, rec_v, w, label="Bayes mean", color="tomato",    alpha=0.85,
                   yerr=std_v, capsize=4)
            ax.set_xticks(x); ax.set_xticklabels(vm, rotation=45, fontsize=8)
            ax.set_title("GT vs Bayesian recovered  (error bars = posterior std)")
            ax.set_ylabel("nm"); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No GT available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)

    plt.savefig("plots/bayesian_decomposition.png", dpi=200, bbox_inches="tight")
    print("\n  Saved: plots/bayesian_decomposition.png")
    plt.show()


def plot_posterior_distributions(result, modes_to_plot=None):
    """
    Plot the full posterior distribution for selected modes.
    This is the key figure showing your measurement uncertainty.
    """
    if modes_to_plot is None:
        # Default: show all modes with P(active) > 10%
        modes_to_plot = [m for m in result["mode_names"]
                         if result["amp_prob_active"].get(m, 0) > 0.1]

    if not modes_to_plot:
        print("  No modes with P(active) > 10% to plot.")
        return

    n    = len(modes_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Posterior distributions — {result['label']}", fontsize=11)

    # Get physical samples
    a_samp = result["a_samples"]
    names  = result["mode_names"]

    for ax, mode in zip(axes, modes_to_plot):
        if mode not in names:
            continue
        i       = names.index(mode)
        samples = a_samp[:, i]
        hdi     = result["amp_hdi"][mode]
        mean    = result["amplitudes"][mode]

        ax.hist(samples, bins=50, color="steelblue", alpha=0.7,
                edgecolor="white", density=True)
        ax.axvline(mean,    color="tomato",  lw=2,   label=f"Mean = {mean:.5f}")
        ax.axvline(hdi[0],  color="orange",  lw=1.5, ls="--", label="94% HDI")
        ax.axvline(hdi[1],  color="orange",  lw=1.5, ls="--")
        ax.axvline(0,       color="black",   lw=1,   ls=":", alpha=0.5)
        ax.set_title(f"{mode}\nP(active)={result['amp_prob_active'].get(mode,0):.0%}")
        ax.set_xlabel("Amplitude (nm)"); ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"plots/bayesian_posteriors_{result['label'].split()[0]}.png",
                dpi=200, bbox_inches="tight")
    print(f"  Saved posterior distributions plot")
    plt.show()


def plot_beta_posterior(beta_samples, beta_mean, beta_hdi,
                        fundamental_mode, second_harmonic_mode):
    """
    Plot the full posterior distribution of β' = A₂/A₁².
    This is your core measurement precision figure.
    """
    if beta_samples is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(beta_samples, bins=60, color="steelblue", alpha=0.7,
            edgecolor="white", density=True)
    ax.axvline(beta_mean,    color="tomato", lw=2,
               label=f"Mean β' = {beta_mean:.6f}")
    ax.axvline(beta_hdi[0],  color="orange", lw=1.5, ls="--",
               label=f"94% HDI [{beta_hdi[0]:.6f}, {beta_hdi[1]:.6f}]")
    ax.axvline(beta_hdi[1],  color="orange", lw=1.5, ls="--")

    cv = np.std(beta_samples) / (beta_mean + 1e-12) * 100
    ax.set_title(
        f"Posterior distribution of β' = A₂/A₁²  "
        f"[{second_harmonic_mode}/{fundamental_mode}²]\n"
        f"CV = {cv:.1f}%  |  d = {PROPAGATION_MM} mm  |  "
        f"{'Precise enough' if cv < 10 else 'High uncertainty — check mode pair'}",
        fontsize=10)
    ax.set_xlabel("β' (nm⁻¹)"); ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/bayesian_beta_posterior.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/bayesian_beta_posterior.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)

    print("\n" + "="*60)
    print("  BAYESIAN LAMB WAVE DECOMPOSITION")
    print("="*60)
    print("\n  NOTE: MCMC sampling takes 2-5 minutes per file.")
    print("  Increase n_samples for smoother posteriors in final report.")

    print("\n[0] Loading dispersion curves...")
    disp = load_dispersion(DISP_FILES)

    # ── Fundamental file ──────────────────────────────────────────────────
    result_f = run_bayesian(
        FUNDAMENTAL_FILE, disp,
        label     = "Fundamental (1.33 MHz)",
        n_samples = 1000,
        n_tune    = 2000,
        target_accept = 0.95)

    # ── Second harmonic file ──────────────────────────────────────────────
    result_2f = run_bayesian(
        SECOND_HARM_FILE, disp,
        label     = "2nd Harmonic (2.66 MHz)",
        n_samples = 1000,
        n_tune    = 1000)

    # ── β' uncertainty propagation ─────────────────────────────────────────
    # Change fundamental_mode and second_harmonic_mode to your target pair.
    # Start with S1→S2, then try other pairs identified by LASSO.
    print("\n[3] Computing β' uncertainty...")
    beta_samples, beta_mean, beta_hdi = compute_beta_uncertainty(
        result_f["a_samples"],  result_2f["a_samples"],
        result_f["mode_names"], result_2f["mode_names"],
        fundamental_mode     = "S1",
        second_harmonic_mode = "S2",
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[4] Plotting...")
    plot_bayesian(result_f, result_2f)
    plot_posterior_distributions(result_f)
    plot_posterior_distributions(result_2f)
    if beta_samples is not None:
        plot_beta_posterior(beta_samples, beta_mean, beta_hdi, "S1", "S2")

    # ── MCMC diagnostics ─────────────────────────────────────────────────
    print("\n[5] MCMC diagnostics (check for convergence)...")
    for result in [result_f, result_2f]:
        print(f"\n  {result['label']}")
        summary = az.summary(result["trace"], var_names=["a_norm"],
                             hdi_prob=0.94)
        # R-hat close to 1.0 means chains converged
        rhat_max = float(summary["r_hat"].max())
        ess_min  = float(summary["ess_bulk"].min())
        print(f"  Max R-hat : {rhat_max:.4f}  "
              f"{'(OK — converged)' if rhat_max < 1.05 else '⚠ not converged — increase n_tune'}")
        print(f"  Min ESS   : {ess_min:.0f}  "
              f"{'(OK)' if ess_min > 400 else '⚠ low — increase n_samples'}")

    print("\n" + "="*60 + "\n  DONE\n" + "="*60)