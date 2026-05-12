import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.linalg import lstsq


# ─────────────────────────────────────────────────────────────────────────────
#  MODAL RECONSTRUCTION
#
#  After solving:  s(t) = M · a
#
#  The reconstructed contribution of mode n is simply:
#
#    s_n(t) = a_n · m_n(t)
#
#  where m_n(t) is the n-th column of M (the pre-computed modal waveform)
#  and   a_n    is the solved scalar amplitude for that mode.
# ─────────────────────────────────────────────────────────────────────────────


def build_modal_matrix(data, mode_columns):
    M = np.column_stack([data[col].to_numpy() for col in mode_columns])
    mode_names = [col.split(" Propagated")[0] for col in mode_columns]
    return M, mode_names


def solve_amplitudes(M, s, method="tikhonov", regularization=1e-6):
    condition_number = np.linalg.cond(M)

    if method == "lstsq":
        a, _, _, _ = lstsq(M, s)

    elif method == "tikhonov":
        MtM = M.T @ M
        Mts = M.T @ s
        a = np.linalg.solve(MtM + regularization * np.eye(M.shape[1]), Mts)

    elif method == "lasso":
        try:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=regularization, fit_intercept=False, max_iter=10000)
            lasso.fit(M, s)
            a = lasso.coef_
        except ImportError:
            print("sklearn not found, falling back to Tikhonov")
            MtM = M.T @ M
            Mts = M.T @ s
            a = np.linalg.solve(MtM + regularization * np.eye(M.shape[1]), Mts)

    residual = np.linalg.norm(s - M @ a) / (np.linalg.norm(s) + 1e-12)
    return a, residual, condition_number


# ─────────────────────────────────────────────────────────────────────────────
#  CORE: RECONSTRUCT SPECIFIC MODAL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_modes(data, signal_column, mode_columns,
                      modes_of_interest,
                      method="tikhonov", regularization=1e-6):
    """
    Solve the full system using all modes, then reconstruct only the
    modes you care about.

    Parameters
    ----------
    data             : pd.DataFrame — your loaded Excel data
    signal_column    : str  — e.g. "Sum Propagated signal (nm)"
    mode_columns     : list — ALL mode columns (used to build M, improves solve)
    modes_of_interest: list — subset to reconstruct, e.g. ["A2", "A4"]
    method           : str  — "lstsq" | "tikhonov" | "lasso"
    regularization   : float

    Returns
    -------
    results : dict
        reconstructions  — {mode_name: time-domain signal array}
        amplitudes       — {mode_name: scalar amplitude}
        combined         — sum of all modes_of_interest reconstructions
        residual         — relative reconstruction error of full signal
        condition_number
        signal           — original measured signal
        M, a, mode_names — full system outputs
    """

    s = data[signal_column].to_numpy()
    M, mode_names = build_modal_matrix(data, mode_columns)

    # Solve for ALL amplitudes simultaneously
    # (important — including all modes makes the solve more accurate
    #  even if you only care about A2 and A4)
    a, residual, cond = solve_amplitudes(M, s, method=method,
                                          regularization=regularization)

    amplitudes_all = {name: float(amp) for name, amp in zip(mode_names, a)}

    # Measured peak amplitudes — directly from the raw modal columns in Excel
    # This is the "ground truth" peak amplitude of each mode as modelled,
    # before any decomposition. Useful to compare against solved amplitudes.
    measured_peaks = {
        name: float(np.max(np.abs(data[col].to_numpy())))
        for name, col in zip(mode_names, mode_columns)
    }

    # Reconstruct only the modes of interest
    reconstructions = {}
    for mode in modes_of_interest:
        if mode not in mode_names:
            print(f"  ⚠  Mode '{mode}' not found in mode_columns. Skipping.")
            continue
        idx = mode_names.index(mode)
        reconstructions[mode] = a[idx] * M[:, idx]   # a_n · m_n(t)

    # Combined reconstruction of modes of interest
    combined = np.sum(list(reconstructions.values()), axis=0) \
               if reconstructions else np.zeros_like(s)

    return {
        "reconstructions":  reconstructions,   # {mode: waveform}
        "amplitudes":       amplitudes_all,     # solved scalar amplitudes
        "measured_peaks":   measured_peaks,     # peak of raw modal waveform
        "combined":         combined,           # sum of modes of interest
        "residual":         residual,
        "condition_number": cond,
        "signal":           s,
        "M":                M,
        "a":                a,
        "mode_names":       mode_names,
    }


def regularization_sweep(M, s, lambdas=None):
    if lambdas is None:
        lambdas = np.logspace(-10, 2, 60)
    residuals, norms = [], []
    for lam in lambdas:
        a = np.linalg.solve(M.T @ M + lam * np.eye(M.shape[1]), M.T @ s)
        residuals.append(np.linalg.norm(s - M @ a))
        norms.append(np.linalg.norm(a))
    return np.array(lambdas), np.array(residuals), np.array(norms)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "measured":  "steelblue",
    "combined":  "black",
    "A2":        "tomato",
    "A4":        "mediumseagreen",
    "default":   ["orchid", "goldenrod", "darkorange", "teal", "crimson"],
}

def _mode_color(mode, i=0):
    return COLORS.get(mode, COLORS["default"][i % len(COLORS["default"])])


def plot_reconstructions(t, results, modes_of_interest,
                         save=True, name="modal_reconstruction"):
    """
    Three-panel plot:
      1. Original signal + combined reconstruction of modes of interest
      2. Individual reconstructed modal waveforms
      3. Amplitude bar chart (all modes, modes of interest highlighted)
    """
    s               = results["signal"]
    reconstructions = results["reconstructions"]
    combined        = results["combined"]
    amplitudes      = results["amplitudes"]

    os.makedirs("plots", exist_ok=True)

    n_panels = 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 11))

    # ── Panel 1: Measured + combined reconstruction ──────────────────────────
    label_modes = " + ".join(modes_of_interest)
    axes[0].plot(t, s, color=COLORS["measured"], linewidth=1,
                 alpha=0.7, label="Measured signal")
    axes[0].plot(t, combined, color=COLORS["combined"], linewidth=1.5,
                 linestyle="--", label=f"Reconstructed ({label_modes})")
    axes[0].set_title(
        f"Measured vs Reconstructed [{label_modes}]  |  "
        f"residual = {results['residual']:.4f}  |  "
        f"cond = {results['condition_number']:.2e}"
    )
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel("Amplitude (nm)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Panel 2: Individual modal waveforms ──────────────────────────────────
    axes[1].plot(t, s, color=COLORS["measured"], linewidth=0.7,
                 alpha=0.3, label="Measured (ref)")
    for i, (mode, waveform) in enumerate(reconstructions.items()):
        amp = results["amplitudes"][mode]
        axes[1].plot(t, waveform,
                     color=_mode_color(mode, i),
                     linewidth=1.8,
                     label=f"{mode}  (a = {amp:+.4f} nm)")
    axes[1].set_title("Reconstructed Modal Components")
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Amplitude (nm)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: Grouped bar chart — measured peak vs solved amplitude ──────────
    measured_peaks = results.get("measured_peaks", {})
    names      = list(amplitudes.keys())
    n_modes    = len(names)
    x          = np.arange(n_modes)
    bar_width  = 0.38

    solved_vals   = np.abs([amplitudes[n]     for n in names])
    measured_vals = np.abs([measured_peaks.get(n, 0.0) for n in names])

    # Measured bars — grey, with gold edge for modes of interest
    bars_meas = axes[2].bar(
        x - bar_width / 2, measured_vals, bar_width,
        label="Measured peak  |m_n(t)|",
        color="lightsteelblue", edgecolor="white", linewidth=0.8
    )
    # Solved bars — coloured, with gold edge for modes of interest
    bars_solv = axes[2].bar(
        x + bar_width / 2, solved_vals, bar_width,
        label="Solved  |a_n|",
        color=[_mode_color(n, i) if n in modes_of_interest else "steelblue"
               for i, n in enumerate(names)],
        edgecolor="white", linewidth=0.8
    )

    # Gold borders on modes of interest
    for bar_m, bar_s, name in zip(bars_meas, bars_solv, names):
        if name in modes_of_interest:
            for bar in (bar_m, bar_s):
                bar.set_edgecolor("gold")
                bar.set_linewidth(2.5)

    # Value labels on top of each bar
    for bar in list(bars_meas) + list(bars_solv):
        h = bar.get_height()
        if h > 0:
            axes[2].text(
                bar.get_x() + bar.get_width() / 2, h * 1.02,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=7, rotation=45
            )

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names)
    axes[2].set_title("Modal Amplitudes — Measured Peak vs Solved  "
                      "(gold border = modes of interest)")
    axes[2].set_ylabel("|Amplitude| (nm)")
    axes[2].set_xlabel("Mode")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        filepath = os.path.join("plots", f"{name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    plt.show()


def plot_lcurve(lambdas, residuals, norms, save=True, name="lcurve"):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.loglog(residuals, norms, "o-", markersize=4, color="steelblue")
    plt.xlabel("Residual norm  ||s - Ma||₂")
    plt.ylabel("Solution norm  ||a||₂")
    plt.title("L-curve — choose λ at the corner")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if save:
        filepath = os.path.join("plots", f"{name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import preprocess

    # ── Load ─────────────────────────────────────────────────────────────────
    data = preprocess.get_data(
        r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"
    )
    t = data["Propagation time (micsec)"].to_numpy()

    # ── All mode columns available in your Excel ──────────────────────────────
    # Include ALL modes here — even ones you don't care about.
    # The solver needs all of them to correctly attribute energy,
    # otherwise A2/A4 absorb contributions from missing modes.
    mode_columns = [
        "S0 Propagated signal (nm)",
        "A2 Propagated signal (nm)",
        "S4 Propagated signal (nm)",
        "S1 Propagated signal (nm)",
        "S2 Propagated signal (nm)",
        "S5 Propagated signal (nm)",
        "A4 Propagated signal (nm)",
    ]

    # ── Modes you want to reconstruct ────────────────────────────────────────
    modes_of_interest = ["A2", "A4"]

    # ── Optional: L-curve to pick λ ──────────────────────────────────────────
    s = data["Sum Propagated signal (nm)"].to_numpy()
    M, _ = build_modal_matrix(data, mode_columns)
    lambdas, residuals, norms = regularization_sweep(M, s)
    plot_lcurve(lambdas, residuals, norms)

    # ── Run ───────────────────────────────────────────────────────────────────
    results = reconstruct_modes(
        data,
        signal_column    = "Sum Propagated signal (nm)",
        mode_columns     = mode_columns,
        modes_of_interest= modes_of_interest,
        method           = "tikhonov",   # "lstsq" | "tikhonov" | "lasso"
        regularization   = 1e-6,         # tune using L-curve
    )

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n── Modal Amplitudes ──────────────────────────────────────────────")
    print(f"  {'Mode':<6}  {'Solved |a_n| (nm)':>20}  {'Measured peak (nm)':>20}")
    print("  " + "─" * 52)
    for mode in results["mode_names"]:
        amp     = results["amplitudes"][mode]
        meas    = results["measured_peaks"].get(mode, 0.0)
        marker  = "  ◄ target" if mode in modes_of_interest else ""
        print(f"  {mode:<6}  {abs(amp):>20.6f}  {meas:>20.6f}{marker}")

    print(f"\n  Relative residual  : {results['residual']:.4f}")
    print(f"  Condition number   : {results['condition_number']:.3e}")

    if results["condition_number"] > 1e6:
        print("\n  ⚠ High condition number — increase regularization λ")

    print("\n── Reconstructed Mode Waveforms ──────────────────────────")
    for mode, waveform in results["reconstructions"].items():
        print(f"  {mode:<6}  peak = {np.max(np.abs(waveform)):.6f} nm  "
              f"  rms = {np.sqrt(np.mean(waveform**2)):.6f} nm")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_reconstructions(t, results, modes_of_interest)