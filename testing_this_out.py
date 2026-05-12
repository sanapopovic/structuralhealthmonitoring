import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.linalg import lstsq


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL-BASED LEAST SQUARES LAMB WAVE MODE DECOMPOSITION
#
#  Core idea:
#    s(t) = M · a
#
#  where:
#    s(t)  = measured signal         [Nt × 1]
#    M     = modal matrix            [Nt × N_modes]  — one column per mode
#    a     = amplitude vector        [N_modes × 1]   — what we solve for
#
#  Your Excel file already contains the individual propagated mode signals,
#  which ARE the columns of M. This is the key insight — no need to synthesize
#  them from scratch.
# ─────────────────────────────────────────────────────────────────────────────


def build_modal_matrix(data, mode_columns):
    """
    Build the modal matrix M from the pre-computed mode signals in your Excel.

    Parameters
    ----------
    data : pd.DataFrame
        Your loaded Excel dataframe
    mode_columns : list of str
        Column names for each mode, e.g.
        ["S0 Propagated signal (nm)", "A2 Propagated signal (nm)", ...]

    Returns
    -------
    M : np.ndarray  [Nt × N_modes]
    mode_names : list of str  (cleaned labels for plotting)
    """
    M = np.column_stack([data[col].to_numpy() for col in mode_columns])
    mode_names = [col.split(" Propagated")[0] for col in mode_columns]
    return M, mode_names


def solve_amplitudes(M, s, method="lstsq", regularization=1e-6):
    """
    Solve for modal amplitudes:  s ≈ M · a

    Parameters
    ----------
    M : np.ndarray [Nt × N_modes]
    s : np.ndarray [Nt]
    method : str
        "lstsq"     — plain least squares (scipy, robust SVD-based)
        "tikhonov"  — Tikhonov / ridge regression (λI regularization)
        "lasso"     — L1 sparse solution (needs sklearn)
    regularization : float
        λ for Tikhonov. Ignored for "lstsq".

    Returns
    -------
    a : np.ndarray [N_modes]  — complex or real amplitudes
    residual : float          — ||s - M·a||₂ / ||s||₂  (relative)
    condition_number : float  — condition number of M (ill-conditioning warning)
    """

    condition_number = np.linalg.cond(M)

    if method == "lstsq":
        a, _, _, _ = lstsq(M, s)

    elif method == "tikhonov":
        # a = (MᴴM + λI)⁻¹ Mᴴ s
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

    else:
        raise ValueError(f"Unknown method: {method}")

    reconstruction = M @ a
    residual = np.linalg.norm(s - reconstruction) / (np.linalg.norm(s) + 1e-12)

    return a, residual, condition_number


def reconstruct_signal(M, a):
    """Reconstruct signal from modal matrix and amplitudes."""
    return M @ a


def decompose(data, signal_column, mode_columns,
              method="tikhonov", regularization=1e-6):
    """
    Full decomposition pipeline.

    Parameters
    ----------
    data : pd.DataFrame
    signal_column : str   e.g. "Sum Propagated signal (nm)"
    mode_columns  : list  e.g. ["S0 Propagated signal (nm)", ...]
    method        : str   "lstsq" | "tikhonov" | "lasso"
    regularization: float

    Returns
    -------
    results : dict with keys:
        amplitudes      — dict {mode_name: amplitude}
        residual        — relative reconstruction error
        condition_number
        reconstruction  — reconstructed time-domain signal
        M               — modal matrix
        a               — raw amplitude array
        mode_names      — list of mode labels
    """

    s = data[signal_column].to_numpy()
    M, mode_names = build_modal_matrix(data, mode_columns)

    a, residual, cond = solve_amplitudes(M, s, method=method,
                                          regularization=regularization)

    reconstruction = reconstruct_signal(M, a)
    amplitudes = {name: float(amp) for name, amp in zip(mode_names, a)}

    return {
        "amplitudes":       amplitudes,
        "residual":         residual,
        "condition_number": cond,
        "reconstruction":   reconstruction,
        "M":                M,
        "a":                a,
        "mode_names":       mode_names,
        "signal":           s,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  REGULARIZATION SWEEP
#  Use this to find the best λ (L-curve / residual vs amplitude norm trade-off)
# ─────────────────────────────────────────────────────────────────────────────

def regularization_sweep(M, s, lambdas=None):
    """
    Sweep over regularization values and return residuals + solution norms.
    Useful for L-curve analysis to pick the best λ.

    Parameters
    ----------
    M       : modal matrix
    s       : measured signal
    lambdas : array of λ values to test

    Returns
    -------
    lambdas, residuals, norms : arrays for plotting the L-curve
    """
    if lambdas is None:
        lambdas = np.logspace(-10, 2, 60)

    residuals = []
    norms = []

    for lam in lambdas:
        MtM = M.T @ M
        Mts = M.T @ s
        a = np.linalg.solve(MtM + lam * np.eye(M.shape[1]), Mts)
        residuals.append(np.linalg.norm(s - M @ a))
        norms.append(np.linalg.norm(a))

    return np.array(lambdas), np.array(residuals), np.array(norms)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_decomposition(t, results, save=True, name="modal_decomposition"):
    """
    Plot:
      1. Measured vs reconstructed signal
      2. Bar chart of modal amplitudes
      3. Individual modal contributions
    """
    s            = results["signal"]
    reconstruction = results["reconstruction"]
    amplitudes   = results["amplitudes"]
    M            = results["M"]
    a            = results["a"]
    mode_names   = results["mode_names"]

    os.makedirs("plots", exist_ok=True)

    # ── 1. Measured vs Reconstructed ────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(t, s, label="Measured", color="steelblue", linewidth=1)
    axes[0].plot(t, reconstruction, label="Reconstructed", color="tomato",
                 linewidth=1, linestyle="--")
    axes[0].set_title(f"Measured vs Reconstructed  "
                      f"(residual = {results['residual']:.4f}, "
                      f"cond = {results['condition_number']:.1e})")
    axes[0].set_xlabel("Time (μs)")
    axes[0].set_ylabel("Amplitude (nm)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── 2. Amplitude bar chart ───────────────────────────────────────────────
    names = list(amplitudes.keys())
    vals  = list(amplitudes.values())
    colors = ["tomato" if v < 0 else "steelblue" for v in vals]
    axes[1].bar(names, np.abs(vals), color=colors)
    axes[1].set_title("Modal Amplitudes |Aₙ|")
    axes[1].set_ylabel("|Amplitude| (nm)")
    axes[1].set_xlabel("Mode")
    axes[1].grid(True, alpha=0.3, axis="y")

    # highlight A1, A2 if present
    for i, name in enumerate(names):
        if name in ("A1", "A2"):
            axes[1].get_children()[i].set_edgecolor("gold")
            axes[1].get_children()[i].set_linewidth(3)

    # ── 3. Individual modal contributions ───────────────────────────────────
    axes[2].plot(t, s, color="black", linewidth=0.8, alpha=0.4, label="Measured")
    for i, (name, amp) in enumerate(amplitudes.items()):
        contribution = a[i] * M[:, i]
        lw = 2.0 if name in ("A1", "A2") else 0.8
        axes[2].plot(t, contribution, label=name, linewidth=lw)
    axes[2].set_title("Individual Modal Contributions")
    axes[2].set_xlabel("Time (μs)")
    axes[2].set_ylabel("Amplitude (nm)")
    axes[2].legend(ncol=4, fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        filepath = os.path.join("plots", f"{name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    plt.show()


def plot_lcurve(lambdas, residuals, norms, save=True, name="lcurve"):
    """Plot the L-curve to choose regularization parameter."""
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
        print(f"Saved: {filepath}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — drop-in usage matching your existing code style
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import preprocess

    # ── Load data (same as your existing scripts) ────────────────────────────
    data = preprocess.get_data(
        r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"
    )

    t = data["Propagation time (micsec)"].to_numpy()

    # ── Define which columns are your modes ─────────────────────────────────
    # Add or remove modes depending on what columns exist in your Excel file
    mode_columns = [
        "S0 Propagated signal (nm)",
        "A2 Propagated signal (nm)",
        "S4 Propagated signal (nm)",
        "S1 Propagated signal (nm)",
        "S2 Propagated signal (nm)",
        "S5 Propagated signal (nm)",
        "A4 Propagated signal (nm)",
    ]

    # ── Optional: L-curve to find best regularization λ ─────────────────────
    s = data["Sum Propagated signal (nm)"].to_numpy()
    M, mode_names = build_modal_matrix(data, mode_columns)

    lambdas, residuals, norms = regularization_sweep(M, s)
    plot_lcurve(lambdas, residuals, norms)

    # ── Run decomposition ────────────────────────────────────────────────────
    # Start with tikhonov — tune regularization based on L-curve
    results = decompose(
        data,
        signal_column="Sum Propagated signal (nm)",
        mode_columns=mode_columns,
        method="tikhonov",      # "lstsq" | "tikhonov" | "lasso"
        regularization=1e-6,    # adjust based on L-curve result
    )

    # ── Print results ────────────────────────────────────────────────────────
    print("\n── Modal Amplitudes ──────────────────────────────")
    for mode, amp in results["amplitudes"].items():
        marker = " ◄" if mode in ("A1", "A2") else ""
        print(f"  {mode:<6}  {amp:+.6f} nm{marker}")

    print(f"\n  Relative residual  : {results['residual']:.4f}")
    print(f"  Condition number   : {results['condition_number']:.3e}")

    if results["condition_number"] > 1e6:
        print("\n  ⚠ High condition number — consider increasing regularization λ")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_decomposition(t, results)