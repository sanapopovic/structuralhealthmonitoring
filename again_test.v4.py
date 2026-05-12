"""
═══════════════════════════════════════════════════════════════════════════════
  LAMB WAVE MODE DECOMPOSITION  —  v3  (model-based, physically grounded)
  ───────────────────────────────────────────────────────────────────────────
  PROBLEM STATEMENT
  ─────────────────
  Given only the measured sum signal s(t) and the excitation signal e(t),
  recover:
    • Which modes are present
    • The amplitude A_n of each mode
    • The reconstructed time-domain waveform of each mode

  WHY THE PREVIOUS APPROACH FAILED
  ──────────────────────────────────
  The Gaussian-gate / spectrogram-peak method treats modes as separable in
  TIME. They are not. Multiple modes (A0, S0, A2, A7, S4...) arrive at
  almost the same time (~75 μs) because they happen to share the same group
  velocity at the excitation frequency. Gating them together and calling it
  one "R2" packet is physically meaningless and gives wrong amplitudes due
  to constructive/destructive interference inside the gate.

  THE CORRECT APPROACH
  ─────────────────────
  Modes are separable by their DISPERSIVE SHAPE — the frequency-dependent
  phase evolution that is unique to each mode. Two modes can overlap
  completely in time but still be distinguishable because their transfer
  function H_n(f) = M_n(f)/E(f) is different.

  The method is:

    STAGE 1 — BUILD MODAL DICTIONARY
      For each mode n, compute the frequency-domain transfer function:
          H_n(f) = FFT(m_n) / FFT(e)
      This encodes the dispersive propagation of mode n.

      TWO SOURCES for H_n:
        A) SIMULATION MODE (this file, verification):
           H_n extracted directly from the GT mode waveforms in Excel.
           Perfect accuracy — used to verify the decomposition works.

        B) EXPERIMENTAL MODE (real measurements):
           H_n synthesized from dispersion curves:
               H_n(f) = A_excitability(f) * exp(i * k_n(f) * d)
           where k_n(f) = 2π f / c_p_n(f) and d = propagation distance.
           Requires: dispersion curves c_p(f) per mode + excitability model.

    STAGE 2 — SYNTHESIZE MODAL WAVEFORMS
      Given H_n and the measured excitation e(t):
          m_n(t) = IFFT[ FFT(e) · H_n(f) ]
      This produces the predicted time-domain waveform for mode n.
      Note: in experiment, e(t) is replaced by the measured signal itself
      (deconvolved or used directly depending on SNR).

    STAGE 3 — TIKHONOV LEAST SQUARES
      Stack all m_n(t) as columns of modal matrix M and solve:
          s(t) = M · a    →    a = (MᵀM + λI)⁻¹ Mᵀ s
      a_n is the scalar amplitude of mode n.
      The reconstructed contribution of mode n is:  a_n · m_n(t)

    STAGE 4 — VERIFICATION (simulation only)
      Compare recovered amplitudes and waveforms against GT columns.
      In a real experiment this stage is omitted.

  KEY PROPERTIES
  ───────────────
  • Works even when modes overlap completely in time
  • Modes separated by dispersion shape, not arrival time
  • Named output: A0, A1, ... S0, S1, ... (not anonymous R0, R1, ...)
  • Condition number diagnostic tells you when two modes are too similar
    to separate reliably (need more frequency bandwidth or longer signal)
  • Regularisation λ controls stability vs accuracy trade-off

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
from scipy.signal import hilbert
from scipy.linalg import lstsq

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    signal_file     = r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@350mm.xlsx"
    signal_col      = "Sum Propagated signal (nm)"
    excitation_col  = "ExcitationSignal"
    time_col        = "Propagation time (micsec)"

    # ── Operating mode ────────────────────────────────────────────────────
    # "simulation" : use GT waveforms from Excel to build H_n (verification)
    # "experimental": use dispersion curves CSV to build H_n (real experiment)
    mode = "simulation"

    # ── Experimental mode settings (ignored in simulation mode) ──────────
    dispersion_csv  = r"Data/dispersion_curves.csv"   # cols: mode,freq_MHz,cp_mms
    propagation_mm  = 200.0

    # ── Solver ────────────────────────────────────────────────────────────
    regularization  = 1e-6    # Tikhonov λ — increase if cond > 1e6
    excitation_snr  = 0.01    # fraction of max |EXC(f)| below which H_n = 0

    # ── Which modes to highlight in plots ────────────────────────────────
    modes_of_interest = ["A2", "A4"]   # gold border in bar chart


C = Config()


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 0 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load(cfg):
    df  = pd.read_excel(cfg.signal_file)
    t   = df[cfg.time_col].to_numpy()
    s   = df[cfg.signal_col].to_numpy()
    dt  = float(np.mean(np.diff(t)))
    fs  = 1.0 / dt                          # MHz
    exc_raw = df[cfg.excitation_col].to_numpy()
    exc = np.where(np.isnan(exc_raw), 0.0, exc_raw)

    # Ground-truth mode columns (simulation only)
    gt_cols = [c for c in df.columns if "Propagated" in c and "Sum" not in c]
    gt      = {c.split(" Propagated")[0]: df[c].to_numpy() for c in gt_cols}
    gt_cols_map = {c.split(" Propagated")[0]: c for c in gt_cols}

    print(f"  Samples  : {len(t)}   dt = {dt:.5f} μs   fs = {fs:.3f} MHz")
    print(f"  Duration : 0 – {t[-1]:.1f} μs")
    print(f"  GT modes : {sorted(gt.keys())}")
    return t, s, exc, fs, dt, gt, gt_cols_map, df


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1A — BUILD DICTIONARY FROM GT WAVEFORMS (simulation mode)
# ─────────────────────────────────────────────────────────────────────────────

def build_dictionary_simulation(gt, exc, cfg):
    """
    Compute H_n(f) = FFT(m_n) / FFT(e) for each GT mode.

    This is the modal transfer function: it encodes exactly how mode n
    transforms the excitation into the received waveform. Using this to
    synthesize m_n is trivially exact (EXC * H_n = M_n), so it serves as
    a perfect verification that the least-squares stage works correctly.

    In a real experiment you would not have GT waveforms and would instead
    compute H_n from dispersion curves (see build_dictionary_experimental).
    """
    N       = len(exc)
    EXC     = np.fft.rfft(exc, n=N)
    EXC_abs = np.abs(EXC)
    valid   = EXC_abs > cfg.excitation_snr * EXC_abs.max()

    Hn = {}
    for mode, sig in gt.items():
        M_n      = np.fft.rfft(sig, n=N)
        H        = np.where(valid, M_n / (EXC + 1e-30), 0.0 + 0j)
        Hn[mode] = H

    print(f"  Built H_n for {len(Hn)} modes from GT waveforms")
    return Hn, EXC, valid


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1B — BUILD DICTIONARY FROM DISPERSION CURVES (experimental mode)
# ─────────────────────────────────────────────────────────────────────────────

def build_dictionary_experimental(exc, fs, cfg):
    """
    Synthesize H_n(f) from dispersion curves:

        H_n(f) = exp( i · k_n(f) · d )
               = exp( i · 2π f d / c_p_n(f) )

    where:
      f      = frequency (MHz)
      d      = propagation distance (mm)
      c_p_n  = phase velocity of mode n (mm/μs = m/ms)

    This encodes the dispersive phase delay of mode n propagating distance d.
    The excitability (how strongly the source couples into mode n) is assumed
    uniform across the bandwidth — refine this if you have excitability data.

    Requires a CSV with columns: mode, freq_MHz, cp_mms
    """
    from scipy.interpolate import interp1d

    df_disp = pd.read_csv(cfg.dispersion_csv)
    N       = len(exc)
    F       = np.fft.rfftfreq(N, d=1.0/fs)   # MHz
    EXC     = np.fft.rfft(exc, n=N)
    EXC_abs = np.abs(EXC)
    valid   = EXC_abs > cfg.excitation_snr * EXC_abs.max()
    d       = cfg.propagation_mm

    Hn = {}
    for mode in df_disp["mode"].unique():
        curve  = df_disp[df_disp["mode"] == mode].sort_values("freq_MHz")
        freq_c = curve["freq_MHz"].to_numpy()
        cp_c   = curve["cp_mms"].to_numpy()
        ok     = (freq_c > 0.05) & (cp_c > 0.05) & (cp_c < 25.0)
        if ok.sum() < 5:
            continue
        interp_cp = interp1d(freq_c[ok], cp_c[ok], kind="linear",
                             bounds_error=False, fill_value=np.nan)
        cp_F  = interp_cp(F)
        phi   = np.where(
            ~np.isnan(cp_F) & (cp_F > 0.05) & (F >= freq_c[ok].min()),
            2.0 * np.pi * F * d / cp_F,
            0.0
        )
        H = np.where(phi != 0, np.exp(1j * phi), 0.0 + 0j)
        H[~valid] = 0.0
        Hn[mode] = H

    print(f"  Built H_n for {len(Hn)} modes from dispersion curves")
    return Hn, EXC, valid


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — SYNTHESIZE MODAL WAVEFORMS
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_waveforms(Hn, EXC, N):
    """
    m_n(t) = IFFT[ EXC(f) · H_n(f) ]

    Each synthesized waveform is the predicted time-domain signal for mode n
    given the actual excitation. This is the column of the modal matrix M.

    In simulation mode this reconstructs the GT waveform almost perfectly.
    In experimental mode this predicts what mode n would look like at the
    receiver, given the excitation spectrum and the dispersion curve.
    """
    modal_waveforms = {}
    for mode, H in Hn.items():
        m = np.fft.irfft(EXC * H, n=N)
        modal_waveforms[mode] = m
    return modal_waveforms


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — TIKHONOV LEAST SQUARES
# ─────────────────────────────────────────────────────────────────────────────

def solve(s, modal_waveforms, regularization):
    """
    Solve  s ≈ M · a  with Tikhonov regularisation:

        a = (MᵀM + λI)⁻¹ Mᵀ s

    The modal matrix M has one column per mode. Each column is the
    synthesized waveform m_n(t). The amplitude a_n is the scalar weight
    that best explains mode n's contribution to the measured signal.

    Because M encodes dispersive shape (not just arrival time), modes that
    arrive at the same time but have different dispersion are distinguishable
    — their columns in M are different even if their envelopes overlap.

    Condition number interpretation:
      cond < 1e3   : well-conditioned, amplitudes are reliable
      1e3–1e6      : moderate ill-conditioning, results usable with λ tuning
      > 1e6        : severe ill-conditioning — two modes are nearly identical
                     in shape over this bandwidth; increase λ or accept that
                     those two modes cannot be separated with this data.
    """
    mode_names = list(modal_waveforms.keys())
    M          = np.column_stack([modal_waveforms[m] for m in mode_names])
    cond       = np.linalg.cond(M)
    MtM        = M.T @ M
    Mts        = M.T @ s
    a          = np.linalg.solve(MtM + regularization * np.eye(M.shape[1]), Mts)
    resid      = np.linalg.norm(s - M @ a) / (np.linalg.norm(s) + 1e-12)

    amplitudes = {name: float(amp) for name, amp in zip(mode_names, a)}
    return amplitudes, M, a, cond, resid


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 — VERIFICATION (simulation only)
# ─────────────────────────────────────────────────────────────────────────────

def verify(t, amplitudes, modal_waveforms, gt):
    """
    Compare recovered amplitudes and waveforms against GT.

    For each mode: report
      • GT peak amplitude (max of envelope of GT waveform)
      • Recovered peak  = |a_n| × max|m_n(t)|
      • Correlation between recovered contribution a_n·m_n and GT waveform
      • Error %
    """
    print("\n── Verification against ground truth ────────────────────────────────")
    print(f"  {'Mode':<5}  {'GT peak':>9}  {'Rec peak':>9}  {'Err%':>7}  {'Corr':>7}  {'a_n':>10}")
    print("  " + "─" * 55)

    results = {}
    for mode in sorted(amplitudes.keys()):
        if mode not in gt:
            continue
        gt_sig   = gt[mode]
        a_n      = amplitudes[mode]
        m_n      = modal_waveforms[mode]
        rec_sig  = a_n * m_n
        gt_peak  = float(np.max(np.abs(hilbert(gt_sig))))
        rec_peak = float(np.max(np.abs(hilbert(rec_sig))))
        n        = min(len(gt_sig), len(rec_sig))
        try:
            corr = float(np.corrcoef(gt_sig[:n], rec_sig[:n])[0, 1])
        except Exception:
            corr = 0.0
        err_pct  = abs(rec_peak - gt_peak) / (gt_peak + 1e-12) * 100

        results[mode] = dict(gt_peak=gt_peak, rec_peak=rec_peak,
                              corr=corr, err_pct=err_pct, a_n=a_n)
        flag = " ◄" if mode in C.modes_of_interest else ""
        print(f"  {mode:<5}  {gt_peak:>9.5f}  {rec_peak:>9.5f}  "
              f"{err_pct:>6.1f}%  {corr:>7.4f}  {a_n:>10.6f}{flag}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = plt.cm.tab20(np.linspace(0, 1, 20))

def _color(mode, i):
    special = {"A0": PALETTE[0], "S0": PALETTE[1],
               "A1": PALETTE[2], "S1": PALETTE[3],
               "A2": PALETTE[4], "S2": PALETTE[5],
               "A4": PALETTE[6], "S4": PALETTE[7]}
    return special.get(mode, PALETTE[i % 20])


def plot_overview(t, s, amplitudes, modal_waveforms, resid, cond, cfg):
    """
    3-panel overview:
      1. Measured signal + full reconstruction
      2. All individual modal contributions stacked
      3. Amplitude bar chart with modes_of_interest highlighted
    """
    mode_names = list(amplitudes.keys())
    M   = np.column_stack([modal_waveforms[m] for m in mode_names])
    a   = np.array([amplitudes[m] for m in mode_names])
    rec = M @ a

    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    # Panel 1: measured vs reconstructed
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, s,   color="steelblue", lw=1, alpha=0.8, label="Measured s(t)")
    ax0.plot(t, rec, color="tomato",    lw=1.3, ls="--",
             label=f"Reconstruction  (residual = {resid:.5f})")
    ax0.set_title(f"Measured vs Full Reconstruction  |  cond(M) = {cond:.2e}",
                  fontsize=11)
    ax0.set_xlabel("Time (μs)"); ax0.set_ylabel("Amplitude (nm)")
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)

    # Panel 2: individual contributions
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, s, color="black", lw=0.6, alpha=0.25, label="Measured (ref)")
    for i, mode in enumerate(mode_names):
        contrib = amplitudes[mode] * modal_waveforms[mode]
        lw = 2.0 if mode in cfg.modes_of_interest else 0.9
        ax1.plot(t, contrib, color=_color(mode, i), lw=lw, label=mode)
    ax1.set_title("Individual Modal Contributions", fontsize=11)
    ax1.set_xlabel("Time (μs)"); ax1.set_ylabel("Amplitude (nm)")
    ax1.legend(ncol=6, fontsize=7); ax1.grid(True, alpha=0.3)

    # Panel 3: amplitude bar chart
    ax2    = fig.add_subplot(gs[2])
    names  = sorted(mode_names)
    vals   = [abs(amplitudes[m]) for m in names]
    colors = [_color(m, i) for i, m in enumerate(names)]
    bars   = ax2.bar(names, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, name in zip(bars, names):
        if name in cfg.modes_of_interest:
            bar.set_edgecolor("gold"); bar.set_linewidth(3)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x()+bar.get_width()/2, v*1.02,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=45)
    ax2.set_title("Recovered Modal Amplitudes  (gold = modes of interest)", fontsize=11)
    ax2.set_ylabel("|a_n|"); ax2.set_xlabel("Mode")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.savefig("plots/decomposition_overview.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/decomposition_overview.png")
    plt.show()


def plot_modes_of_interest(t, s, amplitudes, modal_waveforms, cfg):
    """
    Focused plot on the modes of interest:
    Panel per mode showing: measured (ref) + reconstructed modal contribution.
    """
    moi = [m for m in cfg.modes_of_interest if m in amplitudes]
    if not moi:
        return

    fig, axes = plt.subplots(len(moi) + 1, 1,
                             figsize=(14, 3.5 * (len(moi) + 1)))

    # Combined reconstruction of modes of interest
    combined = sum(amplitudes[m] * modal_waveforms[m] for m in moi)
    axes[0].plot(t, s,        color="steelblue", lw=1, alpha=0.7, label="Measured")
    axes[0].plot(t, combined, color="black",     lw=1.5, ls="--",
                 label="Sum of modes of interest: " + " + ".join(moi))
    axes[0].set_title("Measured vs Combined Modes of Interest")
    axes[0].set_xlabel("Time (μs)"); axes[0].set_ylabel("nm")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for i, mode in enumerate(moi):
        ax   = axes[i + 1]
        contrib = amplitudes[mode] * modal_waveforms[mode]
        env     = np.abs(hilbert(contrib))
        ax.plot(t, s,       color="steelblue", lw=0.7, alpha=0.25, label="Measured (ref)")
        ax.plot(t, contrib, color=_color(mode, i), lw=1.5,
                label=f"{mode}  a = {amplitudes[mode]:+.5f}")
        ax.plot(t, env,     color=_color(mode, i), lw=1, ls=":",
                alpha=0.7, label="Envelope")
        ax.axhline(np.max(env), color=_color(mode, i), lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"Mode {mode}  |  peak = {np.max(env):.5f} nm")
        ax.set_xlabel("Time (μs)"); ax.set_ylabel("nm")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/modes_of_interest.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/modes_of_interest.png")
    plt.show()


def plot_verification(t, amplitudes, modal_waveforms, gt, verify_results):
    """
    Grouped bar chart: GT peak amplitude vs recovered peak amplitude per mode.
    + Waveform comparison for worst-case modes.
    """
    modes = sorted(verify_results.keys())
    gt_v  = [verify_results[m]["gt_peak"]  for m in modes]
    rc_v  = [verify_results[m]["rec_peak"] for m in modes]
    err_v = [verify_results[m]["err_pct"]  for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Grouped bar
    x = np.arange(len(modes)); w = 0.38
    axes[0].bar(x - w/2, gt_v, w, label="GT peak amplitude",
                color="steelblue", alpha=0.85)
    axes[0].bar(x + w/2, rc_v, w, label="Recovered amplitude",
                color="tomato", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(modes, rotation=45, fontsize=8)
    axes[0].set_title("GT peak vs Recovered amplitude"); axes[0].set_ylabel("nm")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")

    # Error % bar
    bar_colors = ["tomato" if e > 10 else "steelblue" for e in err_v]
    axes[1].bar(modes, err_v, color=bar_colors, edgecolor="white")
    axes[1].axhline(5,  color="gold",  lw=1.2, ls="--", label="5%  threshold")
    axes[1].axhline(10, color="tomato",lw=1.2, ls="--", label="10% threshold")
    axes[1].set_title("Amplitude Recovery Error %")
    axes[1].set_ylabel("Error (%)"); axes[1].set_xlabel("Mode")
    axes[1].set_xticklabels(modes, rotation=45, fontsize=8)
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("plots/verification.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/verification.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "═"*70)
    print("  LAMB WAVE MODE DECOMPOSITION")
    print(f"  Operating mode : {C.mode.upper()}")
    print("═"*70)

    # ── Stage 0: Load ─────────────────────────────────────────────────────
    print("\n[0] Loading data...")
    t, s, exc, fs, dt, gt, gt_cols_map, df = load(C)
    N = len(t)

    # ── Stage 1: Build modal dictionary ───────────────────────────────────
    print(f"\n[1] Building modal dictionary ({C.mode} mode)...")
    if C.mode == "simulation":
        Hn, EXC, valid_f = build_dictionary_simulation(gt, exc, C)
    elif C.mode == "experimental":
        Hn, EXC, valid_f = build_dictionary_experimental(exc, fs, C)
    else:
        raise ValueError(f"Unknown mode: {C.mode}")

    # ── Stage 2: Synthesize modal waveforms ───────────────────────────────
    print("\n[2] Synthesizing modal waveforms...")
    modal_waveforms = synthesize_waveforms(Hn, EXC, N)

    # Quick synthesis quality check
    print("  Synthesis correlation vs GT:")
    for mode in sorted(modal_waveforms.keys()):
        if mode in gt:
            mw  = modal_waveforms[mode]
            g   = gt[mode]
            n   = min(len(mw), len(g))
            try:
                c = float(np.corrcoef(mw[:n], g[:n])[0, 1])
            except Exception:
                c = 0.0
            flag = " ◄ MODE OF INTEREST" if mode in C.modes_of_interest else ""
            print(f"    {mode:<5}: corr = {c:.6f}{flag}")

    # ── Stage 3: Solve for amplitudes ─────────────────────────────────────
    print(f"\n[3] Solving for amplitudes (λ = {C.regularization:.1e})...")
    amplitudes, M, a_vec, cond, resid = solve(s, modal_waveforms, C.regularization)

    print(f"  Condition number : {cond:.3e}")
    print(f"  Relative residual: {resid:.6f}")
    if cond > 1e6:
        print("  ⚠  High condition number — consider increasing C.regularization")
    if resid > 0.05:
        print("  ⚠  Residual > 5% — some modes may be missing from the dictionary")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n── Modal Amplitudes ─────────────────────────────────────────────────")
    print(f"  {'Mode':<6}  {'a_n':>10}  {'peak (nm)':>12}  {'arrival (μs)':>14}")
    print("  " + "─"*48)
    for mode in sorted(amplitudes.keys()):
        a_n      = amplitudes[mode]
        mw       = modal_waveforms[mode]
        peak_nm  = float(np.max(np.abs(hilbert(a_n * mw))))
        t_arr    = float(t[np.argmax(np.abs(hilbert(a_n * mw)))])
        flag     = "  ◄" if mode in C.modes_of_interest else ""
        print(f"  {mode:<6}  {a_n:>10.6f}  {peak_nm:>12.6f}  {t_arr:>14.2f}{flag}")

    # ── Stage 4: Verification (simulation only) ───────────────────────────
    verify_results = None
    if C.mode == "simulation" and gt:
        print("\n[4] Verifying against ground truth...")
        verify_results = verify(t, amplitudes, modal_waveforms, gt)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[5] Plotting...")
    plot_overview(t, s, amplitudes, modal_waveforms, resid, cond, C)
    plot_modes_of_interest(t, s, amplitudes, modal_waveforms, C)
    if verify_results:
        plot_verification(t, amplitudes, modal_waveforms, gt, verify_results)

    print("\n" + "═"*70)
    print("  DONE")
    print("═"*70)