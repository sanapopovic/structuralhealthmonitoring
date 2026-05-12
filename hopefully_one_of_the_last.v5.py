"""
═══════════════════════════════════════════════════════════════════════════════
  LAMB WAVE MODE DECOMPOSITION  —  Final version
  ─────────────────────────────────────────────────────────────────────────────

  PURPOSE
  ───────
  Given the measured sum signal s(t) and the excitation e(t), recover which
  Lamb wave modes are present and what amplitude each carries.

  METHOD
  ──────
  For each mode n, synthesize its predicted waveform m_n(t) using:

      phi_n(f) = k_n(f) * d          [rad]   dispersive phase over distance d
      H_n(f)   = exp(i * phi_n(f))           transfer function
      m_n(t)   = IFFT[ FFT(e) * H_n(f) ]    dispersive waveform

  Stack all m_n into modal matrix M and solve:

      s = M * a    =>    a = (M'M + lambda*I)^-1  M' s

  The amplitude a_n tells you how much mode n contributes. 
  Modes with |a_n| / max|a| below a threshold are flagged as absent.


═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import os, warnings

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  — edit these for your experiment
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # ── Signal ────────────────────────────────────────────────────────────
    signal_file    = r"Data/In-plane_A2_TemporalResponse_15_963MHzmm_200mm_trimmed.xlsx"
    signal_col     = "Sum Propagated signal (nm)"
    excitation_col = "ExcitationSignal"
    time_col       = "Propagation time (micsec)"

    # ── Dispersion curves (6mm plate) ─────────────────────────────────────
    disp_files = {
        "A_Lamb":  r"Data/Dispersion Curve 6mm 3000khz_A_Lamb.xlsx",
        "S_Lamb":  r"Data/Dispersion Curve 6mm 3000khz_S_Lamb.xlsx",
       # "A_Shear": r"Data/Dispersion Curve 6mm_A_Shear.xlsx",
        #"S_Shear": r"Data/Dispersion Curve 6mm_S_Shear.xlsx",
    }

    # ── Physics ───────────────────────────────────────────────────────────
    propagation_mm = 200.0   # source-to-receiver distance (mm)

    # ── Solver ────────────────────────────────────────────────────────────
    regularization   = 1e-4   # Tikhonov lambda — increase if cond(M) > 1e6
    amplitude_thresh = 0.0000   # |a_n|/max|a| below this => mode flagged absent
    excitation_snr   = 0.001   # fraction of max EXC(f) below which H_n = 0

    # ── Output ────────────────────────────────────────────────────────────
    modes_of_interest = [ "A1", 'A2', 'A4', 'A5' "S4"]   # highlighted in plots and summary
    run_verification  = True    # compare against GT columns if present in Excel

    force_modes = ["A0", "A1", "A2", "A3", "A4", "A5", 'A6', ' A7' , 'A8',"A9",
               "S0", 'S1', "S2", "S3", "S4", "S5", 'S6',"S8", "S9"]  # if not None, only these modes are included (must be in dispersion files)


C = Config()
print(f'this is part of C: {C.amplitude_thresh}')
# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 0 — LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_signal(cfg):
    df  = pd.read_excel(cfg.signal_file)
    t   = df[cfg.time_col].to_numpy()
    s   = df[cfg.signal_col].to_numpy()
    dt  = float(np.mean(np.diff(t)))
    fs  = 1.0 / dt
    exc_raw = df[cfg.excitation_col].to_numpy()
    exc = np.where(np.isnan(exc_raw), 0.0, exc_raw)

    gt_cols = [c for c in df.columns if "Propagated" in c and "Sum" not in c]
    gt = {c.split(" Propagated")[0]: df[c].to_numpy() for c in gt_cols}

    print(f"  Samples  : {len(t)}   dt = {dt:.5f} μs   fs = {fs:.3f} MHz")
    print(f"  Duration : 0 – {t[-1]:.1f} μs")
    if gt:
        print(f"  GT modes : {sorted(gt.keys())}")
    return t, s, exc, fs, dt, gt


def load_dispersion(cfg):
    """
    Load all four dispersion curve files.
    Returns dict: mode -> {freq, cp, k, tp, cg}

    Column mapping used:
      'f (MHz)'                    -> frequency axis
      'Phase velocity (m/ms)'      -> cp(f)  [used for phase synthesis]
      'Wavenumber (rad/mm)'        -> k(f)   [used directly: phi = k*d]
      'Propagation time (micsec)'  -> tprop(f) = d/cg(f) [diagnostic]
      'Energy velocity (m/ms)'     -> cg(f)  [group velocity, diagnostic]
    """
    disp = {}
    for label, fpath in cfg.disp_files.items():
        try:
            df = pd.read_excel(fpath)
        except FileNotFoundError:
            print(f"  WARNING: {fpath} not found — skipping.")
            continue
        mode_names = [c.split(" f (MHz)")[0]
                      for c in df.columns if c.endswith(" f (MHz)")]
        for mode in mode_names:
            freq = df[f"{mode} f (MHz)"].to_numpy()
            cp   = df[f"{mode} Phase velocity (m/ms)"].to_numpy()
            k    = df[f"{mode} Wavenumber (rad/mm)"].to_numpy()
            tp   = df[f"{mode} Propagation time (micsec)"].to_numpy()
            cg_col = f"{mode} Energy velocity (m/ms)"
            cg   = df[cg_col].to_numpy() if cg_col in df.columns else np.full_like(freq, np.nan)
            valid = (~np.isnan(freq) & ~np.isnan(cp) &
                     (cp > 0) & (cp < 50) & ~np.isnan(k) & (k >= 0))
            if valid.sum() < 5:
                continue
            disp[mode] = {
                "freq": freq[valid], "cp": cp[valid],
                "k":    k[valid],    "tp": tp[valid],
                "cg":   cg[valid],   "type": label,
            }
    return disp


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — BUILD MODAL DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

def build_dictionary(disp, exc, fs, N, cfg):
    """
    For each mode n:

      1. Interpolate k_n(f) onto the signal frequency grid F
      2. Compute phi_n(f) = k_n(f) * d    [wavenumber × distance = phase in rad]
      3. H_n(f) = exp(i * phi_n(f))        [transfer function, unit amplitude]
      4. m_n(t) = IFFT[ EXC(f) * H_n(f) ] [predicted time-domain waveform]

    WHY WAVENUMBER k DIRECTLY:
      k = 2*pi*f / cp, so phi = k*d = 2*pi*f*d/cp
      Using k directly from the dispersion file is more accurate than
      computing k = 2*pi*f/cp since k is tabulated more densely.

    WHY LIMIT TO EXCITATION BANDWIDTH:
      Outside the band, EXC(f) ~ 0 so those bins carry no signal energy.
      Zeroing H_n there prevents noise amplification.

    COVERAGE REPORT explains which modes exist at your excitation frequency.
    Modes with 0% coverage do not exist physically at that frequency —
    this is a consequence of their cutoff frequency being above the excitation.
    """
    F       = np.fft.rfftfreq(N, d=1.0/fs)
    EXC     = np.fft.rfft(exc, n=N)
    EXC_abs = np.abs(EXC)
    valid_f = EXC_abs > cfg.excitation_snr * EXC_abs.max()

    f_min_exc = float(F[valid_f].min())
    f_max_exc = float(F[valid_f].max())
    f_centre  = float(F[np.argmax(EXC_abs)])

    print(f"  Excitation centre : {f_centre:.3f} MHz")
    print(f"  Excitation band   : {f_min_exc:.3f} – {f_max_exc:.3f} MHz")
    print()

    modal_waveforms = {}
    info = {}

    for mode in sorted(disp.keys()):
        c = disp[mode]
        fv = c["freq"]; kv = c["k"]; tv = c["tp"]
        f_mode_min = float(fv.min())
        f_mode_max = float(fv.max())

        # Band coverage
        ov = max(0.0, min(f_mode_max, f_max_exc) - max(f_mode_min, f_min_exc))
        coverage = ov / (f_max_exc - f_min_exc + 1e-12)
        in_band  = coverage > 0.05

        # Predicted arrival time at excitation centre frequency
        valid_tp = ~np.isnan(tv) & (tv > 5) & (tv < 2000)
        if valid_tp.sum() > 2 and in_band:
            itp = interp1d(fv[valid_tp], tv[valid_tp],
                           bounds_error=False, fill_value=np.nan)
            t_pred = float(itp(f_centre))
        else:
            t_pred = np.nan

        # Synthesize m_n(t)
        ik  = interp1d(fv, kv, kind="linear", bounds_error=False, fill_value=0.0)
        k_F = ik(F)
        phi = np.where(valid_f & (F >= f_mode_min) & (F <= f_mode_max) & (k_F > 0),
               k_F * cfg.propagation_mm, 0.0)
        H   = np.where(phi != 0, np.exp(-1j * phi), 0.0 + 0j)
        m_n = np.fft.irfft(EXC * H, n=N)

        modal_waveforms[mode] = m_n
        info[mode] = {
            "f_min": f_mode_min, "f_max": f_mode_max,
            "coverage": coverage, "in_band": in_band,
            "t_pred": t_pred, "type": c["type"],
        }

    return modal_waveforms, info, EXC, F, valid_f


def print_coverage(info, cfg):
    print("  Mode coverage in excitation band:")
    print(f"  {'Mode':<8} {'Type':<10} {'Freq range (MHz)':<20} "
          f"{'Coverage':>10} {'t_pred (μs)':>12} {'Status':>16}")
    print("  " + "─"*78)
    for mode, v in sorted(info.items()):
        frange  = f"{v['f_min']:.2f}–{v['f_max']:.2f}"
        cov     = f"{v['coverage']*100:.0f}%"
        t_pred  = f"{v['t_pred']:.1f}" if not np.isnan(v['t_pred']) else "—"
        status  = "OK" if v["in_band"] else "below cutoff — absent"
        flag    = " ◄" if mode in cfg.modes_of_interest else ""
        print(f"  {mode:<8} {v['type']:<10} {frange:<20} "
              f"{cov:>10} {t_pred:>12} {status:>16}{flag}")
    n_ok  = sum(1 for v in info.values() if v["in_band"])
    n_tot = len(info)
    print()
    print(f"  {n_ok} of {n_tot} modes exist in the excitation band.")
    if n_ok < n_tot:
        print(f"  {n_tot-n_ok} modes are below their cutoff at this frequency")
        print(f"  and are correctly excluded from the decomposition.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — SOLVE
# ─────────────────────────────────────────────────────────────────────────────
# 
# def solve(s, modal_waveforms, info, cfg):
    # """
    # Build modal matrix M from in-band modes and solve s = M*a.
# 
    # Only modes with dispersion data in the excitation band are included.
    # A mode excluded here is absent from the signal — it does not exist
    # at this frequency for this plate thickness, not just missing from the data.
# 
    # Tikhonov regularisation: a = (M'M + lambda*I)^-1 M' s
# 
    # Condition number diagnostic:
    #   < 1e3  : well-conditioned, amplitudes are reliable
    #   1e3-1e6: moderate — results usable, increase lambda if needed
    #   > 1e6  : ill-conditioned — some modes are too similar to distinguish
    # """
    # valid_modes = sorted(m for m, v in info.items() if v["in_band"])
    # excluded    = sorted(m for m in info if m not in valid_modes)
# 
    # if not valid_modes:
        # print("  ERROR: No modes in excitation band. Check dispersion files.")
        # return {}, {}, None, None, [], np.inf, np.inf
# 
    # print(f"  Modes included : {valid_modes}")
    # if excluded:
        # print(f"  Modes excluded : {excluded} (cutoff above excitation band)")
# 
    # M    = np.column_stack([modal_waveforms[m] for m in valid_modes])
    # cond = np.linalg.cond(M)
    # a    = np.linalg.solve(
            #    M.T @ M + cfg.regularization * np.eye(len(valid_modes)),
            #    M.T @ s)
    # resid = np.linalg.norm(s - M @ a) / (np.linalg.norm(s) + 1e-12)
# 
    # a_abs   = np.abs(a)
    # a_max   = a_abs.max() if a_abs.max() > 0 else 1.0
    # present = {m: bool(a_abs[i] / a_max >= cfg.amplitude_thresh)
            #    for i, m in enumerate(valid_modes)}
    # amplitudes = {m: float(a[i]) for i, m in enumerate(valid_modes)}
# 
    # return amplitudes, present, M, a, valid_modes, cond, resid

    #v2
# def solve(s, modal_waveforms, info, cfg):
    # requested_modes = getattr(cfg, "force_modes", None)
# 
    # if requested_modes is None:
        # mode_names = sorted(m for m, v in info.items() if v["in_band"])
    # else:
        # mode_names = [m for m in requested_modes if m in modal_waveforms]
# 
    # missing = []
    # if requested_modes is not None:
        # missing = [m for m in requested_modes if m not in modal_waveforms]
# 
    # if not mode_names:
        # print("  ERROR: No modes found.")
        # return {}, {}, None, None, [], np.inf, np.inf
# 
    # print(f"  Modes included : {mode_names}")
# 
    # if missing:
        # print(f"  Requested but missing from dispersion files: {missing}")

    # Build matrix
#M_raw = np.column_stack([modal_waveforms[m] for m in mode_names])

    # Remove DC offsets
  #  s0 = s - np.mean(s)
   # M0 = M_raw - np.mean(M_raw, axis=0)

    # Normalize columns
    # scales = np.linalg.norm(M0, axis=0)
    # scales[scales == 0] = 1.0
    # M = M0 / scales
# 
    # cond = np.linalg.cond(M)
# 
    # lam = cfg.regularization
    # a_scaled = np.linalg.solve(
        # M.T @ M + lam * np.eye(len(mode_names)),
        # M.T @ s0
    # )
# 
    # a = a_scaled / scales
# 
    # rec = M_raw @ a
    # resid = np.linalg.norm(s - rec) / (np.linalg.norm(s) + 1e-12)
# 
    # a_abs = np.abs(a)
    # a_max = a_abs.max() if a_abs.max() > 0 else 1.0
# 
    # present = {
        # m: bool(a_abs[i] / a_max >= cfg.amplitude_thresh)
        # for i, m in enumerate(mode_names)
    # }
# 
    # amplitudes = {
        # m: float(a[i])
        # for i, m in enumerate(mode_names)
    # }
# 
    # return amplitudes, present, M_raw, a, mode_names, cond, resid

def solve(s, modal_waveforms, info, cfg):
    """
    Solve s = M*a with automatic filtering of zero-norm modes.

    THREE-TIER MODE FILTERING
    ──────────────────────────
    Tier 1 — Physics filter (automatic, always applied):
      Modes whose synthesized waveform has norm < 1e-4 of the maximum are
      excluded. These modes do not exist at the excitation frequency —
      their cutoff is above the excitation band. Including them adds a
      zero column to M making it singular. No user input needed.

    Tier 2 — force_modes filter (optional, user-controlled):
      If force_modes is set, only those modes are used (subject to Tier 1).
      Use this when you have prior knowledge of which modes are present.
      If None, all Tier-1-surviving modes are used.

    Tier 3 — LASSO selection (in modal_selection.py):
      For experimental data where you have no prior knowledge, use
      run_blind_decomposition() instead of this function. It automatically
      determines which modes are present without any user input.
    """
    # ── Tier 1: remove zero-norm modes ────────────────────────────────────
    norms   = {m: float(np.linalg.norm(mw))
               for m, mw in modal_waveforms.items()}
    max_norm = max(norms.values()) if norms else 1.0
    nonzero  = {m for m, n in norms.items() if n > 1e-4 * max_norm}

    zero_modes = sorted(set(modal_waveforms.keys()) - nonzero)
    if zero_modes:
        print(f"  Tier 1 (physics filter): excluded {zero_modes}")
        print(f"    → These modes have cutoff above the excitation band")

    # ── Tier 2: force_modes filter ────────────────────────────────────────
    requested = getattr(cfg, "force_modes", None)
    if requested is None:
        mode_names = sorted(nonzero)
    else:
        mode_names = [m for m in requested if m in nonzero]
        skipped    = [m for m in requested if m in modal_waveforms
                      and m not in nonzero]
        missing    = [m for m in requested if m not in modal_waveforms]
        if skipped:
            print(f"  Tier 2 (force_modes): also excluded {skipped}")
            print(f"    → These were requested but have zero norm (not in band)")
        if missing:
            print(f"  Requested but not in dispersion files: {missing}")

    if not mode_names:
        print("  ERROR: No modes remain after filtering.")
        return {}, {}, None, None, [], np.inf, np.inf

    print(f"  Modes solved : {mode_names}")
    print(f"  Condition number will reflect {len(mode_names)} real modes")

    # ── Build and solve ───────────────────────────────────────────────────
    M_raw  = np.column_stack([modal_waveforms[m] for m in mode_names])
    s0     = s - np.mean(s)
    M0     = M_raw - np.mean(M_raw, axis=0)
    scales = np.linalg.norm(M0, axis=0)
    scales[scales == 0] = 1.0
    Mn     = M0 / scales
    cond   = np.linalg.cond(Mn)
    lam    = cfg.regularization

    a_sc   = np.linalg.solve(Mn.T @ Mn + lam * np.eye(len(mode_names)),
                              Mn.T @ s0)
    a      = a_sc / scales
    rec    = M_raw @ a
    resid  = float(np.linalg.norm(s - rec) / (np.linalg.norm(s) + 1e-12))

    a_abs  = np.abs(a)
    a_max  = a_abs.max() if a_abs.max() > 0 else 1.0
    present    = {m: bool(a_abs[i] / a_max >= cfg.amplitude_thresh)
                  for i, m in enumerate(mode_names)}
    amplitudes = {m: float(a[i]) for i, m in enumerate(mode_names)}

    return amplitudes, present, M_raw, a, mode_names, cond, resid

# ─────────────────────────────────────────────────────────────────────────────
#  VERIFICATION (simulation only)
# ─────────────────────────────────────────────────────────────────────────────

def verify(t, amplitudes, modal_waveforms, present, gt, info, cfg):
    """
    Compare recovered amplitudes against simulation ground truth.
    """
    print("\n── Verification (simulation GT) ─────────────────────────────────────")
    print("  NOTE: High error is expected — simulation plate ≠ 6mm dispersion plate")
    print()
    print(f"  {'Mode':<8} {'GT peak':>9} {'Rec peak':>10} {'Err%':>7} "
          f"{'Corr':>7} {'Present':>9} {'t_pred(μs)':>12}")
    print("  " + "─"*66)

    results = {}
    for mode in sorted(amplitudes.keys()):
        if mode not in gt:
            continue
        gt_sig  = gt[mode]
        rec_sig = amplitudes[mode] * modal_waveforms[mode]
        gt_pk   = float(np.max(np.abs(hilbert(gt_sig))))
        rc_pk   = float(np.max(np.abs(hilbert(rec_sig))))
        n       = min(len(gt_sig), len(rec_sig))
        try:
            corr = float(np.corrcoef(gt_sig[:n], rec_sig[:n])[0, 1])
        except Exception:
            corr = 0.0
        err  = abs(rc_pk - gt_pk) / (gt_pk + 1e-12) * 100
        pres = "YES" if present.get(mode, False) else "no"
        t_pred = info[mode]["t_pred"] if mode in info else np.nan
        t_str  = f"{t_pred:.1f}" if not np.isnan(t_pred) else "—"
        flag   = " ◄" if mode in cfg.modes_of_interest else ""
        results[mode] = dict(gt_peak=gt_pk, rec_peak=rc_pk,
                              corr=corr, err_pct=err)
        print(f"  {mode:<8} {gt_pk:>9.5f} {rc_pk:>10.5f} {err:>6.1f}% "
              f"{corr:>7.4f} {pres:>9} {t_str:>12}{flag}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

CMAP = plt.cm.tab20(np.linspace(0, 1, 20))
_FIXED = {"A0":0,"S0":1,"A1":2,"S1":3,"ASH1":4,"SSH0":5,"SSH1":6,"ASH2":7}

def _col(mode, i):
    return CMAP[_FIXED.get(mode, (i+8) % 20)]


def plot_all(t, s, amplitudes, modal_waveforms, mode_names,
             present, resid, cond, info, verify_results, cfg):

    M   = np.column_stack([modal_waveforms[m] for m in mode_names])
    a   = np.array([amplitudes[m] for m in mode_names])
    rec = M @ a

    # ── Figure 1: Overview ────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, s,   color="steelblue", lw=1, alpha=0.8, label="Measured s(t)")
    ax0.plot(t, rec, color="tomato", lw=1.3, ls="--",
             label=f"Reconstruction  (residual = {resid:.4f})")
    ax0.set_title(f"Signal vs Reconstruction  |  cond(M) = {cond:.2e}  |  "
                  f"λ = {cfg.regularization:.0e}", fontsize=11)
    ax0.set_xlabel("Time (μs)"); ax0.set_ylabel("Amplitude (nm)")
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, s, color="black", lw=0.5, alpha=0.15, label="Measured")
    for i, mode in enumerate(mode_names):
        contrib = amplitudes[mode] * modal_waveforms[mode]
        lw  = 2.0 if mode in cfg.modes_of_interest else 1.0
        alp = 1.0 if present.get(mode, False) else 0.25
        t_pred = info.get(mode, {}).get("t_pred", np.nan)
        lbl = f"{mode} ({t_pred:.1f}μs)" if not np.isnan(t_pred) else mode
        ax1.plot(t, contrib, color=_col(mode, i), lw=lw, alpha=alp, label=lbl)
    ax1.set_title("Individual modal contributions  (label shows predicted arrival time)")
    ax1.set_xlabel("Time (μs)"); ax1.set_ylabel("nm")
    ax1.legend(ncol=3, fontsize=8); ax1.grid(True, alpha=0.3)

    ax2  = fig.add_subplot(gs[2])
    vals = [abs(amplitudes[m]) for m in mode_names]
    cols = [_col(m, i) if present.get(m, False) else "lightgrey"
            for i, m in enumerate(mode_names)]
    bars = ax2.bar(mode_names, vals, color=cols, edgecolor="white", linewidth=0.8)
    for bar, m in zip(bars, mode_names):
        if m in cfg.modes_of_interest:
            bar.set_edgecolor("gold"); bar.set_linewidth(3)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v*1.02,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax2.set_title("Modal amplitudes  (grey = below threshold, gold = modes of interest)")
    ax2.set_ylabel("|a_n|"); ax2.grid(True, alpha=0.3, axis="y")

    plt.savefig("plots/decomposition_final.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/decomposition_final.png")
    plt.show()

    # ── Figure 2: Modes of interest ───────────────────────────────────────
    moi = [m for m in cfg.modes_of_interest if m in amplitudes]
    if moi:
        fig2, axes = plt.subplots(len(moi)+1, 1,
                                  figsize=(14, 3.8*(len(moi)+1)))
        combined = sum(amplitudes[m]*modal_waveforms[m] for m in moi)

        axes[0].plot(t, s,        color="steelblue", lw=1, alpha=0.7,
                     label="Measured")
        axes[0].plot(t, combined, color="black", lw=1.5, ls="--",
                     label="Sum of modes of interest: " + " + ".join(moi))
        axes[0].set_title("Measured vs Combined Modes of Interest")
        axes[0].set_xlabel("Time (μs)"); axes[0].set_ylabel("nm")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        for i, mode in enumerate(moi):
            ax      = axes[i+1]
            contrib = amplitudes[mode] * modal_waveforms[mode]
            env     = np.abs(hilbert(contrib))
            t_pred  = info.get(mode, {}).get("t_pred", np.nan)
            ax.plot(t, s,       color="steelblue", lw=0.7, alpha=0.15)
            ax.plot(t, contrib, color=_col(mode, i), lw=1.5,
                    label=f"{mode}  a_n = {amplitudes[mode]:+.5f}")
            ax.plot(t, env,     color=_col(mode, i), lw=1.0,
                    ls=":", alpha=0.8, label="Envelope")
            if not np.isnan(t_pred):
                ax.axvline(t_pred, color=_col(mode, i), lw=1.0,
                           ls="--", alpha=0.6, label=f"Predicted arrival {t_pred:.1f}μs")
            ax.set_title(f"{mode}  peak = {np.max(env):.5f} nm  |  "
                         f"a_n = {amplitudes[mode]:+.5f}")
            ax.set_xlabel("Time (μs)"); ax.set_ylabel("nm")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/modes_of_interest_final.png", dpi=200, bbox_inches="tight")
        print("  Saved: plots/modes_of_interest_final.png")
        plt.show()

    # ── Figure 3: Verification ────────────────────────────────────────────
    if verify_results:
        modes  = sorted(verify_results.keys())
        gt_v   = [verify_results[m]["gt_peak"]  for m in modes]
        rc_v   = [verify_results[m]["rec_peak"] for m in modes]
        err_v  = [verify_results[m]["err_pct"]  for m in modes]

        fig3, axes = plt.subplots(1, 2, figsize=(16, 5))
        x = np.arange(len(modes)); w = 0.38
        axes[0].bar(x-w/2, gt_v, w, label="GT peak",   color="steelblue", alpha=0.85)
        axes[0].bar(x+w/2, rc_v, w, label="Recovered", color="tomato",    alpha=0.85)
        axes[0].set_xticks(x); axes[0].set_xticklabels(modes, rotation=45, fontsize=8)
        axes[0].set_title("GT vs Recovered amplitudes 200mm\n"
                          "(high error expected: dispersion curves ≠ simulation plate)")
        axes[0].set_ylabel("nm"); axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        bc = ["tomato" if e > 50 else "steelblue" for e in err_v]
        axes[1].bar(modes, err_v, color=bc, edgecolor="white")
        axes[1].axhline(50, color="tomato", lw=1.5, ls="--", label="50%")
        axes[1].set_title("Amplitude error %")
        axes[1].set_ylabel("Error (%)"); axes[1].tick_params(axis="x", rotation=45)
        axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("plots/verification_final.png", dpi=200, bbox_inches="tight")
        print("  Saved: plots/verification_final.png")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*70)
    print("  LAMB WAVE MODE DECOMPOSITION  —  6mm plate")
    print("="*70)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[0] Loading signal... this is 250mm")
    t, s, exc, fs, dt, gt = load_signal(C)
    N = len(t)

    print("\n[0b] Loading dispersion curves (6mm plate)...")
    disp = load_dispersion(C)
    print(f"  {len(disp)} modes loaded: {sorted(disp.keys())}")

    # ── Dictionary ────────────────────────────────────────────────────────
    print("\n[1] Building modal dictionary...")
    modal_waveforms, info, EXC, F, valid_f = build_dictionary(
        disp, exc, fs, N, C)
    print_coverage(info, C)
    print(f'EXC:{EXC}')

    # ── Solve ─────────────────────────────────────────────────────────────
    print("[2] Solving for amplitudes...")
    amplitudes, present, M, a_vec, mode_names, cond, resid = solve(
        s, modal_waveforms, info, C)

    if not mode_names:
        print("\nNo modes solved. Check dispersion files and excitation band.")
    else:
        print(f"\n  Condition number : {cond:.3e}")
        print(f"  Relative residual: {resid:.4f}")
        if cond > 1e6:
            print("  ⚠  High condition number — increase C.regularization")
        if resid > 0.5:
            print("  ⚠  High residual — this is expected when using 6mm curves")
            print("     on the 1mm simulation data. On real 6mm plate data")
            print("     the residual will be much lower.")

        # ── Results table ─────────────────────────────────────────────────
        print("\n── Modal Amplitudes ─────────────────────────────────────────────")
        print(f"  {'Mode':<8} {'a_n':>10} {'|a_n|':>10} "
              f"{'Present':>9} {'t_predicted (μs)':>18}")
        print("  " + "─"*60)
        for mode in mode_names:
            pres   = "YES" if present[mode] else "no"
            t_pred = info[mode]["t_pred"]
            t_str  = f"{t_pred:.2f}" if not np.isnan(t_pred) else "—"
            flag   = "  ◄" if mode in C.modes_of_interest else ""
            print(f"  {mode:<8} {amplitudes[mode]:>10.5f} "
                  f"{abs(amplitudes[mode]):>10.5f} {pres:>9} {t_str:>18}{flag}")

        # ── Verification ──────────────────────────────────────────────────
        verify_results = None
        if C.run_verification and gt:
            print("\n[3] Comparing against simulation GT...")
            verify_results = verify(
                t, amplitudes, modal_waveforms, present, gt, info, C)

        # ── Plots ─────────────────────────────────────────────────────────
        print("\n[4] Plotting...")
        plot_all(t, s, amplitudes, modal_waveforms, mode_names,
                 present, resid, cond, info, verify_results, C)

    print("\n" + "="*70 + "\n  DONE\n" + "="*70)
    print(info)
# from not_happy_v6 import run_all_diagnostics
# 
# scores_reliability = run_all_diagnostics(
    # s               = s,
    # modal_waveforms = modal_waveforms,
    # mode_names      = mode_names,
    # regularization  = C.regularization,
    # n_bootstrap     = 300,
    # noise_fraction  = 0.05,   # 5% noise — adjust to your SNR
# )
# 
# print(scores_reliability)