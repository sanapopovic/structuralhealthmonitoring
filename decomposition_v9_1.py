"""
═══════════════════════════════════════════════════════════════════════════════
  decomposition_base.py
  ─────────────────────────────────────────────────────────────────────────────
  Shared loading, dispersion, dictionary building and verification utilities.
  Imported by lasso_decomposition.py and bayesian_decomposition.py.

  DATA STRUCTURE (from your 200mm zip)
  ──────────────────────────────────────
  You have two simulation files per distance and measurement direction:

    @7.9866MHzmm  →  f_centre ≈ 1.33 MHz  →  FUNDAMENTAL file
    @15.963MHzmm  →  f_centre ≈ 2.66 MHz  →  SECOND HARMONIC file

  Each file contains:
    - 'Propagation time (micsec)'  : time axis
    - 'ExcitationSignal'           : the excitation e(t)
    - 'X Propagated signal (nm)'   : ground truth for each individual mode
    - 'Sum Propagated signal (nm)' : the sum of all modes = what you measure

  The 'Sum' column is your measured signal s(t).
  The individual mode columns are ground truth — only available in simulation.
  In real experiments you only have the Sum.

  YOUR RESEARCH WORKFLOW
  ──────────────────────
  Objective 2.3.1: Run decomposition on Sum, compare against GT columns
                   to validate the method on synthetic data.
  Objective 2.3.2: Apply to experimental data (no GT columns available).
  Objective 2.3.3: Compute β' = A₂ / A₁² from recovered amplitudes.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  FILE PATHS  —  edit these for your machine
# ─────────────────────────────────────────────────────────────────────────────

# Simulation files (Dataset C — ground truth available)
#FUNDAMENTAL_FILE    = r"Data/In-plane_TemporalResponse_7_9866MHzmm_350mm_modified.xlsx" #modified: without A2, A4, A5, S4. Adjusted Sum as well
#SECOND_HARM_FILE    = r"Data/In-plane_A2_TemporalResponse_15_963MHzmm_350mm_modified.xlsx"

FUNDAMENTAL_FILE    = r"Data/In-plane_TemporalResponse_7_9866MHzmm_350mm_modified.xlsx"  #the normal files
SECOND_HARM_FILE    = r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@350mm.xlsx"


# Dispersion curve files (Dataset A)
DISP_FILES = {
    "A_Lamb": r"Data/Dispersion Curve 6mm 3000khz_A_Lamb.xlsx",
    "S_Lamb": r"Data/Dispersion Curve 6mm 3000khz_S_Lamb.xlsx",
}

# Physics
PROPAGATION_MM = 350.0


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ONE SIGNAL FILE
# ─────────────────────────────────────────────────────────────────────────────

def load_file(filepath,
              time_col="Propagation time (micsec)",
              signal_col="Sum Propagated signal (nm)",
              excitation_col="ExcitationSignal"):
    """
    Load a simulation file and return all components.

    Returns
    -------
    t       : time axis (μs)
    s       : measured sum signal (nm)  — this is what you decompose
    exc     : excitation signal
    fs      : sampling frequency (MHz)
    gt      : dict of {mode_name: gt_signal}  — empty for real experiments
    f_centre: dominant frequency of excitation (MHz)
    """
    df      = pd.read_excel(filepath)
    t       = df[time_col].to_numpy(dtype=float)
    s       = df[signal_col].to_numpy(dtype=float)
    dt      = float(np.mean(np.diff(t)))
    fs      = 1.0 / dt

    exc_raw = df[excitation_col].to_numpy(dtype=float)
    
    exc     = np.where(np.isnan(exc_raw), 0.0, exc_raw)

    # Ground truth individual mode columns (simulation only)
    gt_cols = [c for c in df.columns
               if "Propagated" in c and "Sum" not in c]
    gt = {c.split(" Propagated")[0]: df[c].to_numpy(dtype=float)
          for c in gt_cols}

    # Dominant excitation frequency
    F        = np.fft.rfftfreq(len(t), d=1.0/fs)
    EXC_abs  = np.abs(np.fft.rfft(exc))
    f_centre = float(F[np.argmax(EXC_abs)])

    print(f"  File     : {filepath.split('/')[-1]}")
    print(f"  Samples  : {len(t)}   dt = {dt:.5f} μs   fs = {fs:.3f} MHz")
    print(f"  f_centre : {f_centre:.4f} MHz")
    print(f"  GT modes : {sorted(gt.keys()) if gt else 'none (experimental data)'}")
    print(f"  Sum peak : {np.max(np.abs(s)):.6f} nm")

    return t, s, exc, fs, gt, f_centre


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DISPERSION CURVES
# ─────────────────────────────────────────────────────────────────────────────

def load_dispersion(disp_files=DISP_FILES):
    """
    Load dispersion curve files.
    Returns dict: mode_name -> {freq, k, tp, cg, type}
    """
    disp = {}
    for label, fpath in disp_files.items():
        try:
            df = pd.read_excel(fpath)
        except FileNotFoundError:
            print(f"  WARNING: {fpath} not found — skipping.")
            continue
        mode_names = [c.split(" f (MHz)")[0]
                      for c in df.columns if c.endswith(" f (MHz)")]
        for mode in mode_names:
            freq  = df[f"{mode} f (MHz)"].to_numpy()
            cp    = df[f"{mode} Phase velocity (m/ms)"].to_numpy()
            k     = df[f"{mode} Wavenumber (rad/mm)"].to_numpy()
            tp    = df[f"{mode} Propagation time (micsec)"].to_numpy()
            cg_col = f"{mode} Energy velocity (m/ms)"
            cg    = df[cg_col].to_numpy() if cg_col in df.columns \
                    else np.full_like(freq, np.nan)
            valid = (~np.isnan(freq) & ~np.isnan(cp) &
                     (cp > 0) & (cp < 50) & ~np.isnan(k) & (k >= 0))
            if valid.sum() < 5:
                continue
            disp[mode] = {
                "freq": freq[valid], "cp": cp[valid],
                "k":    k[valid],    "tp": tp[valid],
                "cg":   cg[valid],   "type": label,
            }
    print(f"  Loaded {len(disp)} modes: {sorted(disp.keys())}")
    return disp


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD MODAL DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

def build_dictionary(disp, exc, fs, N, f_centre, propagation_mm=PROPAGATION_MM,
                     snr_threshold=0.001):
    """
    Synthesize predicted waveform m_n(t) for every mode in disp.

    For each mode n:
      1. Interpolate k_n(f) onto the signal frequency grid
      2. phi_n(f) = k_n(f) * d           [dispersive phase in rad]
      3. H_n(f)   = exp(-i * phi_n(f))   [transfer function]
      4. m_n(t)   = IFFT[ FFT(exc) * H_n(f) ]

    Only frequency bins where the excitation has meaningful energy are used
    (controlled by snr_threshold). Outside the excitation band, H_n = 0
    so no phantom energy is synthesized.

    Returns
    -------
    modal_waveforms : dict {mode: waveform array of length N}
    info            : dict {mode: {t_pred, f_min, f_max, type}}
    M               : modal matrix (N × n_modes) — columns are waveforms
    mode_names      : list of mode names corresponding to M columns
    """
    F       = np.fft.rfftfreq(N, d=1.0/fs)
    EXC     = np.fft.rfft(exc, n=N)
    EXC_abs = np.abs(EXC)
    valid_f = EXC_abs > snr_threshold * EXC_abs.max()

    modal_waveforms = {}
    info            = {}

    for mode, c in disp.items():
        fv = c["freq"]; kv = c["k"]; tv = c["tp"]

        # Predicted arrival at excitation centre frequency
        valid_tp = ~np.isnan(tv) & (tv > 0) & (tv < 1e6)
        if valid_tp.sum() > 2:
            itp    = interp1d(fv[valid_tp], tv[valid_tp],
                              bounds_error=False, fill_value=np.nan)
            t_pred = float(itp(f_centre))
        else:
            t_pred = np.nan

        # Synthesize transfer function
        ik  = interp1d(fv, kv, kind="linear",
                       bounds_error=False, fill_value=0.0)
        k_F = ik(F)
        phi = np.where(
            valid_f & (F >= fv.min()) & (F <= fv.max()) & (k_F > 0),
            k_F * propagation_mm, 0.0)
        H   = np.where(phi != 0, np.exp(-1j * phi), 0.0 + 0j)
        m_n = np.fft.irfft(EXC * H, n=N)

        modal_waveforms[mode] = m_n
        info[mode] = {
            "t_pred": t_pred,
            "f_min":  float(fv.min()),
            "f_max":  float(fv.max()),
            "type":   c["type"],
        }

    # Filter out zero-norm modes (cutoff above excitation band)
    norms    = {m: float(np.linalg.norm(mw))
                for m, mw in modal_waveforms.items()}
    max_norm = max(norms.values()) if norms else 1.0
    active   = {m: mw for m, mw in modal_waveforms.items()
                if norms[m] > 1e-4 * max_norm}

    excluded = sorted(set(modal_waveforms) - set(active))
    if excluded:
        print(f"  Excluded (zero norm / below cutoff): {excluded}")

    mode_names = sorted(active.keys())
    M          = np.column_stack([active[m] for m in mode_names])

    print(f"  Active modes in band: {mode_names}")
    return active, info, M, mode_names


# ─────────────────────────────────────────────────────────────────────────────
#  ENVELOPE PEAK EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def envelope_peaks(amplitudes, modal_waveforms):
    """
    For every solved mode compute the envelope peak of a_n * m_n(t).

    This is the physical amplitude — peak displacement at the receiver.
    Uses Hilbert transform to get the instantaneous amplitude envelope,
    which is independent of sample grid timing and normalisation choices.
    """
    peaks = {}
    for mode, a_n in amplitudes.items():
        if mode not in modal_waveforms:
            continue
        contribution  = a_n * modal_waveforms[mode]
        envelope      = np.abs(hilbert(contribution))
        peaks[mode]   = float(np.max(envelope))
    return peaks


# ─────────────────────────────────────────────────────────────────────────────
#  VERIFICATION AGAINST GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def verify_against_gt(amplitudes, modal_waveforms, gt, label=""):
    """
    Compare recovered amplitudes against simulation ground truth.

    For each mode where GT is available:
      - Compute envelope peak of GT signal
      - Compute envelope peak of recovered signal
      - Report error % and correlation

    This is your Objective 2.3.1 validation.
    """
    if not gt:
        print("  No ground truth available (experimental data).")
        return {}

    print(f"\n  ── Verification against GT {label} {'─'*(40-len(label))}")
    print(f"  {'Mode':<8} {'GT peak':>10} {'Rec peak':>10} "
          f"{'Error %':>9} {'Corr':>8}")
    print("  " + "─" * 50)

    results = {}
    for mode in sorted(amplitudes.keys()):
        if mode not in gt:
            continue
        gt_sig  = gt[mode]
        rec_sig = amplitudes[mode] * modal_waveforms[mode]

        gt_pk  = float(np.max(np.abs(hilbert(gt_sig))))
        rec_pk = float(np.max(np.abs(hilbert(rec_sig))))

        n = min(len(gt_sig), len(rec_sig))
        try:
            corr = float(np.corrcoef(gt_sig[:n], rec_sig[:n])[0, 1])
        except Exception:
            corr = 0.0

        err = abs(rec_pk - gt_pk) / (gt_pk + 1e-12) * 100
        results[mode] = {"gt_peak": gt_pk, "rec_peak": rec_pk,
                         "err_pct": err,   "corr": corr}
        flag = " ◄" if err < 20 else ("  ⚠" if err > 50 else "")
        print(f"  {mode:<8} {gt_pk:>10.5f} {rec_pk:>10.5f} "
              f"{err:>8.1f}% {corr:>8.4f}{flag}")

    good = sum(1 for v in results.values() if v["err_pct"] < 20)
    print(f"\n  {good}/{len(results)} modes within 20% error")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  NORMALISE MATRIX  (shared between LASSO and Bayes)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_matrix(s, M):
    """
    Remove DC offsets and normalise M columns to unit norm.

    Returns s0, M_norm, scales
    where  a_physical = a_normalised / scales
    """
    s0     = s - np.mean(s)
    M0     = M - np.mean(M, axis=0)
    scales = np.linalg.norm(M0, axis=0)
    scales[scales == 0] = 1.0
    M_norm = M0 / scales
    return s0, M_norm, scales