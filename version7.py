"""
═══════════════════════════════════════════════════════════════════════════════
  LAMB WAVE AMPLITUDE EXTRACTION  —  Two-band approach
  ─────────────────────────────────────────────────────────────────────────────

  PURPOSE
  ───────
  Reliably extract mode amplitudes at the fundamental frequency band (f)
  and the second harmonic band (2f) separately.

  WHY TWO SEPARATE SOLVES INSTEAD OF ONE
  ───────────────────────────────────────
  The original code ran one decomposition on the full broadband signal.
  This is problematic because:

    1. Modes at f and modes at 2f have completely different wavenumbers,
       so their synthesized waveforms m_n(t) look nothing alike. Mixing
       them in one matrix M creates a block-diagonal structure where
       the f-block and 2f-block don't interact — but the solver still
       has to invert the full matrix, which is larger and more sensitive
       to numerical errors than needed.

    2. The second harmonic signal (A₂) is 2-3 orders of magnitude smaller
       than the fundamental (A₁). In a joint solve, the large-amplitude
       fundamental modes dominate the least-squares fit and the solver
       effectively ignores the tiny second harmonic contributions. You
       lose precision exactly where you need it most.

    3. Bandpass filtering before each solve removes out-of-band noise,
       which directly improves the condition number of M and the
       reliability of the recovered amplitudes.

  METHOD
  ──────
  Step 1: Bandpass filter s(t) around the fundamental band → s_f(t)
  Step 2: Build modal dictionary using only modes in that band
  Step 3: Solve s_f = M_f * a_f  →  get fundamental mode amplitudes
  Step 4: Bandpass filter s(t) around the second harmonic band → s_2f(t)
  Step 5: Build modal dictionary using only modes in that band
  Step 6: Solve s_2f = M_2f * a_2f  →  get second harmonic amplitudes
  Step 7: Extract peak envelope amplitude for each mode of interest

  WHY ENVELOPE PEAK INSTEAD OF RAW a_n
  ──────────────────────────────────────
  The scalar a_n from the solver is a scaling factor for the entire
  synthesized waveform m_n(t). The actual physical amplitude of the mode
  at the receiver is the peak of the envelope of a_n * m_n(t), computed
  via the Hilbert transform. This is what you should use for β'.

  REPEATABILITY STRUCTURE
  ───────────────────────
  extract_amplitudes() processes a single signal file and returns a dict
  of amplitudes. Call it in a loop over your 10 repetitions to build
  the distribution needed for your precision/repeatability analysis.

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, hilbert
import os, warnings

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # ── Files ─────────────────────────────────────────────────────────────
    signal_file    = r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"
    signal_col     = "Sum Propagated signal (nm)"
    excitation_col = "ExcitationSignal"
    time_col       = "Propagation time (micsec)"

    disp_files = {
        "A_Lamb": r"Data/Dispersion Curve 6mm 3000khz_A_Lamb.xlsx",
        "S_Lamb": r"Data/Dispersion Curve 6mm 3000khz_S_Lamb.xlsx",
    }

    # ── Physics ───────────────────────────────────────────────────────────
    propagation_mm = 200.0

    # ── Frequency bands ───────────────────────────────────────────────────
    # Set these based on your excitation frequency.
    # If your excitation centre is at f_c MHz:
    #   fundamental band  ≈ [f_c - bw/2,  f_c + bw/2]
    #   second harmonic   ≈ [2*f_c - bw/2, 2*f_c + bw/2]
    # Bandwidth (bw) should match your transducer bandwidth.
    # These are the ONLY parameters you need to tune per experiment.
    f_centre_MHz    = None    # if None, auto-detected from excitation spectrum
    bandwidth_MHz   = None    # if None, auto-detected from excitation spectrum

    # ── Solver ────────────────────────────────────────────────────────────
    regularization  = 1e-5    # Tikhonov lambda
    filter_order    = 4       # Butterworth filter order — 4 is a good default:
                              # sharp enough to isolate bands, gentle enough
                              # not to distort the waveform shape

    # ── Modes of interest for amplitude extraction ─────────────────────────
    # Fundamental modes to extract amplitude from (at f)
    fundamental_modes   = ["S1", "A1"]
    # Second harmonic modes to extract amplitude from (at 2f)
    second_harmonic_modes = ["S2", "A2", "S4", "A4"]


C = Config()


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_signal(cfg):
    df      = pd.read_excel(cfg.signal_file)
    t       = df[cfg.time_col].to_numpy()
    s       = df[cfg.signal_col].to_numpy()
    dt      = float(np.mean(np.diff(t)))
    fs      = 1.0 / dt
    exc_raw = df[cfg.excitation_col].to_numpy()
    exc     = np.where(np.isnan(exc_raw), 0.0, exc_raw)

    # Ground truth columns for validation (simulation data only)
    gt_cols = [c for c in df.columns if "Propagated" in c and "Sum" not in c]
    gt      = {c.split(" Propagated")[0]: df[c].to_numpy() for c in gt_cols}

    print(f"  Samples  : {len(t)}   dt = {dt:.5f} μs   fs = {fs:.3f} MHz")
    print(f"  Duration : 0 – {t[-1]:.1f} μs")
    return t, s, exc, fs, dt, gt


def load_dispersion(cfg):
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
            cg   = df[cg_col].to_numpy() if cg_col in df.columns \
                   else np.full_like(freq, np.nan)
            valid = (~np.isnan(freq) & ~np.isnan(cp) &
                     (cp > 0) & (cp < 50) & ~np.isnan(k) & (k >= 0))
            if valid.sum() < 5:
                continue
            disp[mode] = {"freq": freq[valid], "cp": cp[valid],
                          "k": k[valid], "tp": tp[valid],
                          "cg": cg[valid], "type": label}
    return disp


# ─────────────────────────────────────────────────────────────────────────────
#  BAND DETECTION  —  auto-detect f_centre and bandwidth from excitation
# ─────────────────────────────────────────────────────────────────────────────

def detect_excitation_band(exc, fs):
    """
    Automatically find the centre frequency and -6dB bandwidth of the
    excitation signal from its power spectrum.

    WHY -6dB: This is the standard definition of transducer bandwidth.
    It captures the frequency range where the excitation has meaningful
    energy — below -6dB the signal-to-noise drops too much to trust
    the amplitude estimates from the solver.
    """
    N     = len(exc)
    F     = np.fft.rfftfreq(N, d=1.0/fs)
    EXC   = np.abs(np.fft.rfft(exc))

    # Centre = frequency of peak power
    i_peak   = np.argmax(EXC)
    f_centre = float(F[i_peak])

    # -6dB bandwidth
    threshold  = EXC[i_peak] / 2.0          # amplitude -6dB = power -3dB ÷ 2
    in_band    = EXC > threshold
    f_in_band  = F[in_band]
    bandwidth  = float(f_in_band.max() - f_in_band.min())

    print(f"  Auto-detected excitation centre : {f_centre:.4f} MHz")
    print(f"  Auto-detected -6dB bandwidth    : {bandwidth:.4f} MHz")
    print(f"  Fundamental band : {f_centre - bandwidth/2:.4f} – "
          f"{f_centre + bandwidth/2:.4f} MHz")
    print(f"  2nd harmonic band: {2*f_centre - bandwidth/2:.4f} – "
          f"{2*f_centre + bandwidth/2:.4f} MHz")
    return f_centre, bandwidth


# ─────────────────────────────────────────────────────────────────────────────
#  BANDPASS FILTER
# ─────────────────────────────────────────────────────────────────────────────

def bandpass(signal, f_low, f_high, fs, order=4):
    """
    Zero-phase Butterworth bandpass filter.

    WHY ZERO-PHASE (filtfilt):
      A standard causal filter shifts the signal in time. For Lamb wave
      analysis, timing is everything — the arrival time of each mode is
      how we distinguish them. A phase shift would move the waveform
      packets to the wrong time, corrupting the amplitude estimate.
      filtfilt applies the filter forward then backward, cancelling the
      phase shift entirely while doubling the effective filter order.

    WHY BUTTERWORTH:
      Maximally flat passband — no ripple that could be mistaken for
      signal amplitude variation. Other filters (Chebyshev, elliptic)
      have sharper cutoffs but introduce ripple in the passband which
      directly corrupts amplitude measurements.
    """
    nyq    = fs / 2.0
    f_low  = max(f_low,  1e-4)        # guard against zero/negative
    f_high = min(f_high, nyq * 0.99)  # guard against exceeding Nyquist
    b, a   = butter(order, [f_low / nyq, f_high / nyq], btype="band")
    return filtfilt(b, a, signal)


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD MODAL DICTIONARY  (for one frequency band)
# ─────────────────────────────────────────────────────────────────────────────

def build_dictionary_band(disp, exc_filtered, fs, N, f_low, f_high, cfg):
    """
    Synthesize predicted waveforms for modes that exist in [f_low, f_high].

    KEY DIFFERENCE FROM ORIGINAL:
      The original built waveforms for ALL modes across ALL frequencies
      and then filtered by coverage. Here we only build waveforms for
      modes whose dispersion data overlaps with [f_low, f_high].
      This means M is smaller and better conditioned.

    The excitation passed in here (exc_filtered) is already bandpassed,
    so EXC(f) is zero outside the band. This means H_n(f) outside the
    band contributes nothing — we're not synthesizing phantom energy.
    """
    F       = np.fft.rfftfreq(N, d=1.0/fs)
    EXC     = np.fft.rfft(exc_filtered, n=N)

    modal_waveforms = {}
    info            = {}

    for mode, c in disp.items():
        fv = c["freq"]; kv = c["k"]; tv = c["tp"]

        # Does this mode's dispersion data overlap with our band?
        overlap = (fv.max() >= f_low) and (fv.min() <= f_high)
        if not overlap:
            continue

        # Predicted arrival at band centre
        f_centre_band = (f_low + f_high) / 2.0
        valid_tp = ~np.isnan(tv) & (tv > 0) & (tv < 1e6)
        if valid_tp.sum() > 2:
            itp    = interp1d(fv[valid_tp], tv[valid_tp],
                              bounds_error=False, fill_value=np.nan)
            t_pred = float(itp(f_centre_band))
        else:
            t_pred = np.nan

        # Synthesize transfer function over [f_low, f_high] only
        ik  = interp1d(fv, kv, kind="linear", bounds_error=False, fill_value=0.0)
        k_F = ik(F)

        # Only apply phase where: (a) we are in the band AND
        #                         (b) dispersion data exists (k > 0)
        in_band_mask = (F >= f_low) & (F <= f_high) & (k_F > 0)
        phi = np.where(in_band_mask, k_F * cfg.propagation_mm, 0.0)
        H   = np.where(phi != 0, np.exp(-1j * phi), 0.0 + 0j)
        m_n = np.fft.irfft(EXC * H, n=N)

        modal_waveforms[mode] = m_n
        info[mode] = {"t_pred": t_pred, "type": c["type"],
                      "f_min": fv.min(), "f_max": fv.max()}

    return modal_waveforms, info


# ─────────────────────────────────────────────────────────────────────────────
#  SOLVE  (Tikhonov least squares for one band)
# ─────────────────────────────────────────────────────────────────────────────

def solve_band(s_filtered, modal_waveforms, cfg, band_label=""):
    """
    Solve s_filtered = M * a using Tikhonov regularisation.

    s_filtered is already bandpassed — it contains only the signal energy
    in the band of interest. M contains only modes from that band.
    This is a much smaller, better-conditioned problem than the original.

    NORMALISATION:
      Each column of M is normalised to unit norm before solving. Without
      this, modes with larger natural amplitudes dominate the fit regardless
      of their actual contribution to the signal. After solving, we rescale
      back so the returned amplitudes are in physical units (nm).

    DC REMOVAL:
      Both s and M columns have their means removed. A DC offset in the
      signal would otherwise be attributed to whichever mode happens to
      have a non-zero mean, giving a spurious amplitude.
    """
    # Remove zero-norm modes (modes with no energy in this band)
    norms    = {m: float(np.linalg.norm(mw)) for m, mw in modal_waveforms.items()}
    max_norm = max(norms.values()) if norms else 1.0
    active   = {m: mw for m, mw in modal_waveforms.items()
                if norms[m] > 1e-4 * max_norm}

    if not active:
        print(f"  [{band_label}] No active modes found in band.")
        return {}, {}, np.inf, np.inf

    mode_names = sorted(active.keys())
    print(f"  [{band_label}] Active modes: {mode_names}")

    M_raw  = np.column_stack([active[m] for m in mode_names])
    s0     = s_filtered - np.mean(s_filtered)
    M0     = M_raw - np.mean(M_raw, axis=0)

    # Normalise columns
    scales = np.linalg.norm(M0, axis=0)
    scales[scales == 0] = 1.0
    Mn     = M0 / scales

    cond   = np.linalg.cond(Mn)
    lam    = cfg.regularization

    # Tikhonov solve: a = (MᵀM + λI)⁻¹ Mᵀ s
    a_sc   = np.linalg.solve(
                Mn.T @ Mn + lam * np.eye(len(mode_names)),
                Mn.T @ s0)
    a      = a_sc / scales

    rec    = M_raw @ a
    resid  = float(np.linalg.norm(s_filtered - rec) /
                   (np.linalg.norm(s_filtered) + 1e-12))

    amplitudes = {m: float(a[i]) for i, m in enumerate(mode_names)}
    return amplitudes, active, cond, resid


# ─────────────────────────────────────────────────────────────────────────────
#  AMPLITUDE EXTRACTION  —  envelope peak of each mode's contribution
# ─────────────────────────────────────────────────────────────────────────────

def extract_envelope_peaks(amplitudes, modal_waveforms):
    """
    For every mode the solver actually found, compute the peak of the
    envelope of its reconstructed contribution: a_n * m_n(t).

    WHY ALL SOLVED MODES (not a filter list):
      The previous version filtered by a user-supplied list, which meant
      any mode not in that list silently returned NaN even if the solver
      had perfectly good amplitude for it. Now we extract peaks for
      everything the solver returned — you see all of them.

    WHY ENVELOPE PEAK AND NOT RAW a_n:
      a_n is a scalar that scales the entire synthetic waveform m_n(t).
      The physical amplitude at the receiver is the peak displacement,
      i.e. the peak of the analytic signal envelope via Hilbert transform.
      This is independent of normalisation choices in the solver and is
      directly comparable across experiments and distances.

    WHY HILBERT TRANSFORM FOR ENVELOPE:
      The envelope is |analytic_signal| = |signal + i*H{signal}|.
      This gives instantaneous amplitude — for a dispersive waveform
      packet this is the physically meaningful quantity. Taking just
      max(|signal|) would pick the peak of one oscillation cycle, which
      is sensitive to the exact sample grid timing.
    """
    peaks = {}
    for mode in sorted(amplitudes.keys()):
        if mode not in modal_waveforms:
            continue
        contribution = amplitudes[mode] * modal_waveforms[mode]
        envelope     = np.abs(hilbert(contribution))
        peaks[mode]  = float(np.max(envelope))
    return peaks


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXTRACTION FUNCTION  —  call this per file for repeatability loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_amplitudes(signal_file, cfg, disp, f_centre, bandwidth, verbose=True):
    """
    Full pipeline for one signal file.
    Returns a dict of envelope peak amplitudes for the modes of interest.

    This is designed to be called in a loop over your 10 repetitions:

        results = []
        for file in repetition_files:
            cfg.signal_file = file
            amps = extract_amplitudes(file, cfg, disp, f_centre, bandwidth)
            results.append(amps)

        # Then analyse the distribution across repetitions:
        import pandas as pd
        df = pd.DataFrame(results)
        print(df.describe())   # mean, std, min, max per mode
    """
    # Load
    df      = pd.read_excel(signal_file)
    t       = df[cfg.time_col].to_numpy()
    s       = df[cfg.signal_col].to_numpy()
    dt      = float(np.mean(np.diff(t)))
    fs      = 1.0 / dt
    exc_raw = df[cfg.excitation_col].to_numpy()
    exc     = np.where(np.isnan(exc_raw), 0.0, exc_raw)
    N       = len(t)

    # Ground truth (simulation only)
    gt_cols = [c for c in df.columns if "Propagated" in c and "Sum" not in c]
    gt      = {c.split(" Propagated")[0]: df[c].to_numpy() for c in gt_cols}

    # ── Fundamental band ──────────────────────────────────────────────────
    f_low_f  = f_centre - bandwidth / 2.0
    f_high_f = f_centre + bandwidth / 2.0

    s_f   = bandpass(s,   f_low_f, f_high_f, fs, cfg.filter_order)
    exc_f = bandpass(exc, f_low_f, f_high_f, fs, cfg.filter_order)

    mw_f, info_f = build_dictionary_band(
        disp, exc_f, fs, N, f_low_f, f_high_f, cfg)
    amps_f, active_f, cond_f, resid_f = solve_band(
        s_f, mw_f, cfg, band_label=f"fundamental {f_low_f:.3f}-{f_high_f:.3f} MHz")

    # ── Second harmonic band ──────────────────────────────────────────────
    f_low_2f  = 2 * f_centre - bandwidth / 2.0
    f_high_2f = 2 * f_centre + bandwidth / 2.0

    s_2f   = bandpass(s,   f_low_2f, f_high_2f, fs, cfg.filter_order)
    exc_2f = bandpass(exc, f_low_2f, f_high_2f, fs, cfg.filter_order)

    mw_2f, info_2f = build_dictionary_band(
        disp, exc_2f, fs, N, f_low_2f, f_high_2f, cfg)
    amps_2f, active_2f, cond_2f, resid_2f = solve_band(
        s_2f, mw_2f, cfg, band_label=f"2nd harmonic {f_low_2f:.3f}-{f_high_2f:.3f} MHz")

    if verbose:
        print(f"\n  Fundamental   — cond: {cond_f:.2e}  resid: {resid_f:.4f}")
        print(f"  2nd Harmonic  — cond: {cond_2f:.2e}  resid: {resid_2f:.4f}")
        if cond_f > 1e6 or cond_2f > 1e6:
            print("  ⚠  High condition number — consider increasing regularization")

    # ── Extract envelope peaks — ALL solved modes, no filter list ────────
    peaks_f  = extract_envelope_peaks(amps_f,  active_f)
    peaks_2f = extract_envelope_peaks(amps_2f, active_2f)

    # Tag each peak with which band it came from
    all_peaks = {}
    for m, v in peaks_f.items():
        all_peaks[m] = {"peak": v, "band": "fundamental",
                        "t_pred": info_f.get(m, {}).get("t_pred", np.nan)}
    for m, v in peaks_2f.items():
        all_peaks[m] = {"peak": v, "band": "2nd harmonic",
                        "t_pred": info_2f.get(m, {}).get("t_pred", np.nan)}

    if verbose:
        print("\n  ── Envelope peak amplitudes (all solved modes) ───────────────")
        print(f"  {'Mode':<8} {'Band':<16} {'Peak (nm)':>14}  "
              f"{'t_pred (μs)':>12}")
        print("  " + "─" * 56)
        for band_label, peaks_dict, info_dict in [
            ("fundamental", peaks_f,  info_f),
            ("2nd harmonic", peaks_2f, info_2f),
        ]:
            for mode in sorted(peaks_dict.keys()):
                t_pred  = info_dict.get(mode, {}).get("t_pred", np.nan)
                t_str   = f"{t_pred:.2f}" if not np.isnan(t_pred) else "—"
                val_str = f"{peaks_dict[mode]:.6f}"
                print(f"  {mode:<8} {band_label:<16} {val_str:>14}  {t_str:>12}")

    # ── Ground truth verification (simulation data only) ──────────────────
    verify_results = {}
    if gt and verbose:
        print("\n  ── Verification against ground truth ─────────────────────────")
        print(f"  {'Mode':<8} {'GT peak (nm)':>14} {'Rec peak (nm)':>14} "
              f"{'Error %':>9} {'Corr':>7}")
        print("  " + "─" * 58)

        # Check both band solvers against GT
        for amps, active, band_label in [
            (amps_f,  active_f,  "fundamental"),
            (amps_2f, active_2f, "2nd harmonic"),
        ]:
            for mode in sorted(amps.keys()):
                if mode not in gt:
                    continue
                gt_sig  = gt[mode]
                rec_sig = amps[mode] * active[mode]

                # GT peak via envelope
                gt_pk  = float(np.max(np.abs(hilbert(gt_sig))))
                rec_pk = float(np.max(np.abs(hilbert(rec_sig))))

                # Correlation
                n = min(len(gt_sig), len(rec_sig))
                try:
                    corr = float(np.corrcoef(gt_sig[:n], rec_sig[:n])[0, 1])
                except Exception:
                    corr = 0.0

                err = abs(rec_pk - gt_pk) / (gt_pk + 1e-12) * 100
                verify_results[mode] = {
                    "gt_peak": gt_pk, "rec_peak": rec_pk,
                    "err_pct": err,   "corr": corr,
                    "band": band_label
                }
                print(f"  {mode:<8} {gt_pk:>14.6f} {rec_pk:>14.6f} "
                      f"{err:>8.1f}% {corr:>7.4f}  [{band_label}]")

    return {
        "t": t, "s": s,
        "s_f": s_f, "s_2f": s_2f,
        "amps_f": amps_f, "amps_2f": amps_2f,
        "mw_f": active_f, "mw_2f": active_2f,
        "info_f": info_f, "info_2f": info_2f,
        "peaks": all_peaks,         # {mode: {peak, band, t_pred}}
        "peaks_f": peaks_f,         # fundamental band peaks only
        "peaks_2f": peaks_2f,       # 2nd harmonic band peaks only
        "cond_f": cond_f, "cond_2f": cond_2f,
        "resid_f": resid_f, "resid_2f": resid_2f,
        "gt": gt,
        "verify": verify_results,   # empty dict if no GT columns in file
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(result, cfg, f_centre, bandwidth):
    t    = result["t"]
    s    = result["s"]
    s_f  = result["s_f"]
    s_2f = result["s_2f"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(
        f"Two-band Lamb Wave Decomposition  |  "
        f"d = {cfg.propagation_mm} mm  |  "
        f"f_c = {f_centre:.3f} MHz",
        fontsize=12)

    # ── Full signal ───────────────────────────────────────────────────────
    axes[0].plot(t, s, color="steelblue", lw=0.8, label="Full signal s(t)")
    axes[0].set_title("Full measured signal")
    axes[0].set_ylabel("nm"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # ── Fundamental band: signal + reconstruction + mode contributions ────
    amps_f = result["amps_f"]
    mw_f   = result["mw_f"]
    if amps_f:
        rec_f = sum(amps_f[m] * mw_f[m] for m in amps_f)
        axes[1].plot(t, s_f,  color="steelblue", lw=0.8, alpha=0.6,
                     label=f"Filtered fundamental band "
                           f"({f_centre - bandwidth/2:.3f}–"
                           f"{f_centre + bandwidth/2:.3f} MHz)")
        axes[1].plot(t, rec_f, color="tomato", lw=1.3, ls="--",
                     label=f"Reconstruction  resid={result['resid_f']:.4f}")
        for mode in cfg.fundamental_modes:
            if mode in amps_f and mode in mw_f:
                contrib = amps_f[mode] * mw_f[mode]
                env     = np.abs(hilbert(contrib))
                axes[1].plot(t, contrib, lw=1.0, label=f"{mode}")
                axes[1].plot(t, env, lw=0.8, ls=":", alpha=0.7)
    axes[1].set_title(f"Fundamental band  |  cond = {result['cond_f']:.2e}")
    axes[1].set_ylabel("nm"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # ── Second harmonic band ──────────────────────────────────────────────
    amps_2f = result["amps_2f"]
    mw_2f   = result["mw_2f"]
    if amps_2f:
        rec_2f = sum(amps_2f[m] * mw_2f[m] for m in amps_2f)
        axes[2].plot(t, s_2f,  color="steelblue", lw=0.8, alpha=0.6,
                     label=f"Filtered 2nd harmonic band "
                           f"({2*f_centre - bandwidth/2:.3f}–"
                           f"{2*f_centre + bandwidth/2:.3f} MHz)")
        axes[2].plot(t, rec_2f, color="tomato", lw=1.3, ls="--",
                     label=f"Reconstruction  resid={result['resid_2f']:.4f}")
        for mode in cfg.second_harmonic_modes:
            if mode in amps_2f and mode in mw_2f:
                contrib = amps_2f[mode] * mw_2f[mode]
                env     = np.abs(hilbert(contrib))
                axes[2].plot(t, contrib, lw=1.0, label=f"{mode}")
                axes[2].plot(t, env, lw=0.8, ls=":", alpha=0.7)
    axes[2].set_title(f"Second harmonic band  |  cond = {result['cond_2f']:.2e}")
    axes[2].set_ylabel("nm"); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    # ── Amplitude bar chart — all solved modes, coloured by band ─────────
    peaks_f  = result["peaks_f"]
    peaks_2f = result["peaks_2f"]
    f_modes  = sorted(peaks_f.keys())
    h_modes  = sorted(peaks_2f.keys())
    all_modes_plot = f_modes + h_modes
    vals   = [peaks_f[m]  for m in f_modes] + [peaks_2f[m] for m in h_modes]
    colors = ["steelblue"] * len(f_modes) + ["tomato"] * len(h_modes)

    bars = axes[3].bar(all_modes_plot, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        axes[3].text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                     f"{v:.5f}", ha="center", va="bottom", fontsize=8, rotation=45)
    axes[3].set_title("Envelope peak amplitudes — all solved modes  "
                      "(blue = fundamental band, red = 2nd harmonic band)")
    axes[3].set_ylabel("|A| (nm)"); axes[3].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("plots/two_band_decomposition.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/two_band_decomposition.png")
    plt.show()

    # ── Verification plot (if GT available) ───────────────────────────────
    verify = result.get("verify", {})
    if verify:
        modes_v = sorted(verify.keys())
        gt_v    = [verify[m]["gt_peak"]  for m in modes_v]
        rec_v   = [verify[m]["rec_peak"] for m in modes_v]
        err_v   = [verify[m]["err_pct"]  for m in modes_v]

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Verification against ground truth", fontsize=12)

        x = np.arange(len(modes_v)); w = 0.38
        axes2[0].bar(x - w/2, gt_v,  w, label="GT peak",   color="steelblue", alpha=0.85)
        axes2[0].bar(x + w/2, rec_v, w, label="Recovered", color="tomato",    alpha=0.85)
        axes2[0].set_xticks(x); axes2[0].set_xticklabels(modes_v, rotation=45)
        axes2[0].set_ylabel("nm"); axes2[0].legend()
        axes2[0].set_title("GT vs Recovered envelope peak amplitudes")
        axes2[0].grid(True, alpha=0.3, axis="y")

        bar_colors = ["tomato" if e > 50 else "steelblue" for e in err_v]
        axes2[1].bar(modes_v, err_v, color=bar_colors, edgecolor="white")
        axes2[1].axhline(50, color="tomato", lw=1.5, ls="--", label="50% error")
        axes2[1].set_title("Amplitude error %  (red = >50% error)")
        axes2[1].set_ylabel("Error (%)"); axes2[1].legend()
        axes2[1].tick_params(axis="x", rotation=45)
        axes2[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("plots/verification.png", dpi=200, bbox_inches="tight")
        print("  Saved: plots/verification.png")
        plt.show()


def plot_repeatability(all_results, cfg):
    """
    Plot the distribution of extracted amplitudes across repetitions.
    Call this after running extract_amplitudes() in a loop.
    """
    # Collect all modes that appeared in any repetition across both bands
    all_f_modes = sorted({m for r in all_results for m in r["peaks_f"]})
    all_h_modes = sorted({m for r in all_results for m in r["peaks_2f"]})
    all_modes   = all_f_modes + all_h_modes
    colors      = ["steelblue"] * len(all_f_modes) + ["tomato"] * len(all_h_modes)

    data_f = {m: [r["peaks_f"].get(m, np.nan) for r in all_results] for m in all_f_modes}
    data_h = {m: [r["peaks_2f"].get(m, np.nan) for r in all_results] for m in all_h_modes}
    data   = {**data_f, **data_h}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Amplitude Repeatability Across Repetitions", fontsize=12)

    means  = [np.nanmean(data[m]) for m in all_modes]
    stds   = [np.nanstd(data[m])  for m in all_modes]

    axes[0].bar(all_modes, means, yerr=stds, color=colors,
                capsize=5, edgecolor="white")
    axes[0].set_title("Mean ± std across repetitions")
    axes[0].set_ylabel("|A| (nm)"); axes[0].grid(True, alpha=0.3, axis="y")

    # Coefficient of variation — your primary repeatability metric
    cvs = [np.nanstd(data[m]) / (np.nanmean(data[m]) + 1e-12) * 100
           for m in all_modes]
    axes[1].bar(all_modes, cvs, color=colors, edgecolor="white")
    axes[1].axhline(5,  color="green",  lw=1.5, ls="--", label="5%")
    axes[1].axhline(10, color="orange", lw=1.5, ls="--", label="10%")
    axes[1].axhline(20, color="tomato", lw=1.5, ls="--", label="20%")
    axes[1].set_title("Coefficient of Variation (%)  — lower = more repeatable")
    axes[1].set_ylabel("CV (%)"); axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("plots/repeatability.png", dpi=200, bbox_inches="tight")
    print("  Saved: plots/repeatability.png")
    plt.show()

    # Print summary table
    print("\n  ── Repeatability Summary ────────────────────────────────────────")
    print(f"  {'Mode':<8} {'Band':<16} {'Mean (nm)':>12} {'Std (nm)':>12} "
          f"{'CV (%)':>10} {'N':>5}")
    print("  " + "─" * 66)
    for m, mean, std, cv in zip(all_modes, means, stds, cvs):
        band = "fundamental" if m in all_f_modes else "2nd harmonic"
        n    = sum(1 for v in data[m] if not np.isnan(v))
        print(f"  {m:<8} {band:<16} {mean:>12.6f} {std:>12.6f} {cv:>10.2f} {n:>5}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  LAMB WAVE AMPLITUDE EXTRACTION  —  Two-band approach")
    print("=" * 70)

    # ── Load signal and dispersion ─────────────────────────────────────────
    print("\n[0] Loading signal...")
    t, s, exc, fs, dt, gt = load_signal(C)

    print("\n[0b] Loading dispersion curves...")
    disp = load_dispersion(C)
    print(f"  {len(disp)} modes loaded: {sorted(disp.keys())}")

    # ── Detect frequency bands ─────────────────────────────────────────────
    print("\n[1] Detecting excitation band...")
    f_centre = C.f_centre_MHz   if C.f_centre_MHz  is not None else None
    bandwidth = C.bandwidth_MHz if C.bandwidth_MHz is not None else None
    if f_centre is None or bandwidth is None:
        f_centre, bandwidth = detect_excitation_band(exc, fs)

    # ── Single file extraction ─────────────────────────────────────────────
    print("\n[2] Extracting amplitudes (single file)...")
    result = extract_amplitudes(C.signal_file, C, disp, f_centre, bandwidth)

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\n[3] Plotting...")
    plot_results(result, C, f_centre, bandwidth)

    # ── Example repeatability loop ────────────────────────────────────────
    # Uncomment and adapt this block when you have your 10 repetition files.
    #
    # repetition_files = [
    #     r"Data/signal_rep_01.xlsx",
    #     r"Data/signal_rep_02.xlsx",
    #     ...
    # ]
    # all_results = []
    # for i, fpath in enumerate(repetition_files):
    #     print(f"\n  Repetition {i+1}/{len(repetition_files)}: {fpath}")
    #     C.signal_file = fpath
    #     r = extract_amplitudes(fpath, C, disp, f_centre, bandwidth, verbose=False)
    #     all_results.append(r)
    #
    # plot_repeatability(all_results, C)

    print("\n" + "=" * 70 + "\n  DONE\n" + "=" * 70)