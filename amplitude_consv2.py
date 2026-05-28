"""
Amplitude Conservation Analysis
================================
Evaluates how well three time-frequency transforms (HHT, STFT, CWT/Wavelet)
preserve the S2 and S4 Lamb-wave mode amplitudes, and how accurately they
recover the nonlinearity parameter beta.

Pipeline per propagation distance
----------------------------------
1. Load base-frequency and harmonic datasets.
2. Synthesise a composite signal with a known beta.
3. Find the *clean* (noise-free) reference amplitudes of S2 and S4.
4. Apply each transform → reconstruct base-band and harmonic-band signals.
5. Locate mode peaks via time-of-arrival envelopes (ToA).
6. Compute amplitude loss (%) and beta error for each transform.
"""

import os
import sys

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocess
import decomp as d
from transforms import Hilbert_Huang_processing
from transforms import SST_v2_processing

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Lamb-wave modes to include in the synthesised signal --------------------
MODES_BASE      = ["S2 Propagated signal (nm)",
                   "A1 Propagated signal (nm)",
                   "A4 Propagated signal (nm)"]
MODES_HARMONIC  = ["S2 Propagated signal (nm)",
                   "A1 Propagated signal (nm)",
                   "A4 Propagated signal (nm)",
                   "S4 Propagated signal (nm)"]

# --- Mode column names passed to preprocess.create_signal -------------------
#     base_mode   : column in the base dataset used for amplitude scaling (A1)
#     second_mode : column in the harmonic dataset used for amplitude scaling (A2)
BASE_MODE   = "S2 Propagated signal (nm)"   # fundamental-frequency reference mode
SECOND_MODE = "S4 Propagated signal (nm)"   # second-harmonic reference mode

# --- Physics / simulation parameters ----------------------------------------
BETA            = 10      # predefined nonlinearity parameter
DISTANCE_M      = 0.2     # source-to-sensor distance in metres
NOISE_LEVEL     = 0.0     # relative noise amplitude (0 = none, 1.5 = 150 %)
NOISE_LEVELS    = [0.0]   # sweep list used in Main()

# --- Wave-number calculation constants  (1.33 MHz fundamental) ---------------
F_FUND_HZ   = 1.33e6          # fundamental frequency  [Hz]  — was wrongly 10e6
C_PHASE_MPS = 8303.885       # phase velocity        [m/s]  — was wrongly 10e3*8.3…

# --- Time-of-arrival stamps (µs) per distance --------------------------------
#     base signal @ 1.33 MHz,  harmonic signal @ 2.66 MHz
TIME_STAMPS = {
    200: {
        "base":     {"S2": 48.7581,  "A1": 71.5683,  "A4": 151.605},
        "harmonic": {"A1": 66.3568,  "S4": 71.6575,  "S2": 75.8788,  "A4": 93.6327},
    },
    250: {
        "base":     {"S2": 60.9477,  "A1": 89.4603,  "A4": 189.506},
        "harmonic": {"A1": 82.946,   "S4": 89.5719,  "S2": 94.8485,  "A4": 117.041},
    },
    300: {
        "base":     {"S2": 73.1372,  "A1": 107.352,  "A4": 227.407},
        "harmonic": {"A1": 99.5352,  "S4": 107.486,  "S2": 113.818,  "A4": 140.449},
    },
    350: {
        "base":     {"S2": 85.3267,  "A1": 125.244,  "A4": 265.309},
        "harmonic": {"A1": 116.124,  "S4": 125.401,  "S2": 132.788,  "A4": 163.857},
    },
}

# --- Dataset file paths ------------------------------------------------------
DATASETS = {
    200: {
        "base":     "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx",
        "harmonic": "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx",
    },
    250: {
        "base":     "Data/In-plane_TemporalResponse@7.9866MHzmm@250mm.xlsx",
        "harmonic": "Data/In-plane_A2_TemporalResponse@15.963MHzmm@250mm.xlsx",
    },
    300: {
        "base":     "Data/In-plane_TemporalResponse@7.9866MHzmm@300mm.xlsx",
        "harmonic": "Data/In-plane_A2_TemporalResponse@15.963MHzmm@300mm.xlsx",
    },
    350: {
        "base":     "Data/In-plane_TemporalResponse@7.9866MHzmm@350mm.xlsx",
        "harmonic": "Data/In-plane_A2_TemporalResponse@15.963MHzmm@350mm.xlsx",
    },
}

# --- Per-method, per-noise-level envelope smoothing iterations ---------------
#     Tuple = (n_smoothing_base, n_smoothing_harmonic)
SMOOTHING = {
    "hht": {
        0.0:  (0, 0), 0.25: (0, 0), 0.5:  (2, 2),
        0.75: (3, 3), 1.0:  (4, 4), 1.25: (5, 5), 1.5: (6, 6),
    },
    "stft": {
        0.0:  (0, 0), 0.25: (1, 1), 0.5:  (2, 2),
        0.75: (3, 3), 1.0:  (4, 4), 1.25: (5, 5), 1.5: (6, 6),
    },
    "wt": {
        0.0:  (0, 0), 0.25: (0, 0), 0.5:  (1, 1),
        0.75: (2, 2), 1.0:  (3, 3), 1.25: (4, 4), 1.5: (5, 5),
    },
}

# ==============================================================================
# TRANSFORM FUNCTIONS
# ==============================================================================

def apply_hht(t, signal):
    """
    Decompose *signal* with EMD, then band-pass each IMF to isolate
    the base-frequency and harmonic bands.

    Returns
    -------
    recon_base, recon_harmonic : ndarray
    """
    F_MIN_BASE      = 1_100_000   # Hz
    F_MAX_BASE      = 1_500_000
    F_MIN_HARMONIC  = 2_300_000
    F_MAX_HARMONIC  = 2_900_000

    dt = np.mean(np.diff(t))
    fs = (1.0 / dt) * 1e6   # convert from µs⁻¹ → Hz

    imfs, _residue = Hilbert_Huang_processing.emd(signal)

    recon_base      = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, F_MIN_BASE,     F_MAX_BASE)
    recon_harmonic  = Hilbert_Huang_processing.bandpass_hilbert(imfs, fs, F_MIN_HARMONIC, F_MAX_HARMONIC)

    return recon_base, recon_harmonic


def apply_stft(t, signal):
    """
    Reconstruct base and harmonic bands from an STFT representation.

    Returns
    -------
    recon_base, recon_harmonic : ndarray
    """
    F_MIN_DISPLAY   = 1.0e6    # Hz — TF display range (not used for reconstruction)
    F_MAX_DISPLAY   = 4.5e6

    BAND_MIN_BASE       = 1_100_000
    BAND_MAX_BASE       = 1_500_000
    BAND_MIN_HARMONIC   = 2_300_000
    BAND_MAX_HARMONIC   = 2_900_000

    WIN_LEN = 128   # samples
    HOP_LEN = 2
    N_FFT   = 512

    recon_base = SST_v2_processing.reconstruct_band_stft(
        t, signal,
        band_min=BAND_MIN_BASE,     band_max=BAND_MAX_BASE,
        fmin=F_MIN_DISPLAY,         fmax=F_MAX_DISPLAY,
        win_len=WIN_LEN, hop_len=HOP_LEN, n_fft=N_FFT,
    )

    recon_harmonic = SST_v2_processing.reconstruct_band_stft(
        t, signal,
        band_min=BAND_MIN_HARMONIC, band_max=BAND_MAX_HARMONIC,
        fmin=F_MIN_DISPLAY,         fmax=F_MAX_DISPLAY,
        win_len=WIN_LEN, hop_len=HOP_LEN, n_fft=N_FFT,
    )

    return recon_base, recon_harmonic


def apply_wavelet(t, signal):
    """
    Reconstruct base and harmonic bands using a CWT (complex Morlet wavelet).

    Returns
    -------
    recon_base, recon_harmonic : ndarray
    """
    WAVELET         = "cmor3.0-1.0"
    F_MIN_DISPLAY   = 1.0e6
    F_MAX_DISPLAY   = 4.5e6
    N_FREQS         = 400

    BAND_MIN_BASE       = 1_100_000
    BAND_MAX_BASE       = 1_500_000
    BAND_MIN_HARMONIC   = 2_300_000
    BAND_MAX_HARMONIC   = 2_900_000

    recon_base = SST_v2_processing.reconstruct_band_cwt(
        t, signal,
        band_min=BAND_MIN_BASE,     band_max=BAND_MAX_BASE,
        wavelet=WAVELET,
        fmin=F_MIN_DISPLAY,         fmax=F_MAX_DISPLAY,
        n_freqs=N_FREQS,
    )

    recon_harmonic = SST_v2_processing.reconstruct_band_cwt(
        t, signal,
        band_min=BAND_MIN_HARMONIC, band_max=BAND_MAX_HARMONIC,
        wavelet=WAVELET,
        fmin=F_MIN_DISPLAY,         fmax=F_MAX_DISPLAY,
        n_freqs=N_FREQS,
    )

    return recon_base, recon_harmonic

# ==============================================================================
# TIME-OF-ARRIVAL  (envelope peak → mode amplitude)
# ==============================================================================

def _smooth_envelope(signal, n_sift_iters):
    """
    Compute the upper amplitude envelope of *signal* and optionally
    refine it with *n_sift_iters* sifting passes.

    BUG FIX: `scipy.signal.envelope` does not exist.
    The correct approach is the magnitude of the analytic signal via Hilbert.
    """
    # Analytic signal → instantaneous amplitude envelope
    env = np.abs(scipy.signal.hilbert(signal))

    for _ in range(n_sift_iters):
        env, _lower, _mean = d.sift(env)

    return env


def locate_modes(recon_base, recon_harmonic, t,
                 time_stamp_base, time_stamp_harmonic,
                 noise_level, method):
    """
    Extract per-mode amplitude peaks using time-of-arrival windows.

    Parameters
    ----------
    recon_base, recon_harmonic : ndarray
        Band-reconstructed signals for the base and harmonic bands.
    t : ndarray
        Time axis in µs.
    time_stamp_base, time_stamp_harmonic : dict
        Expected arrival times (µs) keyed by mode name.
    noise_level : float
        Current noise level (selects smoothing preset).
    method : str
        Transform name – one of ``'hht'``, ``'stft'``, ``'wt'``.

    Returns
    -------
    result_base, result_harmonic : dict
        Mode-keyed dicts with ``'peak_value'`` (and optionally other info
        returned by ``d.align_to_envelope_with_time``).
    """
    if method not in SMOOTHING:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(SMOOTHING)}")
    if noise_level not in SMOOTHING[method]:
        raise ValueError(f"Noise level {noise_level} has no smoothing preset for '{method}'.")

    k_base, k_harmonic = SMOOTHING[method][noise_level]

    # --- base band ---
    env_base = _smooth_envelope(recon_base, k_base)
    plt.plot(t, recon_base,  label="recon_base")
    plt.plot(t, env_base,    label="envelope")
    plt.title(f"Base band envelope ({method})")
    plt.legend()
    plt.show()

    result_base = d.align_to_envelope_with_time(env_base, t, time_stamp_base)

    # --- harmonic band ---
    env_harmonic = _smooth_envelope(recon_harmonic, k_harmonic)
    plt.plot(t, recon_harmonic, label="recon_harmonic")
    plt.plot(t, env_harmonic,   label="envelope")
    plt.title(f"Harmonic band envelope ({method})")
    plt.legend()
    plt.show()

    result_harmonic = d.align_to_envelope_with_time(env_harmonic, t, time_stamp_harmonic)

    return result_base, result_harmonic

# ==============================================================================
# AMPLITUDE & BETA UTILITIES
# ==============================================================================

def extract_s2_s4_amplitudes(result_base, result_harmonic):
    """Return the peak amplitudes of the S2 (base) and S4 (harmonic) modes."""
    s2_peak = result_base["S2"]["peak_value"]
    s4_peak = result_harmonic["S4"]["peak_value"]
    return s2_peak, s4_peak


def amplitude_loss_pct(amp_s2_ref, amp_s4_ref, amp_s2_recon, amp_s4_recon):
    """
    Percentage amplitude loss after reconstruction relative to the
    clean reference signal.

    Returns
    -------
    s2_loss_pct, s4_loss_pct : float
        Positive → amplitude decreased; negative → amplitude increased.
    """
    s2_loss = (amp_s2_ref - amp_s2_recon) / amp_s2_ref * 100.0
    s4_loss = (amp_s4_ref - amp_s4_recon) / amp_s4_ref * 100.0
    return s2_loss, s4_loss


def beta_error(amp_s2, amp_s4, beta_ref=10, distance=DISTANCE_M):
    """
    Compute the nonlinearity parameter beta from the reconstructed mode
    amplitudes, then return the absolute error compared to *beta_ref*.

    Formula  (from acoustic nonlinearity theory):
        beta = (A_S4 / A_S2²) * (8 / (x · k²))

    where x is the propagation distance and k = 2π·f / c_phase.

    BUG FIX: The original used `10e6` (= 1 × 10⁷) and `10e3` (= 1 × 10⁴).
    Both should be `1e6` and `1e3` respectively so that k is in rad/m.
    """
    k = (2 * np.pi * F_FUND_HZ) / C_PHASE_MPS   # wave number [rad/m]
    beta_computed = (amp_s4*1e-9 / (amp_s2*1e-9) ** 2) * (8.0 / (distance * k ** 2))
    return abs(beta_ref - beta_computed)


# ==============================================================================
# REFERENCE AMPLITUDE HELPERS
# ==============================================================================

def _reference_amplitude(data_base, data_harmonic,
                          modes_base_sel, modes_harmonic_sel):
    """
    Synthesise a *clean* (noise-free) signal containing only the specified
    modes, then return its peak amplitude.

    Using noise_level=0 here is intentional: we want a deterministic
    reference value that is independent of the random noise realisation.
    """
    t, signal, _ = preprocess.create_signal(
        data_base, data_harmonic,
        BETA, 0.0,                   # beta, noise_level=0 → clean reference
        BASE_MODE, SECOND_MODE,
        modes_base_sel, modes_harmonic_sel,
        DISTANCE_M,
    )
    return float(np.max(signal))


def reference_amplitude_s2(data_base, data_harmonic):
    """Clean peak amplitude of the S2 mode alone."""
    return _reference_amplitude(
        data_base, data_harmonic,
        modes_base_sel=["S2 Propagated signal (nm)"],
        modes_harmonic_sel=[],
    )


def reference_amplitude_s4(data_base, data_harmonic):
    """Clean peak amplitude of the S4 mode alone."""
    return _reference_amplitude(
        data_base, data_harmonic,
        modes_base_sel=[],
        modes_harmonic_sel=["S4 Propagated signal (nm)"],
    )

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_analysis(length_mm, beta_value=BETA):
    """
    Full amplitude-conservation analysis for a given propagation distance.

    Parameters
    ----------
    length_mm : int
        Propagation distance in mm. Must be one of 200, 250, 300, 350.
    beta_value : float
        Nonlinearity parameter used to synthesise the signal.

    Returns
    -------
    results : dict with keys
        'amp_loss_pct'  → dict mapping method → list of [s2_loss, s4_loss] per noise level
        'beta_error'    → dict mapping method → list of beta errors per noise level
    """
    if length_mm not in DATASETS:
        raise ValueError(f"length_mm must be one of {sorted(DATASETS)}; got {length_mm}.")

    # --- Load data ---
    dataset_base     = DATASETS[length_mm]["base"]
    dataset_harmonic = DATASETS[length_mm]["harmonic"]
    data_base     = preprocess.get_data(dataset_base)
    data_harmonic = preprocess.get_data(dataset_harmonic)

    time_stamp_base     = TIME_STAMPS[length_mm]["base"]
    time_stamp_harmonic = TIME_STAMPS[length_mm]["harmonic"]

    # --- Clean reference amplitudes (noise-free) ---
    amp_s2_ref = reference_amplitude_s2(data_base, data_harmonic)
    amp_s4_ref = reference_amplitude_s4(data_base, data_harmonic)

    # --- Accumulators ---
    results = {
        "amp_loss_pct": {"stft": [], "hht": [], "wt": []},
        "beta_error":   {"stft": [], "hht": [], "wt": []},
    }

    for noise in NOISE_LEVELS:
        # Synthesise composite signal at current noise level
        t, signal, _ = preprocess.create_signal(
            data_base, data_harmonic,
            beta_value, noise,
            BASE_MODE, SECOND_MODE,
            MODES_BASE, MODES_HARMONIC,
            DISTANCE_M,
        )

        # --- Apply transforms ---
        recon_base_stft, recon_harmonic_stft = apply_stft(t, signal)
        recon_base_hht,  recon_harmonic_hht  = apply_hht(t, signal)
        recon_base_wt,   recon_harmonic_wt   = apply_wavelet(t, signal)

        # --- Locate modes via ToA envelopes ---
        result_base_stft, result_harmonic_stft = locate_modes(
            recon_base_stft, recon_harmonic_stft, t,
            time_stamp_base, time_stamp_harmonic, noise, "stft")

        result_base_hht, result_harmonic_hht = locate_modes(
            recon_base_hht, recon_harmonic_hht, t,
            time_stamp_base, time_stamp_harmonic, noise, "hht")

        result_base_wt, result_harmonic_wt = locate_modes(
            recon_base_wt, recon_harmonic_wt, t,
            time_stamp_base, time_stamp_harmonic, noise, "wt")

        # --- Extract S2 / S4 peak amplitudes ---
        amp_s2_stft, amp_s4_stft = extract_s2_s4_amplitudes(result_base_stft, result_harmonic_stft)
        amp_s2_hht,  amp_s4_hht  = extract_s2_s4_amplitudes(result_base_hht,  result_harmonic_hht)
        amp_s2_wt,   amp_s4_wt   = extract_s2_s4_amplitudes(result_base_wt,   result_harmonic_wt)

        # --- Amplitude loss (%) ---
        results["amp_loss_pct"]["stft"].append(
            list(amplitude_loss_pct(amp_s2_ref, amp_s4_ref, amp_s2_stft, amp_s4_stft)))
        results["amp_loss_pct"]["hht"].append(
            list(amplitude_loss_pct(amp_s2_ref, amp_s4_ref, amp_s2_hht,  amp_s4_hht)))
        results["amp_loss_pct"]["wt"].append(
            list(amplitude_loss_pct(amp_s2_ref, amp_s4_ref, amp_s2_wt,   amp_s4_wt)))

        # --- Beta error ---
        results["beta_error"]["stft"].append(beta_error(amp_s2_stft, amp_s4_stft))
        results["beta_error"]["hht"].append( beta_error(amp_s2_hht,  amp_s4_hht))
        results["beta_error"]["wt"].append(  beta_error(amp_s2_wt,   amp_s4_wt))

    # --- Print summary ---
    print(f"\n=== Results for {length_mm} mm ===")
    for method in ("stft", "hht", "wt"):
        print(f"  Amplitude loss % [{method}]: {results['amp_loss_pct'][method]}")
    print()
    for method in ("stft", "hht", "wt"):
        print(f"  Beta error       [{method}]: {results['beta_error'][method]}")

    return results


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Quick preview of the full composite signal at 200 mm
    _data_base     = preprocess.get_data(DATASETS[200]["base"])
    _data_harmonic = preprocess.get_data(DATASETS[200]["harmonic"])
    _t, _signal, _ = preprocess.create_signal(
        _data_base, _data_harmonic,
        BETA, NOISE_LEVEL,
        BASE_MODE, SECOND_MODE,
        MODES_BASE, MODES_HARMONIC,
        DISTANCE_M,
    )
    plt.plot(_t, _signal)
    plt.title("Initial waveform: base (A1, S2, A4) + harmonic (A1, S4, S2, A4)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude (nm)")
    plt.tight_layout()
    plt.show()

    # Run main analysis
    run_analysis(length_mm=200)