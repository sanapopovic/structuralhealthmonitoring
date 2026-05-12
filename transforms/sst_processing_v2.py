"""
sst_processing.py
=================
Synchrosqueezing Transform (SST) module for ultrasonic Lamb wave analysis.

Provides SST-sharpened spectrograms and band reconstructions for three base transforms:
  - STFT  → STFT-SST  (reassigned short-time Fourier)
  - CWT   → CWT-SST   (reassigned continuous wavelet)
  - HHT   → post-HHT SST (ridge-sharpened Hilbert spectrum)

All public functions share the same call signature as their counterparts in
wavelet_processing.py so they are drop-in replacements in despair.py / The_Function().

Units convention (matches the rest of the project):
  t_us  : time axis in MICROSECONDS
  freqs : returned / plotted in MHz
  band_min / band_max : in Hz  (like wavelet_processing.py)
  fmin / fmax         : in Hz
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import stft, istft, hilbert

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mkdir(folder: str) -> None:
    os.makedirs(folder, exist_ok=True)


def _save_fig(folder: str, name: str, dpi: int = 300) -> None:
    _safe_mkdir(folder)
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=dpi)
    plt.close()
    print(f"[SST] Plot saved → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STFT-SST
# ─────────────────────────────────────────────────────────────────────────────

def stft_sst(
    t_us,
    sig,
    fmin: float = 1.0e6,
    fmax: float = 4.5e6,
    nperseg: int = 256,
    noverlap: int | None = None,
    gamma: float = 1e-6,
    name: str = "stft_sst",
    plot: bool = True,
    folder: str = "plots",
):
    """
    Short-Time Fourier Transform followed by Synchrosqueezing reassignment.

    Parameters
    ----------
    t_us     : 1-D array, time in µs
    sig      : 1-D array, signal
    fmin/fmax: frequency window for display and reassignment (Hz)
    nperseg  : STFT window length (samples)
    noverlap : STFT overlap (default: nperseg - 1 for max resolution)
    gamma    : threshold — coefficients below gamma * max(|S|) are ignored
               during reassignment (prevents noise blow-up)
    name     : filename stem for saved figure
    plot     : whether to save a figure
    folder   : output folder for figures

    Returns
    -------
    t_out    : time axis (µs), length = number of STFT frames
    freqs_hz : frequency axis (Hz), length = n_freqs bins between fmin/fmax
    Ts       : 2-D reassigned (squeezed) amplitude array [n_freqs × n_frames]
    """
    sig = np.asarray(sig).squeeze()
    t_s = np.asarray(t_us).squeeze() * 1e-6          # µs → s
    dt = float(np.mean(np.diff(t_s)))
    fs = 1.0 / dt

    if noverlap is None:
        noverlap = nperseg - 1

    # ── STFT ──────────────────────────────────────────────────────────────────
    f_stft, t_stft, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                window="hann", return_onesided=True)
    # f_stft in Hz, t_stft in s

    # ── Instantaneous frequency via finite-difference phase derivative ────────
    # phase(t) differentiated along time axis → ω_hat(f, t) in rad/s
    phase = np.angle(Zxx)                             # [n_freqs × n_frames]
    dphase = np.diff(np.unwrap(phase, axis=1), axis=1, prepend=phase[:, :1])
    dt_stft = float(np.mean(np.diff(t_stft)))
    omega_hat = (dphase / dt_stft) / (2.0 * np.pi)   # Hz (reassigned freq)

    # ── Reassignment / synchrosqueezing ───────────────────────────────────────
    # Target frequency grid (uniform, Hz)
    freqs_hz = np.linspace(fmin, fmax, nperseg // 2 + 1)
    df_bin = freqs_hz[1] - freqs_hz[0]

    amp = np.abs(Zxx)
    threshold = gamma * amp.max()

    Ts = np.zeros((len(freqs_hz), len(t_stft)), dtype=float)

    for k in range(amp.shape[0]):
        for m in range(amp.shape[1]):
            if amp[k, m] < threshold:
                continue
            omega_km = omega_hat[k, m]
            # Only reassign if within target band
            if omega_km < fmin or omega_km > fmax:
                continue
            l = int(round((omega_km - fmin) / df_bin))
            if 0 <= l < len(freqs_hz):
                Ts[l, m] += amp[k, m]

    t_out_us = t_stft * 1e6   # s → µs

    if plot:
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t_out_us, freqs_hz / 1e6, Ts, shading="gouraud")
        plt.xlabel("Time [µs]")
        plt.ylabel("Frequency [MHz]")
        plt.title(name)
        plt.colorbar(label="Reassigned amplitude")
        plt.ylim(fmin / 1e6, fmax / 1e6)
        plt.tight_layout()
        _save_fig(folder, name)

    return t_out_us, freqs_hz, Ts


def reconstruct_band_stft_sst(
    t_us,
    sig,
    band_min: float,
    band_max: float,
    fmin: float = 1.0e6,
    fmax: float = 4.5e6,
    nperseg: int = 256,
    noverlap: int | None = None,
    gamma: float = 1e-6,
):
    """
    Band-pass reconstruction in the STFT domain using the SST mask.

    The mask is derived from the reassigned frequency map: a TF coefficient
    (k, m) is kept only when its reassigned frequency ω̂(k,m) falls within
    [band_min, band_max].  This sharpens the band edges considerably compared
    to a simple rectangular STFT filter.

    Returns
    -------
    reconstructed : 1-D array, same length as sig
    """
    sig = np.asarray(sig).squeeze()
    t_s = np.asarray(t_us).squeeze() * 1e-6
    dt = float(np.mean(np.diff(t_s)))
    fs = 1.0 / dt

    if noverlap is None:
        noverlap = nperseg - 1

    f_stft, t_stft, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                window="hann", return_onesided=True)

    phase = np.angle(Zxx)
    dphase = np.diff(np.unwrap(phase, axis=1), axis=1, prepend=phase[:, :1])
    dt_stft = float(np.mean(np.diff(t_stft)))
    omega_hat = (dphase / dt_stft) / (2.0 * np.pi)

    amp = np.abs(Zxx)
    threshold = gamma * amp.max()

    mask = (
        (omega_hat >= band_min) &
        (omega_hat <= band_max) &
        (amp >= threshold)
    )

    Zxx_filtered = Zxx * mask

    _, reconstructed = istft(Zxx_filtered, fs=fs, nperseg=nperseg,
                             noverlap=noverlap, window="hann")

    # Match output length to input
    n = len(sig)
    if len(reconstructed) >= n:
        reconstructed = reconstructed[:n]
    else:
        reconstructed = np.pad(reconstructed, (0, n - len(reconstructed)))

    return reconstructed


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CWT-SST
# ─────────────────────────────────────────────────────────────────────────────

def cwt_sst(
    t_us,
    sig,
    wavelet: str = "cmor3.0-1.0",
    fmin: float = 1.0e6,
    fmax: float = 4.5e6,
    n_freqs: int = 400,
    gamma: float = 1e-6,
    name: str = "cwt_sst",
    plot: bool = True,
    folder: str = "plots",
):
    """
    Continuous Wavelet Transform followed by Synchrosqueezing reassignment.

    The CWT coefficient W(a, b) is reassigned to its instantaneous frequency
    ω̂(a, b) = -Im[ ∂_b W(a, b) / W(a, b) ] / (2π)
    estimated here via finite-difference phase derivative (no extra transform
    needed — works for any pywt-compatible complex wavelet).

    Parameters
    ----------
    t_us      : time in µs
    sig       : signal
    wavelet   : pywt complex wavelet name (e.g. 'cmor3.0-1.0', 'cgau2')
    fmin/fmax : analysis + target frequency range (Hz)
    n_freqs   : number of frequency bins in the squeezed output
    gamma     : soft threshold fraction of max |CWT|

    Returns
    -------
    t_us      : time axis (µs, same as input)
    freqs_hz  : reassigned frequency axis (Hz)
    Ts        : 2-D synchrosqueezed amplitude [n_freqs × n_time]
    """
    import pywt

    sig = np.asarray(sig).squeeze()
    t_s = np.asarray(t_us).squeeze() * 1e-6
    dt = float(np.mean(np.diff(t_s)))

    # Scale ↔ frequency mapping
    fc = pywt.central_frequency(wavelet)
    freqs_target = np.linspace(fmin, fmax, n_freqs)   # Hz
    scales = fc / (freqs_target * dt)                  # descending

    # ── CWT ───────────────────────────────────────────────────────────────────
    cwtmatr, freqs = pywt.cwt(sig, scales, wavelet, sampling_period=dt)
    # cwtmatr : [n_freqs × n_time], complex

    # ── Instantaneous frequency (phase derivative, time axis) ─────────────────
    phase = np.angle(cwtmatr)
    dphase = np.diff(np.unwrap(phase, axis=1), axis=1, prepend=phase[:, :1])
    omega_hat = (dphase / dt) / (2.0 * np.pi)           # Hz  [n_freqs × n_time]

    # ── Synchrosqueezing ──────────────────────────────────────────────────────
    freqs_hz = np.linspace(fmin, fmax, n_freqs)
    df_bin = freqs_hz[1] - freqs_hz[0]
    amp = np.abs(cwtmatr)
    threshold = gamma * amp.max()

    Ts = np.zeros((n_freqs, cwtmatr.shape[1]), dtype=float)

    for k in range(amp.shape[0]):
        mask = amp[k] >= threshold
        omegas = omega_hat[k]
        indices = np.round((omegas - fmin) / df_bin).astype(int)
        valid = mask & (indices >= 0) & (indices < n_freqs)
        np.add.at(Ts, (indices[valid], np.where(valid)[0]), amp[k, valid])

    t_us_arr = np.asarray(t_us).squeeze()

    if plot:
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t_us_arr, freqs_hz / 1e6, Ts, shading="gouraud")
        plt.xlabel("Time [µs]")
        plt.ylabel("Frequency [MHz]")
        plt.title(name)
        plt.colorbar(label="Reassigned amplitude")
        plt.ylim(fmin / 1e6, fmax / 1e6)
        plt.tight_layout()
        _save_fig(folder, name)

    return t_us_arr, freqs_hz, Ts


def reconstruct_band_cwt_sst(
    t_us,
    sig,
    band_min: float,
    band_max: float,
    wavelet: str = "cmor3.0-1.0",
    fmin: float = 1.0e6,
    fmax: float = 4.5e6,
    n_freqs: int = 400,
    gamma: float = 1e-6,
):
    """
    CWT-SST band reconstruction.

    Same amplitude-preserving inversion as wavelet_processing.reconstruct_frequency_band,
    but the band mask is applied in the *reassigned* frequency space: a scale k
    at time m is kept only when ω̂(k, m) ∈ [band_min, band_max].
    This makes the reconstruction far more robust in multi-mode signals where
    modes overlap in scale space but diverge in instantaneous frequency.

    Returns
    -------
    reconstructed : 1-D real array
    """
    import pywt

    sig = np.asarray(sig).squeeze()
    t_s = np.asarray(t_us).squeeze() * 1e-6
    dt = float(np.mean(np.diff(t_s)))

    fc = pywt.central_frequency(wavelet)
    freqs_target = np.linspace(fmin, fmax, n_freqs)
    scales = fc / (freqs_target * dt)

    cwtmatr, freqs = pywt.cwt(sig, scales, wavelet, sampling_period=dt)

    phase = np.angle(cwtmatr)
    dphase = np.diff(np.unwrap(phase, axis=1), axis=1, prepend=phase[:, :1])
    omega_hat = (dphase / dt) / (2.0 * np.pi)

    amp = np.abs(cwtmatr)
    threshold = gamma * amp.max()

    mask = (
        (omega_hat >= band_min) &
        (omega_hat <= band_max) &
        (amp >= threshold)
    )

    cwt_band = cwtmatr * mask
    scales_band = scales[:, None]  # broadcast over time

    # Amplitude-preserving Morlet inversion (same formula as wavelet_processing.py)
    reconstructed = np.real(np.sum(cwt_band / (scales_band ** 2), axis=0))
    reconstructed *= np.mean(np.diff(np.log(scales)))

    return reconstructed


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HHT post-processing with SST-style ridge sharpening
# ─────────────────────────────────────────────────────────────────────────────
#
# True SST is not defined for HHT because HHT already produces IF curves
# (ridges) rather than a 2-D TF plane from which reassignment could depart.
# What we do instead:
#   1. Compute the standard Hilbert spectrum (H[f, t]) from IMFs.
#   2. Apply a soft Gaussian ridge-sharpening kernel along the frequency axis:
#      each IF curve is kept but redistributed into a narrow Gaussian bump
#      centred on the instantaneous frequency.  This is the closest analogue
#      to CWT-SST for the HHT world and is sometimes called "re-assigned
#      Hilbert spectrum" in the literature.
#
# This is a separate plotting / analysis helper; it does NOT replace
# Hilbert_Huang_processing.plot_hilbert_spectrum — call that first, then
# optionally call hht_reassigned_spectrum for the sharpened version.

def hht_reassigned_spectrum(
    inst_freq,
    inst_amp,
    t_us,
    fs_hz: float,
    fmin: float = 1.0e6,
    fmax: float = 4.5e6,
    f_bins: int = 400,
    t_bins: int = 600,
    sigma_hz: float = 2e4,
    name: str = "hht_sst",
    plot: bool = True,
    folder: str = "plots",
    log_amplitude: bool = False,
):
    """
    Ridge-sharpened (pseudo-SST) Hilbert spectrum.

    Parameters
    ----------
    inst_freq  : [n_imfs × n_time] instantaneous frequency array (Hz)
                 — from Hilbert_Huang_processing.hilbert_analysis
    inst_amp   : [n_imfs × n_time] instantaneous amplitude array
    t_us       : time axis in µs
    fs_hz      : sampling rate in Hz
    fmin/fmax  : frequency window (Hz)
    f_bins     : number of output frequency bins
    t_bins     : number of output time bins
    sigma_hz   : std of the Gaussian redistribution kernel (Hz)
                 smaller → sharper ridges;  set to 0 to get delta-function ridges
    name/plot/folder : figure controls
    log_amplitude: apply log10 to amplitude before plotting

    Returns
    -------
    T2d  : 2-D time array (µs)  [f_bins × t_bins]
    F2d  : 2-D freq array (MHz) [f_bins × t_bins]
    H    : reassigned Hilbert spectrum [f_bins × t_bins]
    """
    t_us = np.asarray(t_us).squeeze()
    inst_freq = np.asarray(inst_freq)
    inst_amp  = np.asarray(inst_amp)
    if inst_freq.ndim == 1:
        inst_freq = inst_freq[None, :]
        inst_amp  = inst_amp[None, :]

    # Build output grid
    f_axis = np.linspace(fmin, fmax, f_bins)          # Hz
    t_axis = np.linspace(t_us[0], t_us[-1], t_bins)   # µs

    # Interpolate each IMF onto output time grid
    from scipy.interpolate import interp1d

    H = np.zeros((f_bins, t_bins), dtype=float)

    df = f_axis[1] - f_axis[0]
    dt_out = t_axis[1] - t_axis[0]

    for imf_idx in range(inst_freq.shape[0]):
        # Interpolate IF and IA to output time axis
        if_interp = interp1d(t_us, inst_freq[imf_idx], kind="linear",
                             bounds_error=False, fill_value=0.0)(t_axis)
        ia_interp = interp1d(t_us, inst_amp[imf_idx],  kind="linear",
                             bounds_error=False, fill_value=0.0)(t_axis)

        for m in range(t_bins):
            f0  = if_interp[m]
            amp = ia_interp[m]
            if f0 < fmin or f0 > fmax or amp == 0:
                continue

            if sigma_hz == 0:
                k = int(round((f0 - fmin) / df))
                if 0 <= k < f_bins:
                    H[k, m] += amp
            else:
                # Gaussian smearing — vectorised over f_axis
                gauss = np.exp(-0.5 * ((f_axis - f0) / sigma_hz) ** 2)
                gauss /= (gauss.sum() + 1e-30)
                H[:, m] += amp * gauss

    if log_amplitude:
        H = np.log10(H + 1e-12)

    T2d, F2d = np.meshgrid(t_axis, f_axis / 1e6)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(T2d, F2d, H, shading="gouraud")
        plt.xlabel("Time [µs]")
        plt.ylabel("Frequency [MHz]")
        plt.title(name)
        plt.colorbar(label="Log amplitude" if log_amplitude else "Amplitude")
        plt.ylim(fmin / 1e6, fmax / 1e6)
        plt.tight_layout()
        _save_fig(folder, name)

    return T2d, F2d, H


def reconstruct_band_hht_sst(
    inst_freq,
    inst_amp,
    imfs,
    t_us,
    fs_hz: float,
    band_min: float,
    band_max: float,
    sigma_hz: float = 2e4,
):
    """
    Reconstruct a frequency band from IMFs using SST-style IF gating.

    Instead of a hard bandpass filter (which smears modes), each IMF is kept
    at time m only when its instantaneous frequency lies within
    [band_min - sigma_hz, band_max + sigma_hz].
    This gives a time-varying bandpass that tracks the actual mode.

    Parameters
    ----------
    inst_freq : [n_imfs × n_time] IF array (Hz)
    inst_amp  : [n_imfs × n_time] IA array
    imfs      : [n_imfs × n_time] raw IMF matrix
    t_us      : time axis (µs)
    fs_hz     : sampling rate (Hz)
    band_min/band_max : band edges (Hz)
    sigma_hz  : soft gate half-width (Hz); 0 = hard gate

    Returns
    -------
    reconstructed : 1-D real array, same length as imfs.shape[1]
    """
    inst_freq = np.asarray(inst_freq)
    inst_amp  = np.asarray(inst_amp)
    imfs      = np.asarray(imfs)
    if inst_freq.ndim == 1:
        inst_freq = inst_freq[None, :]
        inst_amp  = inst_amp[None, :]
        imfs      = imfs[None, :]

    n_imfs, n_time = imfs.shape
    reconstructed = np.zeros(n_time, dtype=float)

    for k in range(n_imfs):
        f0 = inst_freq[k]   # shape (n_time,)

        if sigma_hz == 0:
            gate = ((f0 >= band_min) & (f0 <= band_max)).astype(float)
        else:
            # Soft sigmoid ramp at each edge
            gate = (
                _sigmoid_gate(f0, band_min, sigma_hz) *
                _sigmoid_gate(band_max - (f0 - band_min), band_max, sigma_hz)
            )
            # simpler version: Gaussian centred at band midpoint
            f_mid   = 0.5 * (band_min + band_max)
            f_width = 0.5 * (band_max - band_min) + sigma_hz
            gate    = np.exp(-0.5 * ((f0 - f_mid) / f_width) ** 2)
            gate   *= (f0 >= band_min - 2 * sigma_hz) & (f0 <= band_max + 2 * sigma_hz)

        reconstructed += imfs[k] * gate

    return reconstructed


def _sigmoid_gate(x, centre, width):
    return 1.0 / (1.0 + np.exp(-(x - centre) / (width / 5.0)))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Convenience: all-in-one SST spectrogram (drop-in for despair.py)
# ─────────────────────────────────────────────────────────────────────────────

def sst_scalogram(
    t_us,
    sig,
    method: str = "cwt",           # "cwt" | "stft"
    wavelet: str = "cmor3.0-1.0",
    fmin: float = 1.0e6,
    fmax: float = 4.5e6,
    n_freqs: int = 400,
    nperseg: int = 256,
    gamma: float = 1e-6,
    name: str = "sst_scalogram",
    folder: str = "plots",
):
    """
    Single entry-point for both CWT-SST and STFT-SST spectrograms.
    Mirrors the signature of wavelet_processing.wavelet_scalogram so it can
    be called identically in despair.py.
    """
    if method == "cwt":
        cwt_sst(t_us, sig, wavelet=wavelet, fmin=fmin, fmax=fmax,
                n_freqs=n_freqs, gamma=gamma, name=name, plot=True,
                folder=folder)
    elif method == "stft":
        stft_sst(t_us, sig, fmin=fmin, fmax=fmax, nperseg=nperseg,
                 gamma=gamma, name=name, plot=True, folder=folder)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'cwt' or 'stft'.")


def reconstruct_frequency_band(
    t_us,
    sig,
    band_min: float,
    band_max: float,
    method: str = "cwt",
    wavelet: str = "cmor3.0-1.0",
    fmin: float = 1.0e6,
    fmax: float = 4.5e6,
    n_freqs: int = 400,
    nperseg: int = 256,
    gamma: float = 1e-6,
):
    """
    Unified band reconstruction — SST version of wavelet_processing.reconstruct_frequency_band.
    Call with method='cwt' or method='stft'.
    """
    if method == "cwt":
        return reconstruct_band_cwt_sst(
            t_us, sig, band_min=band_min, band_max=band_max,
            wavelet=wavelet, fmin=fmin, fmax=fmax, n_freqs=n_freqs, gamma=gamma,
        )
    elif method == "stft":
        return reconstruct_band_stft_sst(
            t_us, sig, band_min=band_min, band_max=band_max,
            fmin=fmin, fmax=fmax, nperseg=nperseg, gamma=gamma,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'cwt' or 'stft'.")


        