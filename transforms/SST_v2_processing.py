"""
═══════════════════════════════════════════════════════════════════════════════
  SST PROCESSING — Synchrosqueezing Transform
  ────────────────────────────────────────────
  Two flavours:
    1. STFT-SST  : Short-Time Fourier Transform + synchrosqueezing
    2. CWT-SST   : Continuous Wavelet Transform  + synchrosqueezing

  Synchrosqueezing sharpens the time-frequency representation by
  reassigning energy to the instantaneous frequency ridge, turning
  smeared modal packets into crisp ridges — improving both visual
  readability and downstream reconstruction accuracy.

  Public API
  ──────────
    stft_sst(t_us, sig, ...)         → Ts, f_hz, t_s, S_sst
    cwt_sst(t_us, sig, ...)          → Tc, f_hz, t_us_out, C_sst
    plot_comparison(...)             → saves side-by-side figure
    reconstruct_band_stft(...)       → 1-D reconstructed signal
    reconstruct_band_cwt(...)        → 1-D reconstructed signal

  Compatible with the rest of the pipeline (same t in µs, Hz throughout).
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pywt
import os
import matplotlib.pyplot as plt
from scipy.signal import stft as scipy_stft, istft as scipy_istft


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_seconds(t_us):
    return np.asarray(t_us, dtype=float) * 1e-6


def _db(power, floor_db=-60):
    """Convert power to dB, clipped to floor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(power + 1e-30)
    return np.clip(db, floor_db, None)


def _save(fig, name, folder="plots"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  1.  STFT-SST
# ─────────────────────────────────────────────────────────────────────────────

def stft_sst(
    t_us,
    sig,
    fmin=1.0e6,
    fmax=4.5e6,
    win_len=128,
    hop_len=2,
    n_fft=512,
    gamma=1e-6,
):
    """
    Compute STFT and its synchrosqueezed version.

    Parameters
    ----------
    t_us    : array, time in microseconds
    sig     : array, signal
    fmin    : float, Hz — lower display/reassignment bound
    fmax    : float, Hz — upper display/reassignment bound
    win_len : int,   STFT window length (samples)
    hop_len : int,   hop size (samples)
    n_fft   : int,   FFT size (zero padding)
    gamma   : float, threshold — STFT bins with |V| < gamma are skipped

    Returns
    -------
    S       : 2-D array (n_freq × n_time), plain STFT magnitude
    S_sst   : 2-D array (n_freq × n_time), SST magnitude
    f_hz    : 1-D array, frequency axis (Hz)  — covers fmin..fmax
    t_s     : 1-D array, time axis (s)
    """

    t = _to_seconds(t_us)
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt

    sig = np.asarray(sig, dtype=float).squeeze()

    window = np.hanning(win_len)

    # ── Standard STFT ────────────────────────────────────────────────────────
    f_all, t_s, Vx = scipy_stft(
        sig,
        fs=fs,
        window=window,
        nperseg=win_len,
        noverlap=win_len - hop_len,
        nfft=n_fft,
        return_onesided=True,
    )

    # frequency bin width
    df = f_all[1] - f_all[0]

    # ── Instantaneous frequency via phase derivative ──────────────────────────
    # Compute STFT of sig' (time-derivative) to get d/dt V(f,t)
    sig_dot = np.gradient(sig, dt)
    _, _, Vx_dot = scipy_stft(
        sig_dot,
        fs=fs,
        window=window,
        nperseg=win_len,
        noverlap=win_len - hop_len,
        nfft=n_fft,
        return_onesided=True,
    )

    # Instantaneous frequency estimate (Hz):  omega_hat = Im( V_dot / V ) / (2π)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega_hat = np.where(
            np.abs(Vx) > gamma,
            np.imag(Vx_dot / (Vx + 1e-30)) / (2 * np.pi),
            np.nan,
        )

    # ── Synchrosqueezing reassignment ────────────────────────────────────────
    # Target grid: same frequency bins as STFT but limited to [fmin, fmax]
    f_mask = (f_all >= fmin) & (f_all <= fmax)
    f_hz = f_all[f_mask]
    n_f = len(f_hz)
    n_t = Vx.shape[1]

    S = np.abs(Vx[f_mask, :])          # plain STFT (band only)
    S_sst = np.zeros((n_f, n_t), dtype=float)

    # Map each (f, t) STFT bin → nearest target frequency bin
    for k, fk in enumerate(f_all):
        if not f_mask[k]:
            continue
        col_omega = omega_hat[k, :]     # shape (n_t,)
        for ti in range(n_t):
            om = col_omega[ti]
            if np.isnan(om):
                continue
            # find nearest bin in f_hz
            l = int(round((om - f_hz[0]) / df))
            if 0 <= l < n_f:
                S_sst[l, ti] += np.abs(Vx[k, ti])

    return S, S_sst, f_hz, t_s


# ─────────────────────────────────────────────────────────────────────────────
#  2.  CWT-SST
# ─────────────────────────────────────────────────────────────────────────────

def cwt_sst(
    t_us,
    sig,
    wavelet="cmor3.0-1.0",
    fmin=1.0e6,
    fmax=4.5e6,
    n_freqs=400,
    gamma=1e-8,
):
    """
    Compute CWT scalogram and its synchrosqueezed version.

    Parameters
    ----------
    t_us    : array, time in microseconds
    sig     : array, signal
    wavelet : str,   pywt-compatible complex wavelet name
    fmin    : float, Hz
    fmax    : float, Hz
    n_freqs : int,   number of frequency bins
    gamma   : float, CWT coefficient threshold

    Returns
    -------
    C       : 2-D array (n_freqs × n_time), plain CWT magnitude
    C_sst   : 2-D array (n_freqs × n_time), SST magnitude
    f_hz    : 1-D array, frequency axis (Hz)
    t_us_out: 1-D array, time axis (µs) — same as input t_us
    """

    t = _to_seconds(t_us)
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt

    sig = np.asarray(sig, dtype=float).squeeze()

    # ── Scale ↔ frequency mapping ─────────────────────────────────────────────
    fc = pywt.central_frequency(wavelet)
    freqs_target = np.linspace(fmin, fmax, n_freqs)   # Hz
    scales = fc / (freqs_target * dt)                  # pywt scales

    # ── CWT ──────────────────────────────────────────────────────────────────
    cwtmatr, freqs_cwt = pywt.cwt(sig, scales, wavelet, sampling_period=dt)
    # cwtmatr : (n_freqs × n_samples), complex

    # ── Instantaneous frequency via scale derivative of phase ─────────────────
    # d/dt arg(W(a,t))  ≈  Im( dW/dt / W ) / (2π)
    # We use central differences along the time axis.
    dW_dt = np.gradient(cwtmatr, dt, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        omega_hat = np.where(
            np.abs(cwtmatr) > gamma,
            np.imag(dW_dt / (cwtmatr + 1e-30)) / (2 * np.pi),
            np.nan,
        )

    # ── Synchrosqueezing reassignment ────────────────────────────────────────
    df = freqs_target[1] - freqs_target[0]

    C = np.abs(cwtmatr)                                # plain CWT magnitude
    C_sst = np.zeros_like(C, dtype=float)

    for k in range(n_freqs):
        for ti in range(C.shape[1]):
            om = omega_hat[k, ti]
            if np.isnan(om):
                continue
            l = int(round((om - freqs_target[0]) / df))
            if 0 <= l < n_freqs:
                C_sst[l, ti] += C[k, ti]

    f_hz = freqs_target
    t_us_out = np.asarray(t_us)

    return C, C_sst, f_hz, t_us_out


# ─────────────────────────────────────────────────────────────────────────────
#  3.  SIDE-BY-SIDE COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    t_axis,
    f_hz,
    original,
    sst,
    method="STFT",
    name="sst_comparison",
    log_scale=True,
    fmin=None,
    fmax=None,
):
    """
    Plot original TF representation alongside its SST version.

    Parameters
    ----------
    t_axis   : time axis (µs for CWT / s for STFT — will label correctly)
    f_hz     : frequency axis in Hz
    original : 2-D magnitude array (n_freq × n_time)
    sst      : 2-D magnitude array (n_freq × n_time)
    method   : 'STFT' or 'CWT'
    name     : output filename (without extension)
    log_scale: if True, plot dB; otherwise linear
    """

    t_label = "Time [µs]" if method == "CWT" else "Time [s]"
    f_mhz = f_hz / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    def _render(ax, data, title):
        if log_scale:
            img = _db(data ** 2)
        else:
            img = data
        pcm = ax.pcolormesh(
            t_axis, f_mhz, img,
            shading="gouraud",
            cmap="inferno",
        )
        ax.set_xlabel(t_label, fontsize=11)
        ax.set_ylabel("Frequency [MHz]", fontsize=11)
        ax.set_title(title, fontsize=12)
        if fmin is not None:
            ax.set_ylim(fmin / 1e6, fmax / 1e6)
        fig.colorbar(pcm, ax=ax, label="dB" if log_scale else "Amplitude")

    _render(axes[0], original, f"{method} — Original")
    _render(axes[1], sst,      f"{method} — Synchrosqueezed")

    plt.suptitle(f"Time–Frequency Comparison: {method} vs SST", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, name)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  BAND RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_band_stft(
    t_us,
    sig,
    band_min,
    band_max,
    fmin=1.0e6,
    fmax=4.5e6,
    win_len=128,
    hop_len=2,
    n_fft=512,
):
    """
    Reconstruct a frequency band from the STFT (no SST needed for reconstruction —
    SST is a display reassignment; we zero-out unwanted bins then iSTFT).

    Returns
    -------
    recon : 1-D array, reconstructed signal in the band [band_min, band_max] Hz
    """

    t = _to_seconds(t_us)
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    sig = np.asarray(sig, dtype=float).squeeze()

    window = np.hanning(win_len)

    f_all, t_s, Vx = scipy_stft(
        sig,
        fs=fs,
        window=window,
        nperseg=win_len,
        noverlap=win_len - hop_len,
        nfft=n_fft,
        return_onesided=True,
    )

    # zero-out bins outside the band
    band_mask = (f_all >= band_min) & (f_all <= band_max)
    Vx_band = np.where(band_mask[:, None], Vx, 0.0)

    # inverse STFT
    _, recon = scipy_istft(
        Vx_band,
        fs=fs,
        window=window,
        nperseg=win_len,
        noverlap=win_len - hop_len,
        nfft=n_fft,
    )

    # trim / pad to original length
    n = len(sig)
    if len(recon) >= n:
        recon = recon[:n]
    else:
        recon = np.pad(recon, (0, n - len(recon)))

    return recon


def reconstruct_band_cwt(
    t_us,
    sig,
    band_min,
    band_max,
    wavelet,
    fmin=1.0e6,
    fmax=4.5e6,
    n_freqs=400,
):
    """
    CWT band reconstruction — direct port of the working
    wavelet_processing.reconstruct_frequency_band logic.
    No reordering, no custom normalisation — just the same
    formula that already works in the pipeline.

    Returns
    -------
    recon : 1-D array
    """

    t = _to_seconds(t_us)
    dt = float(np.mean(np.diff(t)))
    sig = np.asarray(sig, dtype=float).squeeze()

    fc = pywt.central_frequency(wavelet)
    freqs_target = np.linspace(fmin, fmax, n_freqs)   # high → low (matches pywt order)
    scales = fc / (freqs_target * dt)

    cwtmatr, freqs = pywt.cwt(sig, scales, wavelet, sampling_period=dt)

    # band selection using the freqs pywt actually returned
    mask = (freqs >= band_min) & (freqs <= band_max)
    cwt_band    = cwtmatr[mask, :]
    scales_band = scales[mask]

    if scales_band.size < 2:
        return np.zeros_like(sig)

    # Torrence & Compo 1998 — correct formula for analytic wavelets:
    # sum( W(a,t) * da/a )
    da   = np.abs(np.diff(scales_band))
    da   = np.append(da, da[-1])          # pad to match length
    recon = np.real(np.sum(cwt_band * da[:, None] / scales_band[:, None], axis=0))

    return recon


# ─────────────────────────────────────────────────────────────────────────────
#  5.  RECONSTRUCTION PLOT  (signal vs reconstructed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstruction(
    t_us,
    original_signal,
    recon_base,
    recon_harmonic,
    gt_base=None,
    gt_harmonic=None,
    method="STFT",
    name="sst_reconstruction",
):
    """
    Five-panel reconstruction plot:
      1 — composite signal (f + 2f combined)
      2 — ground truth fundamental  (gt_base)
      3 — base band reconstruction  → compare directly with panel 2
      4 — ground truth second harmonic (gt_harmonic)
      5 — harmonic band reconstruction → compare directly with panel 4

    If gt_base / gt_harmonic are None the GT panels are skipped and
    a 3-panel layout is used instead.
    """

    have_gt = (gt_base is not None) and (gt_harmonic is not None)
    n_panels = 5 if have_gt else 3

    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 3.2 * n_panels), sharex=True)

    # panel 1 — composite
    axes[0].plot(t_us, original_signal, color="steelblue", lw=0.9)
    axes[0].set_title(f"Composite signal  (fundamental + 2nd harmonic)  [{method}]", fontsize=11)
    axes[0].set_ylabel("Amplitude [nm]")
    axes[0].grid(True, alpha=0.25)

    if have_gt:
        # panel 2 — GT fundamental
        axes[1].plot(t_us, gt_base, color="seagreen", lw=0.9, label="GT fundamental")
        axes[1].set_title("Ground truth — fundamental (sum of base modes)", fontsize=11)
        axes[1].set_ylabel("Amplitude [nm]")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.25)

        # panel 3 — base reconstruction vs GT
        axes[2].plot(t_us, gt_base,    color="seagreen", lw=0.9,  alpha=0.6, label="GT fundamental")
        axes[2].plot(t_us, recon_base, color="tomato",   lw=1.0,  label="Base band reconstruction")
        axes[2].set_title("Base band reconstruction  vs  GT fundamental", fontsize=11)
        axes[2].set_ylabel("Amplitude [nm]")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.25)

        # panel 4 — GT harmonic
        axes[3].plot(t_us, gt_harmonic, color="mediumpurple", lw=0.9, label="GT 2nd harmonic")
        axes[3].set_title("Ground truth — 2nd harmonic (sum of harmonic modes × scale)", fontsize=11)
        axes[3].set_ylabel("Amplitude [nm]")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.25)

        # panel 5 — harmonic reconstruction vs GT
        axes[4].plot(t_us, gt_harmonic,    color="mediumpurple", lw=0.9, alpha=0.6, label="GT 2nd harmonic")
        axes[4].plot(t_us, recon_harmonic, color="darkorange",   lw=1.0, label="Harmonic band reconstruction")
        axes[4].set_title("Harmonic band reconstruction  vs  GT 2nd harmonic", fontsize=11)
        axes[4].set_ylabel("Amplitude [nm]")
        axes[4].set_xlabel("Time [µs]")
        axes[4].legend(fontsize=8)
        axes[4].grid(True, alpha=0.25)

    else:
        # fallback 3-panel (no GT available)
        axes[1].plot(t_us, recon_base, color="tomato", lw=1.0)
        axes[1].set_title("Base band reconstruction", fontsize=11)
        axes[1].set_ylabel("Amplitude [nm]")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(t_us, recon_harmonic, color="darkorange", lw=1.0)
        axes[2].set_title("Harmonic band reconstruction", fontsize=11)
        axes[2].set_ylabel("Amplitude [nm]")
        axes[2].set_xlabel("Time [µs]")
        axes[2].grid(True, alpha=0.25)

    plt.suptitle(f"Band Reconstructions — {method}-SST", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, name)
    plt.close(fig)