import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
from ssqueezepy import ssq_stft, issq_stft


# ─────────────────────────────────────────────────────────────
#  I/O helpers  (unchanged from original)
# ─────────────────────────────────────────────────────────────

def get_data(file):
    data = pd.read_csv(file)
    for col in data:
        data[col] = data[col].str.replace(',', '.')
    data = data.astype(float)
    return data


# ─────────────────────────────────────────────────────────────
#  Internal helper
# ─────────────────────────────────────────────────────────────

def _to_numpy(arr):
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        return arr.to_numpy().squeeze()
    return np.asarray(arr)


# ─────────────────────────────────────────────────────────────
#  Plotting helpers  (unchanged + new plot_sst)
# ─────────────────────────────────────────────────────────────

def plot(x, y, downsampling=1, name="plot"):
    folder = "plots"
    os.makedirs(folder, exist_ok=True)
    x_plot = _to_numpy(x)
    y_plot = _to_numpy(y)
    plt.figure(figsize=(8, 4))
    if y_plot.ndim == 1:
        plt.plot(x_plot[::downsampling], y_plot[::downsampling], linewidth=1)
    elif y_plot.ndim == 2:
        plt.pcolormesh(x_plot, np.arange(y_plot.shape[0]), y_plot, shading='gouraud')
        plt.xlabel("Time"); plt.ylabel("Frequency bin")
        plt.colorbar(label='Amplitude')
    else:
        raise ValueError("y_plot must be 1D or 2D")
    plt.title(name)
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filepath}")


def plot_stft(f, t_seg, amplitude, downsampling=1, name="stft_plot", dB=False):
    f = _to_numpy(f); t_seg = _to_numpy(t_seg); amplitude = np.asarray(amplitude)
    t_plot = t_seg[::downsampling]
    amp_plot = amplitude[:, ::downsampling]
    if dB:
        amp_plot = 20 * np.log10(amp_plot + 1e-12)
    folder = "plots"; os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_plot, f, amp_plot, shading='gouraud')
    plt.xlabel("Time [s]"); plt.ylabel("Frequency [Hz]"); plt.title(name)
    plt.colorbar(label='Amplitude (dB)' if dB else 'Amplitude')
    plt.tight_layout()
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300); plt.close()
    print(f"Plot saved to {filepath}")


def plot_sst(f, t_seg, amplitude, downsampling=1, name="sst_plot", dB=False):
    """Same signature as plot_stft — correctly labelled for SST output."""
    plot_stft(f, t_seg, amplitude, downsampling=downsampling, name=name, dB=dB)


# ─────────────────────────────────────────────────────────────
#  STFT  (original, minor param fix)
# ─────────────────────────────────────────────────────────────

def stft(sig_in, time, nperseg=256, noverlap=None, window='hann'):
    time_np = _to_numpy(time); sig_np = _to_numpy(sig_in)
    dt = np.mean(np.diff(time_np)); fs = 1.0 / dt
    if noverlap is None:
        noverlap = nperseg // 2
    f, t_seg, Zxx = signal.stft(sig_np, fs=fs, window=window,
                                  nperseg=nperseg, noverlap=noverlap, boundary=None)
    return f, t_seg, np.abs(Zxx), fs



# ─────────────────────────────────────────────────────────────
#  SST — Synchrosqueezed STFT  (equivalent to MATLAB fsst)
# ─────────────────────────────────────────────────────────────

def sst(sig_in, time, win_len=256, hop_len=1, window='hann', gamma=None):
    """
    Synchrosqueezed Short-Time Fourier Transform.
    Equivalent to MATLAB:  [s, f, t] = fsst(x, fs, window)

    Parameters
    ----------
    sig_in   : array-like / pd.Series   input signal
    time     : array-like / pd.Series   uniformly-spaced time vector
    win_len  : int    analysis window length in samples  (≈ MATLAB 'window')
    hop_len  : int    hop between frames (1 = maximum overlap, like MATLAB default)
    window   : str    window shape ('hann', 'hamming', etc.)
    gamma    : float  phase-transform threshold; None = auto (recommended)

    Returns
    -------
    ssq_freqs : np.ndarray (F,)     frequency axis [Hz]
    t_seg     : np.ndarray (T,)     time axis [s]
    Tx_amp    : np.ndarray (F, T)   |SST| amplitude  ← sharpened TF map
    Sx_amp    : np.ndarray (F, T)   |STFT| amplitude ← blurry baseline
    fs        : float               sample rate [Hz]
    """
    time_np = _to_numpy(time)
    sig_np  = _to_numpy(sig_in).astype(np.float64)
    dt = np.mean(np.diff(time_np)); fs = 1.0 / dt
    win_arr = signal.get_window(window, win_len)

    Tx, Sx, ssq_freqs, _ = ssq_stft(sig_np, window=win_arr,
                                      hop_len=hop_len, fs=fs, gamma=gamma)
    n_frames = Tx.shape[1]
    t_seg = np.arange(n_frames) * hop_len / fs
    return ssq_freqs, t_seg, np.abs(Tx), np.abs(Sx), fs


def sst_complex(sig_in, time, win_len=256, hop_len=1, window='hann', gamma=None):
    """
    Like sst() but returns raw complex Tx — needed before calling isst().
    """
    time_np = _to_numpy(time)
    sig_np  = _to_numpy(sig_in).astype(np.float64)
    dt = np.mean(np.diff(time_np)); fs = 1.0 / dt
    win_arr = signal.get_window(window, win_len)

    Tx, Sx, ssq_freqs, _ = ssq_stft(sig_np, window=win_arr, hop_len=hop_len,
                                      fs=fs, gamma=gamma, preserve_transform=True)
    n_frames = Tx.shape[1]
    t_seg = np.arange(n_frames) * hop_len / fs
    return ssq_freqs, t_seg, Tx, Sx, fs        # Tx and Sx are complex


def isst(Tx, win_len=256, hop_len=1, window='hann'):
    """
    Inverse SST — reconstruct the time-domain signal.
    Equivalent to MATLAB:  xrec = ifsst(s, window)

    Parameters
    ----------
    Tx      : complex np.ndarray (F, T)  from sst_complex()
    win_len, hop_len, window  — must match the forward sst_complex() call

    Returns
    -------
    x_rec   : np.ndarray   reconstructed signal (real-valued)
    """
    win_arr = signal.get_window(window, win_len)
    x_rec = issq_stft(Tx, window=win_arr, hop_len=hop_len)
    return np.real(x_rec)

def plot_sst_3d( t_seg, f, amplitude, downsampling=1, name="sst_3d", dB=False,
                freq_min=None, freq_max=None, elev=30, azim=-60):
    """
    3-D surface plot of an SST or STFT amplitude matrix.

    Parameters
    ----------
    f, t_seg, amplitude : same as plot_stft / plot_sst
    downsampling        : thin the time axis for speed
    dB                  : convert amplitude to dB before plotting
    freq_min / freq_max : clip the frequency axis (e.g. your 2–3.5 MHz range)
    elev, azim          : viewing angle (degrees)
    """
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

    f       = _to_numpy(f)
    t_seg   = _to_numpy(t_seg)
    amp     = np.asarray(amplitude)

    # ── frequency crop ──────────────────────────────────────────────────────
    mask = np.ones(len(f), dtype=bool)
    if freq_min is not None:
        mask &= f >= freq_min
    if freq_max is not None:
        mask &= f <= freq_max
    f   = f[mask]
    amp = amp[mask, :]

    # ── time downsampling ───────────────────────────────────────────────────
    t_seg = t_seg[::downsampling]
    amp   = amp[:, ::downsampling]

    # ── dB conversion ───────────────────────────────────────────────────────
    if dB:
        amp = 20 * np.log10(amp + 1e-12)

    # ── meshgrid ─────────────────────────────────────────────────────────────
    T, F = np.meshgrid(t_seg, f)

    # ── plot ─────────────────────────────────────────────────────────────────
    folder = "plots"
    os.makedirs(folder, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T, F, amp, cmap='turbo', linewidth=0, antialiased=False)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [MHz]")
    ax.set_zlabel("Amplitude (dB)" if dB else "Amplitude")
    ax.set_title(name)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Amplitude (dB)" if dB else "Amplitude")

    plt.tight_layout()
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {filepath}")

# ─────────────────────────────────────────────────────────────
#  Self-test  (mirrors the MATLAB fsst documentation example)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Chirp: instantaneous frequency sweeps 100 → 200 Hz over 2 s
    fs_demo = 1000
    t_demo  = np.arange(0, 2, 1 / fs_demo)
    x_demo  = np.sin(2 * np.pi * (100 * t_demo + 50 * t_demo ** 2))

    sig_s  = pd.Series(x_demo)
    time_s = pd.Series(t_demo)

    # STFT baseline
    f_stft, t_stft, amp_stft, _ = stft(sig_s, time_s, nperseg=256, noverlap=128)
    plot_stft(f_stft, t_stft, amp_stft, name="demo_stft", dB=True)

    # SST — sharper frequency localisation
    f_sst, t_sst, Tx_amp, Sx_amp, fs_sst = sst(sig_s, time_s, win_len=256, hop_len=1)
    plot_sst(f_sst, t_sst, Tx_amp, name="demo_sst", dB=True)

    # Reconstruction via inverse SST
    _, _, Tx_c, Sx_c, _ = sst_complex(sig_s, time_s, win_len=256, hop_len=1)
    x_rec = isst(Tx_c, win_len=256, hop_len=1)
    n = min(len(t_demo), len(x_rec))
    plot(t_demo[:n], x_demo[:n],  name="demo_original")
    plot(t_demo[:n], x_rec[:n],   name="demo_reconstructed")

    print(f"\nSample rate   : {fs_sst:.0f} Hz")
    print(f"STFT shape    : {amp_stft.shape}  (freq × time)")
    print(f"SST  shape    : {Tx_amp.shape}  (freq × time)")
    print(f"Reconstructed : {len(x_rec)} samples")
    print("\nDone — check the plots/ folder.")