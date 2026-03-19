import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import hilbert

def wigner_ville_distribution(x):
    x = np.asarray(x, dtype=complex)
    if np.isrealobj(x):
        x = hilbert(x)

    N = len(x)
    W = np.zeros((N, N), dtype=complex)

    for t in range(N):
        for tau in range(-(N//2), N//2):  # exclude +N//2 to avoid collision
            t1 = t + tau
            t2 = t - tau
            if 0 <= t1 < N and 0 <= t2 < N:
                W[t, tau % N] = x[t1] * np.conj(x[t2])

        W[t, :] = np.fft.fft(W[t, :])

    W = np.fft.fftshift(W, axes=1)
    return np.real(W)

def plot_wvd(x, fs=1.0, title="Wigner-Ville Distribution", cmap="inferno", figsize=(10, 5)):
    """
    Plot the Wigner-Ville Distribution of a signal.

    Parameters:
        x       : input signal (real or complex)
        fs      : sampling frequency in Hz (default 1.0)
        title   : plot title
        cmap    : matplotlib colormap (default 'inferno')
        figsize : figure size tuple
    """
    W = wigner_ville_distribution(x)
    N = len(x)
    t = np.arange(N) / fs
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: full WVD including negative frequencies
    im = axes[0].imshow(
        W.T, aspect='auto', origin='lower', cmap=cmap,
        extent=[t[0], t[-1], f[0], f[-1]]
    )
    axes[0].set_title(title)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=axes[0], label="Amplitude")

    # Right: positive frequencies only (cleaner for real signals)
    pos = f >= 0
    im2 = axes[1].imshow(
        W.T[pos], aspect='auto', origin='lower', cmap=cmap,
        extent=[t[0], t[-1], 0, f[-1]]
    )
    axes[1].set_title(f"{title} (positive freq)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    plt.colorbar(im2, ax=axes[1], label="Amplitude")

    plt.tight_layout()
    plt.show()

def plot_spectrogram(x, fs, use_db=True):
    """
    Plot spectrogram of a signal.

    Parameters:
        x (np.ndarray): input signal
        fs (float): sampling frequency
        use_db (bool): if True, plot in dB scale
    """
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=128, noverlap=64)

    # Convert to dB if requested
    if use_db:
        Sxx_plot = 10 * np.log10(Sxx + 1e-10)  # avoid log(0)
        label = 'Power (dB)'
    else:
        Sxx_plot = Sxx
        label = 'Power'

    plt.figure()
    plt.pcolormesh(t, f, Sxx_plot, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram (STFT)')
    plt.colorbar(label=label)
    plt.show()