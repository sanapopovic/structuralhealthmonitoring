import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import matplotlib.colors as colors
import os




def envelope(signal, peaks):
    """Compute cubic spline envelope through peaks."""
    x = np.arange(len(signal))
    if len(peaks) < 2:
        return np.zeros_like(signal)
    spline = CubicSpline(peaks, signal[peaks])
    return spline(x)




def sift(signal):
    """Perform one sift operation to extract IMF candidate."""
    x = np.arange(len(signal))

    # find maxima and minima
    maxima, _ = find_peaks(signal)
    minima, _ = find_peaks(-signal)

    if len(maxima) < 2 or len(minima) < 2:
        return signal

    upper_env = envelope(signal, maxima)
    lower_env = envelope(signal, minima)

    mean_env = (upper_env + lower_env) / 2
    return signal - mean_env




def extract_imf(signal, max_iter=100, tol=1e-5):
    """Extract one IMF using iterative sifting."""
    h = signal.copy()

    for _ in range(max_iter):
        h1 = sift(h)

        if np.linalg.norm(h - h1) < tol:
            break

        h = h1

    return h




def emd(signal, max_imfs=10):
    """Decompose signal into IMFs."""
    residue = signal.copy()
    imfs = []

    for _ in range(max_imfs):

        imf = extract_imf(residue)

        if np.allclose(imf, 0):
            break

        imfs.append(imf)
        residue = residue - imf

        if np.sum(np.abs(residue)) < 1e-6:
            break

    return np.array(imfs), residue




def hilbert_analysis(imfs, fs):
    """Compute instantaneous amplitude and frequency."""
    inst_amp = []
    inst_freq = []

    for imf in imfs:

        analytic = hilbert(imf)

        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))

        frequency = np.diff(phase) * fs / (2*np.pi)

        inst_amp.append(amplitude[:-1])
        inst_freq.append(frequency)

    return inst_amp, inst_freq




def plot_hilbert_spectrum_cloud(inst_freq, inst_amp, t):

    plt.figure(figsize=(10,6))

    for f, a in zip(inst_freq, inst_amp):

        plt.scatter(
            t[:-1],
            f,
            s=5*a/np.max(a),
            alpha=0.6
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    #plt.ylim(1.5*(10**6), 3.5*(10**6))
    plt.title("Hilbert Time-Frequency Spectrum")
    plt.show()


def imf_extraction(imfs, inst_freq, freq, bandwidth, smoothness = 0.1):

    imf_p = []
    

    for i, imf in enumerate(imfs):
        freq_mean = np.mean(inst_freq[i])
        freq_std = np.std(inst_freq[i])

        
        # Criterion: low variation relative to mean
        if np.abs(freq_std/freq_mean)  < np.array([smoothness]):
            if (freq_mean >= freq - bandwidth/2) and (freq_mean <= freq + bandwidth/2):
                imf_p.append(imf)

    if len(imf_p) == 0:
        raise ValueError("No sufficiently constant IMF was found inside the bandwidth")
    else:           
        return imf_p


def Bandpass(signal, fs, freq, bandwidth=5, order=4):
    """
    Extract an IMF from a signal at a specified frequency using a band-pass filter.
    
    Parameters
    ----------
    signal : ndarray
        Input 1D signal
    fs : float
        Sampling frequency in Hz
    freq : float
        Target frequency to extract (Hz)
    bandwidth : float, optional
        +/- bandwidth around freq (Hz), default is 5 Hz
    order : int, optional
        Order of the Butterworth filter, default is 4
    
    Returns
    -------
    imf : ndarray
        The extracted IMF centered around `freq`
    """
    
    # Define lower and upper cutoff frequencies
    low = max(freq - bandwidth, 0.01)  # avoid 0 Hz
    high = freq + bandwidth
    
    # Normalize by Nyquist frequency
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq
    
    # Design Butterworth bandpass filter
    b, a = butter(order, [low_norm, high_norm], btype='band')
    
    # Apply zero-phase filtering
    imf = filtfilt(b, a, signal)
    
    return imf

def plot_hilbert_spectrum(inst_freq,inst_amp,t,fs, name,f_bins=1000,t_bins=800,freq_percentile=99.7,log_amplitude=False):
    """
    Plot a Hilbert spectrum (time-frequency energy density map).

    Features:
    - Keeps only instantaneous frequencies > 0
    - Automatically rescales frequency axis using percentile clipping
    - Optional logarithmic scaling of amplitude values
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Flatten all IMFs into single arrays
    all_t = []
    all_f = []
    all_w = []

    for f, a in zip(inst_freq, inst_amp):

        # Keep only strictly positive finite frequencies
        mask = np.isfinite(f) & (f > 500)

        all_t.append(t[:-1][mask])
        all_f.append(f[mask])
        all_w.append(a[mask])

    all_t = np.concatenate(all_t)
    all_f = np.concatenate(all_f)
    all_w = np.concatenate(all_w)

    # ------------------------------------------------------------------
    # Better frequency scaling
    # ------------------------------------------------------------------
    f_min = np.min(all_f)
    f_max = np.percentile(all_f, freq_percentile)

    # Clip extreme frequencies
    keep = all_f <= f_max

    all_t = all_t[keep]
    all_f = all_f[keep]
    all_w = all_w[keep]

    # ------------------------------------------------------------------
    # Build histogram
    # ------------------------------------------------------------------
    H, t_edges, f_edges = np.histogram2d(
        all_t,
        all_f,
        bins=[t_bins, f_bins],
        range=[[all_t.min(), all_t.max()], [f_min, f_max]],
        weights=all_w
    )

    # ------------------------------------------------------------------
    # Log-scale amplitude values (NOT axis)
    # ------------------------------------------------------------------
    if log_amplitude:
        H = np.log10(H + 1e-12)

    T, F = np.meshgrid(t_edges[:-1], f_edges[:-1], indexing="ij")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    folder = "plots"
    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    pcm = ax.pcolormesh(
        T,
        F,
        H,
        shading="auto",
        cmap="viridis"
    )
    if f_max >= 4.0e6:
        ax.set_ylim(f_min, 4.0e6)
    if f_max < 4.0e6:
        ax.set_ylim(f_min, f_max)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    if log_amplitude:
        ax.set_title("Hilbert Spectrum (Log Amplitude)")
        cbar_label = "log10(Amplitude)"
    else:
        ax.set_title("Hilbert Spectrum")
        cbar_label = "Amplitude"

    fig.colorbar(pcm, ax=ax, label=cbar_label)

    plt.tight_layout()
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {filepath}")
    

    return fig, ax, H, T, F

def reconstruction(inst_amp, inst_freq, fs, fmin, fmax):
    n = len(inst_amp[0]) + 1  # restore original length
    band_signal = np.zeros(n)

    for amp, freq in zip(inst_amp, inst_freq):

        # Mask
        mask = (freq >= fmin) & (freq <= fmax)
        filtered_freq = freq * mask

        # Integrate phase (length N-1)
        phase = np.cumsum(filtered_freq) * (2 * np.pi / fs)

        # Pad phase and amplitude to length N
        phase = np.insert(phase, 0, phase[0])
        amp_full = np.insert(amp, 0, amp[0])

        band_signal += amp_full * np.cos(phase)

    return band_signal

def bandpass_hilbert(imfs, fs, fmin, fmax):
    band_signal = 0

    for imf in imfs:
        analytic = hilbert(imf)

        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * fs / (2*np.pi)
        inst_freq = np.concatenate(([inst_freq[0]], inst_freq))

        mask = (inst_freq >= fmin) & (inst_freq <= fmax)

        # Apply mask directly to analytic signal
        filtered = analytic * mask

        band_signal += np.real(filtered)

    return band_signal