import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline





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




def extract_imf(signal, max_iter=50, tol=1e-3):
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




def plot_hilbert_spectrum(inst_freq, inst_amp, t):

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


def extract_imf_at_frequency(signal, fs, freq, bandwidth=5, order=4):
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


