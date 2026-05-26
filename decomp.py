import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from collections import defaultdict
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert




def envelope(signal, peaks):
    x = np.arange(len(signal))
    if len(peaks) < 2:
        return np.zeros_like(signal)
    spline = CubicSpline(peaks, signal[peaks])
    return spline(x)


def sift(signal):
    x = np.arange(len(signal))

    # find maxima and minima
    maxima, _ = find_peaks(signal)
    minima, _ = find_peaks(-signal)

    if len(maxima) < 2 or len(minima) < 2:
        return signal, signal, np.zeros_like(signal)

    upper_env = envelope(signal, maxima)
    lower_env = envelope(signal, minima)

    mean_env = (np.absolute(upper_env) + np.absolute(lower_env)) / 2
    return upper_env, lower_env, mean_env


def align_to_envelope_with_time(envelope, time_array, timestamps, distance=10):

    envelope = np.asarray(envelope)
    time_array = np.asarray(time_array)

    peaks, _ = find_peaks(envelope, distance=distance)

    if len(peaks) == 0:
        return {k: None for k in timestamps}

    peak_times = time_array[peaks]

    result = {}

    for name, t in timestamps.items():
        t = float(t)

        # Clamp to valid time range
        t = max(time_array[0], min(t, time_array[-1]))

        # Find closest peak in time
        idx = np.argmin(np.abs(peak_times - t))

        peak_index = peaks[idx]
        peak_time = peak_times[idx]

        result[name] = {
            "peak_index": int(peak_index),
            "peak_time": float(peak_time),
            "peak_value": float(envelope[peak_index]),
            "time_offset": float(peak_time - t)
        }

    return result


    x,
    fs=1.0,
    debug=False,


def sliding_rms_envelope(x, window_size):
    """
    Sliding RMS envelope of signal x.

    Parameters:
    - x: input signal
    - window_size: number of samples in window (e.g., 1–3 periods)

    Returns:
    - envelope: RMS envelope
    """

    x = np.asarray(x)
    kernel = np.ones(window_size) / window_size

    # mean of squared signal
    x2 = x**2
    mean_x2 = np.convolve(x2, kernel, mode='same')

    return np.sqrt(mean_x2)

def lowpass_filter(x, fs, cutoff):
    """
    Simple Butterworth low-pass filter.
    """
    b, a = butter(4, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, x)


def lock_in_envelope(x, fs, f0, lp_cutoff=None):
    """
    Lock-in / demodulation amplitude envelope.

    Parameters:
    - x: input signal
    - fs: sampling frequency
    - f0: carrier frequency
    - lp_cutoff: low-pass cutoff (default: f0/10)

    Returns:
    - envelope
    """

    x = np.asarray(x)
    t = np.arange(len(x)) / fs

    # mix down
    I = x * np.cos(2 * np.pi * f0 * t)
    Q = x * np.sin(2 * np.pi * f0 * t)

    # low-pass filter to remove 2*f0 term
    if lp_cutoff is None:
        lp_cutoff = f0 / 10

    I_lp = lowpass_filter(I, fs, lp_cutoff)
    Q_lp = lowpass_filter(Q, fs, lp_cutoff)

    # amplitude envelope
    return np.sqrt(I_lp**2 + Q_lp**2)

class KalmanEnvelope:
    def __init__(self, fs, f0, q=1e-4, r=1e-1):
        self.fs = fs
        self.w = 2 * np.pi * f0 / fs

        # state: [I, Q]
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)

        self.Q = q * np.eye(2)
        self.R = np.array([[r]])

    def step(self, z, n):
        # time-varying measurement matrix
        c = np.cos(self.w * n)
        s = np.sin(self.w * n)

        H = np.array([[c, -s]])

        # predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # innovation
        y = z - (H @ x_pred)[0]
        S = H @ P_pred @ H.T + self.R

        K = P_pred @ H.T @ np.linalg.inv(S)

        # update
        self.x = x_pred + K * y
        self.P = (np.eye(2) - K @ H) @ P_pred

        I, Q = self.x.flatten()
        A = np.sqrt(I**2 + Q**2)

        return A, I, Q


def kalman_envelope(x, fs, f0, q=1e-2, r=1):
    kf = KalmanEnvelope(fs, f0, q, r)

    env = np.zeros(len(x))

    for n, xn in enumerate(x):
        env[n], _, _ = kf.step(xn, n)

    return env

def amplitude_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Compute the amplitude envelope of a real-valued signal using the analytic signal method (Hilbert transform).

    Parameters
    ----------
    signal : np.ndarray
        Input real-valued (possibly noisy) signal.

    Returns
    -------
    np.ndarray
        Amplitude envelope of the signal.
    """
    # Compute analytic signal
    analytic_signal = hilbert(signal)

    # Amplitude envelope is the magnitude of the analytic signal
    envelope = np.abs(analytic_signal)

    return envelope

def smooth_envelope(signal, fs, cutoff, order=4):
    """
    Envelope smoothing using Hilbert + Butterworth low-pass filter.

    cutoff is in Hz (NOT normalized)
    """

    # 1. analytic signal
    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    # 2. normalize cutoff
    nyquist = fs / 2
    wn = cutoff / nyquist  # MUST be between 0 and 1

    # safety check
    if not 0 < wn < 1:
        raise ValueError(
            f"Invalid cutoff: {cutoff} Hz (normalized Wn={wn}). "
            f"Must satisfy 0 < cutoff < {nyquist} Hz"
        )

    # 3. filter
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, envelope)



#Me fucking around below here

def cluster_time_stamps(data_points, threshold=0.1):
    """
    Cluster timestamps using distance-to-cluster-center rule,
    and keep mapping to original names.

    Parameters
    ----------
    data_points : dict
        {name: {'peak_time': float, 'peak_value': float}}
    threshold : float

    Returns
    -------
    clustered : dict
        {cluster_id: {'peak_time': mean_time, 'peak_value': mean_value}}
    mapping : dict
        {original_name: cluster_id}
    clusters_raw : list
        raw cluster contents (for debugging/inspection)
    """

    items = sorted(data_points.items(), key=lambda x: x[1]['peak_time'])

    clusters = []
    mapping = {}

    current_cluster = []
    current_center = None

    for name, v in items:
        t = v['peak_time']

        if not current_cluster:
            current_cluster = [(name, v)]
            current_center = t
            continue

        # compare to cluster center (NOT previous point)
        if abs(t - current_center) <= threshold:
            current_cluster.append((name, v))

            # update center (mean of cluster)
            times = [x[1]['peak_time'] for x in current_cluster]
            current_center = float(np.mean(times))
        else:
            clusters.append(current_cluster)
            current_cluster = [(name, v)]
            current_center = t

    if current_cluster:
        clusters.append(current_cluster)

    # build outputs
    clustered = {}

    for i, cluster in enumerate(clusters):
        cid = f"C{i}"

        times = np.array([v['peak_time'] for _, v in cluster])
        values = np.array([v['peak_value'] for _, v in cluster])

        clustered[cid] = {
            "peak_time": float(times.mean()),
            "peak_value": float(values.mean())
        }

        for name, _ in cluster:
            mapping[name] = cid

    return clustered, mapping, clusters

def Hann_Base(r, L, eps=1e-12):
    """
    Row-normalised Hann radial basis function.
    Ensures each row sums to 1 (numerically stable).
    """

    r = np.asarray(r)
    f = 2.6*(10**6)
    # base kernel
    K = np.zeros_like(r, dtype=float)

    mask = np.abs(r) <= L / 2
    K[mask] =  (1/L)*(np.cos(np.pi * r[mask] / L) ** 2) #* np.cos(2 * np.pi * f * r[mask])

    # row-normalise
    row_sum = K.sum(axis=1, keepdims=True)

    K = K / (row_sum + eps)

    return K

def gaussian_base(r, gamma=1.0):
    r = np.asarray(r)
    return np.exp(-gamma * r**2)

def multiquadratic_base(r, epsilon=1.0):
    return  np.sqrt(1 + (epsilon * r)**2)

def rbf_base(X, Y=None, gamma=1.0):
    
    X = np.asarray(X)
    Y = X if Y is None else np.asarray(Y)

    # Squared norms
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)

    # Compute squared Euclidean distance
    dist_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)

    # Apply RBF kernel
    K = np.exp(-gamma * dist_sq)
    return K


def Hann_decomp(data_points, base, regularization=1e-5, L=20, gamma=1, epsilon = 1, threshold = 0.1):
    """
    RBF regression after clustering timestamps.
    """

    clustered, mapping, _ = cluster_time_stamps(data_points, threshold)

    items = sorted(clustered.items(), key=lambda x: x[1]['peak_time'])

    names = [name for name, _ in items]
    times = np.array([v['peak_time'] for _, v in items])
    values = np.array([v['peak_value'] for _, v in items])

    # build kernel matrix
    diff = np.abs(times[:, None] - times[None, :])
    if base =='Hann':
        A = Hann_Base(diff,L)
    elif base =='Gaussian':
        A = gaussian_base(diff, gamma)
    elif base == 'Multiquadratic':
        A = multiquadratic_base(diff, epsilon= epsilon)
    elif base == 'RBF':
        A = rbf_base(diff, gamma= gamma)

    # solve system
    A += regularization * np.eye(len(A))
    x = np.linalg.solve(A, values)

    # map back to original names via cluster id
    coeffs_cluster = {name: float(c) for name, c in zip(names, x)}

    coeffs_original = {
        name: coeffs_cluster[mapping[name]]
        for name in data_points
    }

    return coeffs_original, coeffs_cluster

    
def peak_time(envelope, time_array, timestamps, distance=10):
    envelope = np.asarray(envelope)
    time_array = np.asarray(time_array)


    result = {}

    for name, t in timestamps.items():
        t = float(t)

        # Clamp to valid time range
        t = max(time_array[0], min(t, time_array[-1]))

        # Find closest peak in time
        idx = np.argmin(np.abs(time_array - t))

        peak_time = time_array[idx]

        result[name] = {
            "peak_index": int(idx),
            "peak_time": float(peak_time),
            "peak_value": float(envelope[idx]),
            "time_offset": float(peak_time - t)
        }

    return result
     


    




