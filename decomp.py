import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from collections import defaultdict

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
        return signal

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
     


    




