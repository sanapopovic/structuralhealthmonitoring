import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# -----------------------------
# Utility: Analytic signal
# -----------------------------
def analytic_signal(x):
    return hilbert(x)


# -----------------------------
# 1. Raw Wigner-Ville Distribution
# -----------------------------
def wvd(x):
    x = analytic_signal(x)
    N = len(x)
    W = np.zeros((N, N), dtype=complex)

    for t in range(N):
        for tau in range(-min(t, N-t-1), min(t, N-t-1)):
            W[t, tau + N//2] = x[t + tau] * np.conj(x[t - tau])

    # FFT over lag variable → frequency axis
    W = np.fft.fftshift(np.fft.fft(W, axis=1), axes=1)
    return np.real(W)


# -----------------------------
# 2. Smoothed Pseudo WVD (SPWVD)
# -----------------------------
def spwvd(x, time_sigma=2, freq_sigma=2):
    W = wvd(x)

    # Smooth in time and frequency
    W_smooth = gaussian_filter(W, sigma=[time_sigma, freq_sigma])
    return W_smooth


# -----------------------------
# 3. Choi-Williams Kernel
# -----------------------------
def choi_williams_kernel(N, sigma=1.0):
    t = np.linspace(-1, 1, N)
    f = np.linspace(-1, 1, N)
    T, F = np.meshgrid(t, f, indexing='ij')

    # Choi-Williams kernel
    kernel = np.exp(-(T**2 * F**2) / sigma)
    return kernel


def wvd_choi_williams(x, sigma=1.0):
    W = wvd(x)
    N = W.shape[0]

    kernel = choi_williams_kernel(N, sigma)
    return np.real(np.fft.ifft2(np.fft.fft2(W) * np.fft.fft2(kernel)))


# -----------------------------
# 4. Gaussian Kernel (Cohen's Class)
# -----------------------------
def gaussian_kernel(N, sigma_t=0.2, sigma_f=0.2):
    t = np.linspace(-1, 1, N)
    f = np.linspace(-1, 1, N)
    T, F = np.meshgrid(t, f, indexing='ij')

    kernel = np.exp(-(T**2 / sigma_t**2 + F**2 / sigma_f**2))
    return kernel


def wvd_gaussian(x, sigma_t=0.2, sigma_f=0.2):
    W = wvd(x)
    N = W.shape[0]

    kernel = gaussian_kernel(N, sigma_t, sigma_f)
    return np.real(np.fft.ifft2(np.fft.fft2(W) * np.fft.fft2(kernel)))


# -----------------------------
# Example Signal
# -----------------------------
def test_signal(N=256):
    t = np.linspace(0, 1, N)
    x = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * (60 * t + 20 * t**2))
    return t, x


# -----------------------------
# Visualization
# -----------------------------
def plot_tfr(W, title, fs=1.0):
    """
    W: time-frequency matrix (time x frequency)
    fs: sampling frequency (used to scale frequency axis)
    """

    N = W.shape[0]

    # Frequency axis (normalized or physical)
    freqs = np.linspace(-fs/2, fs/2, N)

    # Time axis
    times = np.arange(N)

    plt.figure()

    # Transpose so freq is vertical axis
    plt.imshow(
        W.T,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.colorbar()

    # Optional: fix frequency limits (remove autoscaling)
    #plt.ylim(0, fs/2)   # show only positive frequencies

    plt.show()


# -----------------------------
# Run Demo
# -----------------------------
if __name__ == "__main__":
    t, x = test_signal()

    W_raw = wvd(x)
    W_sp = spwvd(x, 0.5, 0.5)
    W_cw = wvd_choi_williams(x, sigma=0.9)
    W_gauss = wvd_gaussian(x, sigma_t=0.9, sigma_f=0.9)

    plot_tfr(W_raw, "Raw Wigner-Ville")
    plot_tfr(W_sp, "Smoothed Pseudo WVD")
    plot_tfr(W_cw, "Choi-Williams")
    plot_tfr(W_gauss, "Gaussian Kernel WVD")