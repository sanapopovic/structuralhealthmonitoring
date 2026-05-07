import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed as sk_watershed
# ================= USER OPTION =================
N_REGIONS = 9   # set integer (e.g. 10) or None
# ==============================================

# --- path fix ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transforms import Hilbert_Huang_processing
import preprocess


# ============================================================
# --- LOAD DATA ---
# ============================================================
data = preprocess.get_data(
    r"Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
)

t = data["Propagation time (micsec)"].to_numpy()
signal = data["Sum Propagated signal (nm)"].to_numpy()

dt = np.mean(np.diff(t))
fs = (1.0 / dt) * 1e6


# ============================================================
# --- HHT ---
# ============================================================
imfs, residue = Hilbert_Huang_processing.emd(signal)
inst_amp, inst_freq = Hilbert_Huang_processing.hilbert_analysis(imfs, fs)

fig, ax, H, T, F = Hilbert_Huang_processing.plot_hilbert_spectrum(inst_freq, inst_amp, t, fs)


def watershed_hilbert_spectrum(H, T, F, n_regions=5, smooth_sigma=0.5):
    """
    Apply watershed segmentation directly to a Hilbert spectrum.

    Parameters
    ----------
    H : 2D array
        Hilbert spectrum (time-frequency energy density)
    T, F : 2D arrays
        Meshgrid coordinates (from plot function)
    n_regions : int
        Desired number of regions
    smooth_sigma : float
        Gaussian smoothing level

    Returns
    -------
    labels : 2D array
        Watershed segmentation labels
    fig, ax : matplotlib objects
    """

    # --- Log scaling (stabilizes dynamic range) ---
    H_log = np.log1p(H)

    # --- Smooth to reduce noise ---
    H_smooth = gaussian_filter(H_log, sigma=smooth_sigma)

    # --- Find markers (peaks in energy) ---
    coords = peak_local_max(
        H_smooth,
        num_peaks=n_regions,
        exclude_border=False
    )

    markers = np.zeros_like(H_smooth, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # --- Watershed (invert so peaks become basins) ---
    labels = sk_watershed(-H_smooth, markers)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    pcm = ax.pcolormesh(T, F, H_log, shading="auto", cmap="viridis")

    # Overlay watershed boundaries
    ax.contour(T, F, labels, linewidths=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Hilbert Spectrum + Watershed ({n_regions} regions)")

    fig.colorbar(pcm, ax=ax, label="Log Energy")

    plt.tight_layout()
    plt.show()

    return labels, fig, ax




labels, fig2, ax2 = watershed_hilbert_spectrum(H, T, F, n_regions=N_REGIONS)


