import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from numpy.linalg import lstsq
import pandas as pd
from preprocess import get_data 
from scipy.signal import stft
import matplotlib.pyplot as plt

data = get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx")


#Step 1
#TFR  # shape: (Nt, Nf) → time-frequency representation
#t    # time vector (Nt,)
#f    # frequency vector (Nf,)
#signal  # original time signal y(t)

t = data["Propagation time (micsec)"].values * 1e-6  # convert to seconds
signal = data["Sum Propagated signal (nm)"].values

dt = t[1] - t[0]
fs = 1 / dt


f, tt, Zxx = stft(signal, fs=fs, nperseg=256, noverlap=200)
TFR = np.abs(Zxx).T  # shape (time, freq)



#Step 2

def hessian_enhancement(TFR, sigma=1.0):
    """
    Enhance dispersion curves using Hessian-based method
    """
    # Smooth (important for stability)
    S = gaussian_filter(TFR, sigma=sigma)

    # Second derivatives
    S_tt = sobel(sobel(S, axis=0), axis=0)   # ∂²/∂t²
    S_ff = sobel(sobel(S, axis=1), axis=1)   # ∂²/∂f²
    S_tf = sobel(sobel(S, axis=0), axis=1)   # ∂²/∂t∂f

    Nt, Nf = S.shape
    V = np.zeros_like(S)

    for i in range(Nt):
        for j in range(Nf):
            # Hessian matrix
            H = np.array([
                [S_tt[i, j], S_tf[i, j]],
                [S_tf[i, j], S_ff[i, j]]
            ])

            # Eigenvalues
            eigvals = np.linalg.eigvalsh(H)
            l1, l2 = eigvals

            # Avoid division issues
            denom = (l1**2 + l2**2)
            if denom == 0:
                continue

            # Enhancement function (simplified version of paper eq. 6)
            V[i, j] = (2 * abs((l1 - l2) * l2)) / (denom + 1e-12)

    # Normalize to [0,1]
    V = V / np.max(V)

    return V




#Step 3
def extract_ridge(V, f, w=0.01, a=10):
    """
    Extract one ridge using continuity-constrained tracking
    """
    Nt, Nf = V.shape
    ridge = np.zeros(Nt, dtype=int)

    # Initialize: pick strongest frequency at first time
    ridge[0] = np.argmax(V[0])

    for n in range(1, Nt):
        prev_f = ridge[n-1]

        best_score = -np.inf
        best_k = prev_f

        for k in range(Nf):
            # Continuity constraint
            if abs(k - prev_f) > a:
                continue

            score = V[n, k]**2 - w * abs(k - prev_f)

            if score > best_score:
                best_score = score
                best_k = k

        ridge[n] = best_k

    return f[ridge], ridge



def extract_multiple_ridges(V, f, num_modes=15):
    V_copy = V.copy()
    ridges = []

    for _ in range(num_modes):
        freq_ridge, idx_ridge = extract_ridge(V_copy, f)
        ridges.append((freq_ridge, idx_ridge))

        # Suppress this ridge (to find others)
        for t in range(len(idx_ridge)):
            V_copy[t, max(0, idx_ridge[t]-2):idx_ridge[t]+3] *= 0

    return ridges



#Step 4


def reconstruct_mode_local(signal, t, tt, freq_ridge, window_size=200):
    """
    Reconstruct mode with time-varying amplitude
    """
    freq_interp = np.interp(t, tt, freq_ridge)

    dt = t[1] - t[0]
    phase = 2 * np.pi * np.cumsum(freq_interp) * dt

    cos_term = np.cos(phase)
    sin_term = np.sin(phase)

    Nt = len(t)
    mode = np.zeros(Nt)
    amplitude = np.zeros(Nt)

    half_w = window_size // 2

    for i in range(Nt):
        start = max(0, i - half_w)
        end = min(Nt, i + half_w)

        A = np.vstack([
            cos_term[start:end],
            sin_term[start:end]
        ]).T

        y = signal[start:end]

        if len(y) < 10:
            continue

        coeffs, _, _, _ = lstsq(A, y, rcond=None)

        # reconstruct center point
        mode[i] = (
            coeffs[0] * cos_term[i] +
            coeffs[1] * sin_term[i]
        )

        amplitude[i] = np.sqrt(coeffs[0]**2 + coeffs[1]**2)

    return mode, amplitude


V = hessian_enhancement(TFR)
ridges = extract_multiple_ridges(V, f, num_modes=15)

modes = []
amplitudes = []

for freq_ridge, _ in ridges:
    mode, amp = reconstruct_mode_local(signal, t, tt, freq_ridge)
    modes.append(mode)
    amplitudes.append(amp)


def detect_presence(amplitude, threshold_ratio=0.1):
    threshold = threshold_ratio * np.max(amplitude)
    mask = amplitude > threshold
    return mask


reconstructed = np.sum(modes, axis=0)

plt.figure()
plt.plot(t, signal, label="Measured")
plt.plot(t, reconstructed, '--', label="Reconstructed")
plt.legend()
plt.title("Measured vs Reconstruction")
plt.show()


fig, ax = plt.subplots(figsize=(10,6))

lines = []

# Optional upgrade: physical mode labels
mode_labels = [
    "A0","A1","A2","A3","A4","A5","A6","A7",
    "S0","S1","S2","S3","S4","S5","S6","S7","S8"
]

# Plot modes
for i, mode in enumerate(modes):
    if i < len(mode_labels):
        label = mode_labels[i]
    else:
        label = f"Mode {i}"
        
    line, = ax.plot(t, mode, label=label, alpha=0.7)
    lines.append(line)

# Plot measured signal
measured_line, = ax.plot(t, signal, 'k', label="Measured", linewidth=2)
lines.append(measured_line)

# Create legend
leg = ax.legend(loc="upper right", fancybox=True, shadow=True)

# Map legend lines to original lines
lined = {}
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(5)  # clickable with tolerance
    lined[legline] = origline

# Click event function
def on_pick(event):
    legline = event.artist
    origline = lined[legline]

    # Toggle visibility
    visible = not origline.get_visible()
    origline.set_visible(visible)

    # Fade legend entry if hidden
    legline.set_alpha(1.0 if visible else 0.2)

    fig.canvas.draw()

# Connect event
fig.canvas.mpl_connect('pick_event', on_pick)

# Labels and title
ax.set_title("Individual Modal Contributions (click legend to toggle)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

plt.show()


true_modes = {}
mode_names = [
    "A0","A1","A2","A3","A4","A5","A6","A7",
    "S0","S1","S2","S3","S4","S5","S6","S7","S8"
]

for name in mode_names:
    col = f"{name} Propagated signal (nm)"
    if col in data.columns:
        true_modes[name] = data[col].values

matches = []

for i, mode in enumerate(modes):
    best_match = None
    best_corr = -np.inf

    for name, true_mode in true_modes.items():
        corr = np.corrcoef(mode, true_mode)[0,1]

        if corr > best_corr:
            best_corr = corr
            best_match = name

    matches.append((i, best_match, best_corr))


fig, ax = plt.subplots(figsize=(12,7))

lines = []

for i, name, corr in matches:
    extracted = modes[i]
    true = true_modes[name]

    # Plot extracted
    line1, = ax.plot(t, extracted, '--',
                     label=f"Extracted {i} → {name} (corr={corr:.2f})",
                     alpha=0.8)

    # Plot true
    line2, = ax.plot(t, true,
                     label=f"True {name}",
                     alpha=0.5)

    lines.append(line1)
    lines.append(line2)

# Legend
leg = ax.legend(loc="upper right", fontsize=8)

# Make clickable
lined = {}
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(5)
    lined[legline] = origline

def on_pick(event):
    legline = event.artist
    origline = lined[legline]

    visible = not origline.get_visible()
    origline.set_visible(visible)

    legline.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', on_pick)

ax.set_title("Extracted vs True Mode Comparison (click to toggle)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

plt.show()