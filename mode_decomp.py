import numpy as np
import matplotlib.pyplot as plt
import preprocess

def estimate_group_velocity(signal, fs, L, fc,
                            bw=20e3, win_len=2048, hop=None):
    """
    signal : 1D numpy array, recorded Lamb-wave signal
    fs     : sampling frequency (Hz)
    L      : source–sensor distance (m)
    fc     : center frequency for analysis (Hz)
    bw     : half-bandwidth around fc (Hz)
    """
    if hop is None:
        hop = win_len // 4

    signal = np.asarray(signal)
    N = len(signal)

    # Build STFT frames using sliding windows
    win = np.hanning(win_len)
    n_frames = 1 + (N - win_len) // hop
    frames = np.empty((n_frames, win_len))

    for i in range(n_frames):
        start = i * hop
        frames[i, :] = signal[start:start + win_len] * win

    # FFT
    STFT = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(win_len, 1.0 / fs)
    t_mid = (np.arange(n_frames) * hop + win_len // 2) / fs

    # Energy around fc
    band = (freqs >= fc - bw) & (freqs <= fc + bw)
    energy = np.abs(STFT[:, band]) ** 2
    env = energy.sum(axis=1)

    # Arrival time = position of max envelope
    idx = np.argmax(env)
    t_arrival = t_mid[idx]

    cg = L / t_arrival
    return cg, t_arrival, t_mid, env

# Practice

data = preprocess.get_data(r"Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.csv")


t = data["Propagation time (micsec)"].to_numpy()
y = data["Sum Propagated signal (nm)"].to_numpy()

t_s = t * 10e-6
# Sampling frequency
dt = t_s[1] - t_s[0]
fs = 1.0 / dt

cg, t_arr, t_env, env = estimate_group_velocity(y, fs, L=0.2, fc=300e3)

plt.figure()
plt.plot(t_env * 1e6, env)          # convert seconds back to µs
plt.axvline(t_arr * 1e6, color="r", linestyle="--",
            label=f"Picked arrival ≈ {t_arr*1e6:.1f} µs")
plt.xlabel("Time (µs)")
plt.ylabel("Narrowband energy")
#plt.title(f"Energy envelope around {fc/1e3:.0f} kHz")
plt.legend()
plt.grid(True)
plt.show()