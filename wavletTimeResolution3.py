import pywt

import numpy as np

import matplotlib.pyplot as plt
 
fs = 52e6
f0 = 1.33e6
dt = 1 / fs
 
parameter = "wavelet_bandwidth"
 
b_start, b_end, b_step = 0.5, 25.0, 0.5

c_start, c_end, c_step = 0.1, 15.0, 0.5
 
if parameter == "wavelet_bandwidth":

    test_range = np.arange(b_start, b_end, b_step)

    xlabel = "Wavelet Bandwidth (B)"

elif parameter == "wavelet_center_freq":

    test_range = np.arange(c_start, c_end, c_step)

    xlabel = "Wavelet Center Frequency (C)"
 
time_res = []
 
for val in test_range:

    if parameter == "wavelet_bandwidth":

        wavelet_label = f"cmor{val:.1f}-1.0"

    else:

        wavelet_label = f"cmor1.0-{val:.1f}"
 
    W_obj = pywt.ContinuousWavelet(wavelet_label)
 
    fc = pywt.central_frequency(W_obj)

    a  = fc / (f0 * dt)
 
    psi, x = W_obj.wavefun(level=10)

    x_norm = x / (W_obj.upper_bound - W_obj.lower_bound)
 
    psi = np.abs(psi) ** 2

    dx = x_norm[1] - x_norm[0]

    psi /= (psi.sum() * dx)
 
    t_phys = x_norm * a * dt * 1e6

    mean_t = np.sum(t_phys * psi) * dx

    width  = 2 * np.sqrt(np.sum(((t_phys - mean_t) ** 2) * psi) * dx)

    time_res.append(width)
 
plt.figure(figsize=(8, 4))

plt.plot(test_range, time_res, color='blue', linestyle='-', marker='o', markersize=4)

plt.xlabel(xlabel)

plt.ylabel("Time Resolution (µs)")

plt.title(f"Time Resolution vs {xlabel}  |  f₀ = {f0/1e6:.2f} MHz  |  fs = {fs/1e6:.0f} MHz")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.axhline(0.0892, color='r') # vertical


plt.savefig("wavelet_time_resolution.png", dpi=150)

plt.show()
 

