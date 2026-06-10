import pywt
import numpy as np
import matplotlib.pyplot as plt  # Fixed import

parameter = "wavelet_bandwidth" 

# Range for Bandwidth (Proxy: Complex Morlet B parameter)
b_start, b_end, b_step = 0.5, 25.0, 0.5
# Range for Center Frequency (Proxy: Complex Morlet C parameter)
c_start, c_end, c_step = 2.0, 15.0, 0.5 

if parameter == "wavelet_bandwidth":
    test_range = np.arange(b_start, b_end, b_step)
    xlabel = "Wavelet Bandwidth (B)"
elif parameter == "wavelet_center_freq":
    test_range = np.arange(c_start, c_end, c_step)
    xlabel = "Wavelet Center Frequency (C)"

std_dev = []

for val in test_range:
    if parameter == "wavelet_bandwidth":
        wavelet_label = f"cmor{val:.1f}-3.0"
    else:
        wavelet_label = f"cmor3.0-{val:.1f}"
        
    W_obj = pywt.ContinuousWavelet(wavelet_label)
    
    psi, x = W_obj.wavefun(level=10)
    
    std_dev.append(np.std(psi))

# Plotting the results
plt.figure(figsize=(8, 4))
plt.plot(test_range, std_dev, color='blue', linestyle='-', marker='o', markersize=4)
plt.xlabel(xlabel)
plt.ylabel("Standard Deviation")
plt.title(f"Wavelet Standard Deviation vs. {parameter.replace('_', ' ').title()}")
plt.grid(True)
plt.show()

# Print the last calculated standard deviation as an example
print(f"Final evaluated wavelet string: {wavelet_label}")
print(f"Standard Deviation of the final wavelet: {std_dev[-1]:.4f}")