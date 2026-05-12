import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt 

file_path = Path(__file__).parent / 'Data'/ "In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"

df = pd.read_excel(file_path)
# Time column
time = df["Propagation time (micsec)"]

# Mode columns: all A/S modes, excluding excitation and sum
mode_cols = [
    col for col in df.columns
    if "Propagated signal" in col and "Sum" not in col
]

# Sum column
sum_col = "Sum Propagated signal (nm)"


# --------------------------------------------------
# 1. Plot only the summed signal
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(time, df[sum_col], linewidth=1.5)
plt.xlabel("Propagation time (µs)")
plt.ylabel("Amplitude (nm)")
plt.title("Summed propagated signal")
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. Plot each individual mode separately
# --------------------------------------------------
for col in mode_cols:
    plt.figure(figsize=(10, 4))
    plt.plot(time, df[col], linewidth=1.2)
    plt.xlabel("Propagation time (µs)")
    plt.ylabel("Amplitude (nm)")
    plt.title(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 3. Plot all individual modes overlapping
# --------------------------------------------------
plt.figure(figsize=(12, 6))

for col in mode_cols:
    plt.plot(time, df[col], linewidth=1.0, label=col.split()[0])

plt.xlabel("Propagation time (µs)")
plt.ylabel("Amplitude (nm)")
plt.title("Overlapping individual propagated modes")
plt.grid(True)
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 4. Optional: plot sum together with all modes
# --------------------------------------------------
plt.figure(figsize=(12, 6))

for col in mode_cols:
    plt.plot(time, df[col], linewidth=0.8, alpha=0.6, label=col.split()[0])

plt.plot(time, df[sum_col], color="black", linewidth=2.0, label="Sum")

plt.xlabel("Propagation time (µs)")
plt.ylabel("Amplitude (nm)")
plt.title("Individual modes with summed signal")
plt.grid(True)
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()