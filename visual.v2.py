import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

file_path = Path(__file__).parent / "Data" / "In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"

df = pd.read_excel(file_path)

time = df["Propagation time (micsec)"]

excitation_col = "ExcitationSignal"

mode_cols = [
    col for col in df.columns
    if "Propagated signal" in col and "Sum" not in col
]

print(mode_cols)

sum_col = "Sum Propagated signal (nm)"
# 

# ==================================================
# 1. Sum and all individual modes as subplots
# ==================================================

n_plots = 1 + len(mode_cols)   # one for sum + one for each mode

fig, axes = plt.subplots(
    n_plots,
    1,
    figsize=(10, 0.75 * n_plots),
    sharex=True
)

# In case there is only one subplot
if n_plots == 1:
    axes = [axes]

# First subplot: summed signal
axes[0].plot(time, df[sum_col], linewidth=1.5)
axes[0].set_title("Summed propagated signal")
axes[0].set_ylabel("Amplitude (nm)")
axes[0].grid(True)

# Remaining subplots: individual modes
for ax, col in zip(axes[1:], mode_cols):
    mode_name = col.split()[0]

    ax.plot(time, df[col], linewidth=1.0)
    ax.set_title(mode_name)
    ax.set_ylabel("Amplitude (nm)")
    ax.grid(True)

# Shared x-axis label
axes[-1].set_xlabel("Propagation time (µs)")

plt.tight_layout()
plt.show()


# ==================================================
# 2. Optional: all individual modes overlapping
# ==================================================

plt.figure(figsize=(12, 6))

for col in mode_cols:
    mode_name = col.split()[0]
    plt.plot(time, df[col], linewidth=1.0, label=mode_name)

plt.xlabel("Propagation time (µs)")
plt.ylabel("Amplitude (nm)")
plt.title("Overlapping individual propagated modes")
plt.grid(True)
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()


# ==================================================
# 3. Optional: sum + modes overlapping
# ==================================================

plt.figure(figsize=(12, 6))

for col in mode_cols:
    mode_name = col.split()[0]
    plt.plot(time, df[col], linewidth=0.8, alpha=0.6, label=mode_name)

plt.plot(time, df[sum_col], linewidth=2.0, label="Sum")

plt.xlabel("Propagation time (µs)")
plt.ylabel("Amplitude (nm)")
plt.title("Individual modes with summed signal")
plt.grid(True)
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# ==================================================
# 4. Optional: excitation + sum + modes overlapping
# ==================================================

plt.figure(figsize=(12, 6))

plt.plot(time, df[excitation_col], linewidth=1.5, label="Excitation")

for col in mode_cols:
    mode_name = col.split()[0]
    plt.plot(time, df[col], linewidth=0.8, alpha=0.6, label=mode_name)

plt.plot(time, df[sum_col], linewidth=2.0, label="Sum")

plt.xlabel("Propagation time (µs)")
plt.ylabel("Amplitude (nm)")
plt.title("Excitation, individual modes, and summed signal")
plt.grid(True)
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()
