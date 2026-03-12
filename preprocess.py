import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

def get_data(file):
   
    # Read CSV
    data = pd.read_csv(file)
    
    for col in data:
        data[col] = data[col].str.replace(',', '.')   
    data = data.astype(float)
    
    return data

def plot(x, y, downsampling=1, name="plot"):
    

    folder = "plots"
    os.makedirs(folder, exist_ok=True)

    # Convert Pandas to NumPy
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x_plot = x.to_numpy().squeeze()
    else:
        x_plot = np.array(x)

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y_plot = y.to_numpy().squeeze()
    else:
        y_plot = np.array(y)

    plt.figure(figsize=(8,4))
    if y_plot.ndim == 1:
        plt.plot(x_plot[::downsampling], y_plot[::downsampling], linewidth=1)
    elif y_plot.ndim == 2:
        # 2D spectrogram plot
        plt.pcolormesh(x_plot, np.arange(y_plot.shape[0]), y_plot, shading='gouraud')
        plt.xlabel("Time")
        plt.ylabel("Frequency bin")
        plt.colorbar(label='Amplitude')
    else:
        raise ValueError("y_plot must be 1D or 2D")   

    plt.title(name)
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filepath}")



