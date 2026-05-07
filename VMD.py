import numpy as np
import pandas as pd
from preprocess import get_data
import matplotlib.pyplot as plt  
from vmdpy import VMD


data = get_data(r"Data\In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx")

time = data.iloc[:,0] 
t = time.to_numpy() #Numpy array of time in microsec
print(t)


"""
#. some sample parameters for VMD  
alpha = 2000       # moderate bandwidth constraint  
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 3              # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7 

#. Run actual VMD code  
u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol) 


# 4. Plot results
# -----------------------------
plt.figure(figsize=(10, 6))
plt.subplot(K+1, 1, 1)
plt.plot(t, signal, 'k')
plt.title("Original Signal")
plt.xlabel("Time [s]")

for i in range(K):
    plt.subplot(K+1, 1, i+2)
    plt.plot(t, u[i, :])
    plt.title(f"Mode {i+1} (Center freq: {omega[i]:.2f} Hz)")
    plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()

"""