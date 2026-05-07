import numpy as np
import pandas as pd
from preprocess import get_data
import matplotlib.pyplot as plt  
from vmdpy import VMD


data = get_data(r"Data\In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx")

#. some sample parameters for VMD  
alpha = 2000       # moderate bandwidth constraint  
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 3              # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7 

#. Run actual VMD code  
u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol) 
