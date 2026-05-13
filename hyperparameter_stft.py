"""
═══════════════════════════════════════════════════════════════════════════════
  hyperparameter_stft.py  — standalone STFT+SST evaluation based on hyperparameters
  ─────────────────────────────────────────────────
  Run from the project root (same folder as preprocess.py / decomp.py).

  What this script produces
  ─────────────────────────
  plots/sst_stft_comparison.png   — STFT original vs SST side-by-side
  plots/sst_stft_reconstruction.png — base + harmonic band recon (STFT)

  All parameters are grouped at the top — edit freely.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# ── make sure the project root is on the path ──────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocess
from transforms import SST_v2_processing          


# ═══════════════════════════════════════════════════════════════════════════
#  USER PARAMETERS  — edit here
# ═══════════════════════════════════════════════════════════════════════════

# --- signal construction (mirrors despair.py) -------------------------------
noise_level = 0            # 0 = clean, 1.5 = 150% noise
beta        = 6            # non-linearity parameter

A1_mode = "S2 Propagated signal (nm)"   # base harmonic mode for β
A2_mode = "S4 Propagated signal (nm)"   # second harmonic mode for β

dataset_base      = "Data/In-plane_TemporalResponse@7.9866MHzmm@200mm.xlsx"
dataset_harmonic   = "Data/In-plane_A2_TemporalResponse@15.963MHzmm@200mm.xlsx"

modes_base = ["S2 Propagated signal (nm)","A1 Propagated signal (nm)","A4 Propagated signal (nm)"]
modes_harmonic = ["S2 Propagated signal (nm)", "S4 Propagated signal (nm)", "A1 Propagated signal (nm)","A4 Propagated signal (nm)"]

# --- time-frequency analysis -----------------------------------------------
f_min_analyse = 1.0e6      # Hz — lower bound for TF display
f_max_analyse = 4.5e6      # Hz — upper bound for TF display
n_freq        = 400        # frequency bins (CWT)

band_min_base      = 1_100_000   # Hz
band_max_base      = 1_500_000
band_min_harmonic  = 2_300_000
band_max_harmonic  = 2_900_000

# --- STFT parameters (from blind-decomp stage 1) ---------------------------
stft_win_len = 256     # samples
stft_hop_len = 20
stft_n_fft   = 2048

# --- SST thresholds --------------------------------------------------------
stft_gamma = 1e-6      # STFT bins weaker than this are not reassigned

# --- plot flags ------------------------------------------------------------
log_scale = True       # dB colour scale in TF plots


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & BUILD SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n[0] Loading data …")
data_base      = preprocess.get_data(dataset_base)
data_harmonic  = preprocess.get_data(dataset_harmonic)

print("[1] Building composite signal …")
t, signal, second_scale = preprocess.create_signal(
    data_base, data_harmonic,
    beta, noise_level,
    A1_mode, A2_mode,
    modes_base, modes_harmonic,
)
print(f"    samples={len(t)}   t=[{t[0]:.2f}, {t[-1]:.2f}] µs   "
      f"second_scale={second_scale:.4f}")

# ── Ground truth components (Option A — no changes to preprocess.py) ────────
# Reconstruct exactly what create_signal summed, but keep the two parts separate.
# This mirrors the internal logic of preprocess.create_signal.

t_base = data_base["Propagation time (micsec)"].to_numpy()
t_harm = data_harmonic["Propagation time (micsec)"].to_numpy()

gt_base = np.zeros(len(t))
for mode in modes_base:
    gt_base += np.interp(t, t_base, data_base[mode].to_numpy())

gt_harmonic = np.zeros(len(t))
for mode in modes_harmonic:
    gt_harmonic += second_scale * np.interp(t, t_harm, data_harmonic[mode].to_numpy())

print(f"    GT base peak    : {np.max(np.abs(gt_base)):.4f} nm")
print(f"    GT harmonic peak: {np.max(np.abs(gt_harmonic)):.4f} nm")


# ═══════════════════════════════════════════════════════════════════════════
#  STFT-SST
# ═══════════════════════════════════════════════════════════════════════════
def stft_sst(t,signal,f_min_analyse,f_max_analyse,stft_win_len,stft_hop_len,stft_n_fft,stft_gamma,band_min_base,band_max_base,log_scale,gt_base,gt_harmonic,plot=True):
    #print("\n[2] STFT-SST …")
    S_orig, S_sst, f_stft, t_stft = SST_v2_processing.stft_sst(
        t, signal,
        fmin=f_min_analyse,
        fmax=f_max_analyse,
        win_len=stft_win_len,
        hop_len=stft_hop_len,
        n_fft=stft_n_fft,
        gamma=stft_gamma,)

    if plot:
        # side-by-side spectrogram comparison
        SST_v2_processing.plot_comparison(
            t_stft, f_stft,
            S_orig, S_sst,
            method="STFT",
            name="sst_stft_comparison",
            log_scale=log_scale,
            fmin=f_min_analyse,
            fmax=f_max_analyse,)
    
    #print(f"    STFT shape: {S_orig.shape}   SST shape: {S_sst.shape}")
    # band reconstructions
    #print("    Reconstructing base band (STFT) …")
    recon_base_stft = SST_v2_processing.reconstruct_band_stft(
        t, signal,
        band_min=band_min_base,
        band_max=band_max_base,
        fmin=f_min_analyse,
        fmax=f_max_analyse,
        win_len=stft_win_len,
        hop_len=stft_hop_len,
        n_fft=stft_n_fft,)

    #print("    Reconstructing harmonic band (STFT) …")
    recon_harmonic_stft = SST_v2_processing.reconstruct_band_stft(
        t, signal,
        band_min=band_min_harmonic,
        band_max=band_max_harmonic,
        fmin=f_min_analyse,
        fmax=f_max_analyse,
        win_len=stft_win_len,
        hop_len=stft_hop_len,
        n_fft=stft_n_fft,)
    
    if plot:
        SST_v2_processing.plot_reconstruction(
            t, signal,
            recon_base_stft,
            recon_harmonic_stft,
            gt_base=gt_base,
            gt_harmonic=gt_harmonic,
            method="STFT",
            name="sst_stft_reconstruction",)

    return recon_harmonic_stft,recon_base_stft


#find and plot errors = original signal - reconstructed signal
def get_residuals(r_h,r_b,og_h,og_b,plot=False):
    og_full = og_h+og_b
    r_full = r_h+r_b 
    h_error = abs(r_h-og_h)
    b_error = abs(r_b-og_b)
    full_error = abs(r_full-og_full)

    if plot: 
        fig,ax = plt.subplots(1,2,figsize=(12,5))
        ax[0].plot(t,h_error,color="blue",alpha=0.5,label='Harmonic Error')
        ax[0].plot(t,b_error,color='red',alpha=0.5,label='Base Error')
        ax[0].set_title("Base & Harmonic Error")
        ax[0].legend()
        ax[1].plot(t, full_error)
        ax[1].set_title("Total Error")

        plt.tight_layout()
        plt.savefig("plots/abs_errors.png", dpi=300)
    return h_error,b_error,full_error


# get the sum of the absolute error 
def process_error(recon_harmonic,recon_base):
    harmonic_error, base_error, total_error = get_residuals(
        recon_harmonic,recon_base,gt_harmonic,gt_base)

    sum_h_error = np.sum(harmonic_error)
    sum_b_error = np.sum(base_error)
    sum_t_error = np.sum(total_error) 
    #print(f"Total error: {sum_t_error}, base error: {sum_b_error}, harmonic error: {sum_h_error}")
    return sum_h_error,sum_b_error,sum_t_error

        





# =============================================================
# Execute
# =============================================================

#plot signal
a,b = stft_sst(t,signal,f_min_analyse,f_max_analyse,stft_win_len,stft_hop_len,stft_n_fft,stft_gamma,band_min_base,band_max_base,log_scale,gt_base,gt_harmonic,plot=True)
sum_h_error,sum_b_error,sum_t_error = process_error(a,b)
print(f"harmonic error: {sum_h_error}, base error: {sum_b_error}, total error: {sum_t_error}")

#options: stft_win_len, stft_hop_len, stft_n_fft
parameter = "none"
#options: win_len = 50, hop_len = 1, n_fft = 1
eval_min = 7
#options: win_len = 200, hop_len = 10, n_fft = 10
eval_max = 14

#stft_win_len = 128     # samples
#stft_hop_len = 2
#stft_n_fft   = 512

if parameter == "stft_hop_len":
    h_error_lst = []
    b_error_lst = []
    t_error_lst = []
    par = []
    #loop over all values of parameters to consider
    for i in range(eval_min,eval_max):
        #get the reconstructed signal
        recon_harmonic_stft,recon_base_stft = stft_sst(
            t,signal,f_min_analyse,f_max_analyse,
            stft_win_len,i,stft_n_fft,
            stft_gamma,band_min_base,band_max_base,
            log_scale,gt_base,gt_harmonic,plot=False)
        #get the summed errors and add them to the list
        sum_h_error,sum_b_error,sum_t_error = process_error(recon_harmonic_stft,recon_base_stft)
        h_error_lst.append(sum_h_error)
        b_error_lst.append(sum_b_error)
        t_error_lst.append(sum_t_error)
        par.append(i)
    #plot the errors
    plt.figure(figsize=(12, 5))
    plt.plot(par,h_error_lst,color="blue",label="Harmonic Error")
    plt.plot(par,b_error_lst,color="red",label="Base Error")
    plt.plot(par,t_error_lst,color="green",label="Total Error")
    plt.xlabel("Hop Length")
    plt.ylabel("Summed Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/errors_{parameter}.png", dpi=300)


if parameter == "stft_win_len":
    h_error_lst = []
    b_error_lst = []
    t_error_lst = []
    par = []
    #loop over all values of parameters to consider
    for i in range(eval_min,eval_max):
        #get the reconstructed signal
        recon_harmonic_stft,recon_base_stft = stft_sst(
            t,signal,f_min_analyse,f_max_analyse,
            i,stft_hop_len,stft_n_fft,
            stft_gamma,band_min_base,band_max_base,
            log_scale,gt_base,gt_harmonic,plot=False)
        #get the summed errors and add them to the list
        sum_h_error,sum_b_error,sum_t_error = process_error(recon_harmonic_stft,recon_base_stft)
        h_error_lst.append(sum_h_error)
        b_error_lst.append(sum_b_error)
        t_error_lst.append(sum_t_error)
        par.append(i)
    #plot the errors
    plt.figure(figsize=(12, 5))
    plt.plot(par,h_error_lst,color="blue",label="Harmonic Error")
    plt.plot(par,b_error_lst,color="red",label="Base Error")
    plt.plot(par,t_error_lst,color="green",label="Total Error")
    plt.xlabel("Window Length")
    plt.ylabel("Summed Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/errors_{parameter}.png", dpi=300)


if parameter == "stft_n_fft":
    h_error_lst = []
    b_error_lst = []
    t_error_lst = []
    par = []
    #loop over all values of parameters to consider
    for i in range(eval_min,eval_max):
        k=2**i
        #get the reconstructed signal
        recon_harmonic_stft,recon_base_stft = stft_sst(
            t,signal,f_min_analyse,f_max_analyse,
            stft_win_len,stft_hop_len,k,
            stft_gamma,band_min_base,band_max_base,
            log_scale,gt_base,gt_harmonic,plot=False)
        #get the summed errors and add them to the list
        sum_h_error,sum_b_error,sum_t_error = process_error(recon_harmonic_stft,recon_base_stft)
        h_error_lst.append(sum_h_error)
        b_error_lst.append(sum_b_error)
        t_error_lst.append(sum_t_error)
        par.append(k)
    #plot the errors
    plt.figure(figsize=(12, 5))
    plt.plot(par,h_error_lst,color="blue",label="Harmonic Error",marker='o')
    plt.plot(par,b_error_lst,color="red",label="Base Error",marker='o')
    plt.plot(par,t_error_lst,color="green",label="Total Error",marker='o')
    plt.xlabel("n_fft")
    plt.ylabel("Summed Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/errors_{parameter}.png", dpi=300)

print("\nDone — all plots saved to plots/")

