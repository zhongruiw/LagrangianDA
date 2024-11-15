"""
Lagrangian DA for the 2-layer QG system
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, norm
from Lagrangian_tracer import Lagrange_tracer_model
from conj_symm_tools import verify_conjugate_symmetry, find_non_conjugate_pairs, avg_conj_symm, map_conj_symm
from Lagrangian_DA import Lagrangian_DA_OU, Lagrangian_DA_CG, mu2psi, mu2layer, R2layer, relative_entropy_psi_k
from ene_spectrum import ene_spectrum, adjust_ik, trunc2full
from LSM_QG import solve_eigen, calibrate_OU, run_OU, eigen2layer, layer2eigen, growth_rate
from mode_truc import inv_truncate, truncate
from plot import ifftnroll, psi2q, plot_contour_fields, plot_psi_k_seriespdf, plot_layer_seriespdf, plot_psi1_k_seriespdf, plot_rmses, loop_ifft2_var, ifft2_var, plot_mog, plot_mog_k, scatterplot, calculate_skewness_kurtosis
from statsmodels.tsa.stattools import acf, ccf
from scipy.optimize import curve_fit
from scipy.io import loadmat
from scipy import sparse
import h5py
from time import time

# fix the random seed
np.random.seed(2024)



# load data
data_path = '../qg/QG_DATA_topo40_nu1e-12_beta22_K128_dt2e-3_subs.mat'
with h5py.File(data_path, 'r') as file:
    print("Keys: %s" % file.keys())
    # psi1_k_t = np.transpose(file['psi_1_t'][()], axes=(2, 1, 0)) # reorder the dimensions from Python's row-major order back to MATLAB's column-major order 
    psi2_k_t = np.transpose(file['psi_2_t'][()], axes=(2, 1, 0)) # reorder the dimensions from Python's row-major order back to MATLAB's column-major order 
    psi1_k_t_fine = np.transpose(file['psi_1_t_fine'][()], axes=(2, 1, 0)) # reorder the dimensions from Python's row-major order back to MATLAB's column-major order 
    dt = file['dt'][()][0,0]
    s_rate = int(file['s_rate'][()][0,0])
    params_dataset = file['params']
    nu = params_dataset['nu'][()] [0,0]
    topo = params_dataset['H'][()] [0,0]
    kd = params_dataset['kd'][()] [0,0]
    U = params_dataset['U'][()] [0,0]
    kb = params_dataset['kb'][()] [0,0]
    kappa = params_dataset['r'][()] [0,0]
    beta = kb**2
    K = int(params_dataset['N'][()] [0,0])
    H = params_dataset['H'][()] [0,0]
    topo = np.transpose(file['topo'][()], axes=(1,0))
# psi1_k_t = psi1_k_t['real'] + 1j * psi1_k_t['imag']
psi2_k_t = psi2_k_t['real'] + 1j * psi2_k_t['imag']
psi1_k_t_fine = psi1_k_t_fine['real'] + 1j * psi1_k_t_fine['imag']
h_hat = np.fft.fft2(topo)

temp = np.zeros((33,33,200000), dtype='complex')
temp[:16, :16, :] = psi1_k_t_fine[:16, :16, :]
temp[-15:, -15:, :] = psi1_k_t_fine[-15:, -15:, :]
temp[:16, -15:, :] = psi1_k_t_fine[:16, -15:, :]
temp[-15:, :16, :] = psi1_k_t_fine[-15:, :16, :]
psi1_k_t_fine = temp

# truncate parameter
r_cut = 16
style = 'circle'

# load data of LSM
eigens = np.load('../data/eigens_K128_beta22.npz')
omega1 = eigens['omega1']
omega2 = eigens['omega2']
r1 = eigens['r1']
r2 = eigens['r2']
est_params = np.load('../data/est_paras_ou_K128_beta22_tr.npz')
gamma_est = est_params['gamma']
omega_est = est_params['omega']
f_est = est_params['f']
sigma_est = est_params['sigma']
# est_params = np.load('../data/est_paras_cn_ou_K128_beta22_tr.npz')
# sigma_cn = est_params['sigma']
# cov_cn = est_params['cov']

# Lagrangian DA
N_chunk = 10000
N = 200000
lsm_da = Lagrangian_DA_OU(K, r1, r2, f_est, gamma_est, omega_est, sigma_est, r_cut, style)
mu_t_lsm, R_t_lsm = lsm_da.forward(N, N_chunk, dt, s_rate=1, tracer=False, psi1_k_t=psi1_k_t_fine, psi2_k_t0=psi2_k_t[:, :, 0])

# save data
da_pos = {
    'mu_t': mu_t_lsm,
    'R_t': R_t_lsm,
    'r_cut':r_cut,
    'style':style,
    'dt': dt
}
np.savez('../data/LSMDA_pos_K128_beta22_tr16_flow.npz', **da_pos)