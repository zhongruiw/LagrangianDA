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
from Lagrangian_DA import Lagrangian_DA_OU, Lagrangian_DA_CG, mu2psi, mu2layer
from ene_spectrum import ene_spectrum, adjust_ik, trunc2full
from LSM_QG import solve_eigen, calibrate_OU, run_OU, eigen2layer, layer2eigen, growth_rate
from mode_truc import inv_truncate, truncate
from plot import ifftnroll, psi2q, plot_contour_fields, plot_psi_k_seriespdf, plot_layer_seriespdf
from statsmodels.tsa.stattools import acf, ccf
from scipy.optimize import curve_fit
from scipy.io import loadmat
import h5py
from scipy import sparse
from time import time
import gc

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
# dt = dt * s_rate
# print('psi1_k_t.shape',psi1_k_t.shape)
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

# N_chunk = 5000
# cg_da = Lagrangian_DA_CG(K, kd, beta, kappa, nu, U, h_hat, r_cut, style)
# Sigma1, Sigma2 = cg_da.calibrate_sigma(N_chunk, dt*s_rate, psi1_k_t, psi2_k_t)

# # save data
# data = {
#     'Sigma1': Sigma1,
#     'Sigma2': Sigma2
# }
# np.savez('../data/Sigma_cali_CGDA_K128_beta111_tr.npz', **data)

# load data
data = np.load('../data/Sigma_cali_CGDA_K128_beta22_tr.npz')
Sigma1 = data['Sigma1']
Sigma2 = data['Sigma2']

# Lagrangian DA
N = 100000
N_chunk = 2500
N_s = 1 # number of sample trajectories

# minimize RAM usage
psi2_k_t = psi2_k_t[:, :, -N//s_rate]
psi1_k_t_fine = psi1_k_t_fine[:, :, -N:]
gc.collect()  # Explicitly trigger garbage collection

cg_da = Lagrangian_DA_CG(K, kd, beta, kappa, nu, U, h_hat, r_cut, style)
mu_t, R_t = cg_da.forward(N, N_chunk, dt, N_s, Sigma1, Sigma2, s_rate, psi2_k_t0=psi2_k_t, psi1_k_t=psi1_k_t_fine)

# save data
da_pos = {
    'mu_t': mu_t,
    'R_t': R_t,
    'r_cut':r_cut,
    'style':style,
    'dt': dt
}
np.savez('../data/CGDA_pos_K128_beta22_tr16_flow.npz', **da_pos)