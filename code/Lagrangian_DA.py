import numpy as np
from numba import jit
from numba.typed import Dict
from numba.core import types
from scipy import sparse
from mode_truc import truncate, inv_truncate

''' 
Lagrangian DA 
    1. using tracers to recover the two-layer flow field 
    2. the reference (true) flow field is from QG model
    3. the reduced-order model is complex OU process / nonlinear conditional Gaussian model
'''

def get_A_OU(x,y,K,r1,r2,KX,KY):
    '''
    1. The mode truncation should be done outside this function. 
    2. K is the original number of modes in each x and y direction. 
    3. x, y should be of shape (N,L) for proper broadcasting.
    '''
    N, L = x.shape
    KX_flat = KX
    KY_flat = KY
    E = np.zeros((N, 2*L, KX_flat.shape[0]), dtype=np.complex_)

    exp_term = np.exp(1j * (x[:, :, None] @ KX_flat[None,:] + y[:, :, None] @ KY_flat[None,:]))
    E[:, :L,:] = exp_term * (1j) * KY_flat
    E[:, L:,:] = exp_term * (-1j) * KX_flat
    # R_bot = sparse.hstack([sparse.diags(r1.flatten(order='F')), sparse.diags(r2.flatten(order='F'))], format="csr")
    R_bot = np.hstack((np.diag(r1), np.diag(r2)))
    A = E @ R_bot / K**2

    return A


def get_A_OU_flow(psi1_k_t, A0_, a0_, F1, F2):
    A0 = A0_ * psi1_k_t + F1
    a0 = a0_ * psi1_k_t + F2
    
    return A0, a0


@jit(nopython=True)
def get_A_CG_nonlinear(K, psi1_hat, h_hat, linear_A1, linear_a1, k_index_map, list_dic_k_map, kd):
    # nonlinear summation part for A0, a0, A1 and a1
    nonlinear_sum_A0 = np.zeros_like(psi1_hat, dtype=np.complex128)
    nonlinear_sum_a0 = np.zeros_like(psi1_hat, dtype=np.complex128)
    nonlinear_sum_A1 = np.zeros_like(linear_A1, dtype=np.complex128)
    nonlinear_sum_a1 = np.zeros_like(linear_a1, dtype=np.complex128)
    
    for ik_, (k, ik) in enumerate(k_index_map.items()):
        kx, ky = k
        ikx, iky = ik
        k_sq = kx**2 + ky**2
        for im_, (m, im) in enumerate(k_index_map.items()):
            mx, my = m
            imx, imy = im
            m_sq = mx**2 + my**2
            psi1_m = psi1_hat[:, im_]
            n = (kx-mx, ky-my)
            if n in k_index_map:
                in_ = list_dic_k_map.index(n)
                psi1_n = psi1_hat[:, in_]
                h_n = h_hat[in_]
                det_mn = np.linalg.det(np.array([m, n]))
                nonlinear_sum_A0[:, ik_] -= det_mn * ((k_sq + kd**2/2) * (m_sq + kd**2/2) * psi1_n*psi1_m)
                nonlinear_sum_a0[:, ik_] -= det_mn * (kd**2/2 * (m_sq + kd**2/2) * psi1_n*psi1_m)
                nonlinear_sum_A1[:, ik_, im_] += det_mn * kd**2/2 * (k_sq * psi1_n - h_n)
                nonlinear_sum_a1[:, ik_, im_] -= det_mn * (k_sq * kd**2/2 * psi1_n + (k_sq + kd**2/2) * h_n) 

    nonlinear_sum_A0 = nonlinear_sum_A0 / K**2
    nonlinear_sum_a0 = nonlinear_sum_a0 / K**2
    nonlinear_sum_A1 = nonlinear_sum_A1 / K**2
    nonlinear_sum_a1 = nonlinear_sum_a1 / K**2
    
    return nonlinear_sum_A0, nonlinear_sum_a0, nonlinear_sum_A1, nonlinear_sum_a1


def get_A_CG(K, KX, KY, k_index_map, kd, beta, kappa, nu, U, psi1_hat, h_hat):
    '''
    1. The mode truncation should be done outside this function. 
    2. K is the original number of modes in each x and y direction. 
    3. psi_hat, hk should be of shape (N,k_left) for proper broadcasting.
    '''
    N, k_left = psi1_hat.shape
    
    # Precompute constants
    K_squared = KX**2 + KY**2
    K_squared_kd2 = K_squared + kd**2 / 2
    K_squared2 = K_squared**2
    K_squared4 = K_squared**4
    invCk = K_squared * (K_squared + kd**2)
    dX = 1j * KX
    list_dic_k_map = list(k_index_map.keys())

    # linear part for A0, a0, A1 and a1
    linear_A0 = dX * ((K_squared_kd2 * beta - K_squared2 * U) * psi1_hat - kd**2/2 * U * h_hat) - nu * K_squared4 * (invCk * psi1_hat - kd**2/2 * h_hat)
    linear_a0 = dX * ((kd**2/2 * beta - kd**2 * K_squared * U) * psi1_hat - K_squared_kd2 * U * h_hat) + nu * K_squared4 * K_squared_kd2 * h_hat
    linear_A1_diag = dX * (kd**2/2 * beta + kd**2 * K_squared * U) - (kd**2/2 * kappa * K_squared)
    linear_a1_diag = dX * (K_squared_kd2 * beta + K_squared2 * U) - K_squared_kd2 * kappa * K_squared - nu * K_squared4 * invCk
    linear_A1 = np.tile(np.diag(linear_A1_diag)[None,:,:], (N,1,1)) 
    linear_a1 = np.tile(np.diag(linear_a1_diag)[None,:,:], (N,1,1)) 

    nonlinear_sum_A0, nonlinear_sum_a0, nonlinear_sum_A1, nonlinear_sum_a1 = get_A_CG_nonlinear(K, psi1_hat, h_hat, linear_A1, linear_a1, k_index_map, list_dic_k_map, kd)

    # aggregate 
    A0 = linear_A0 + nonlinear_sum_A0
    a0 = linear_a0 + nonlinear_sum_a0
    A1 = linear_A1 + nonlinear_sum_A1
    a1 = linear_a1 + nonlinear_sum_a1
    
    # normalization 
    Ck = 1 / invCk
    Ck[K_squared == 0] = 0  # avoid division by zero at k = 0, constant mode
    Ck_ = np.tile(Ck[None,:,None], (N,1,k_left))
    A0 = Ck * A0
    a0 = Ck * a0
    A1 = Ck_ * A1
    a1 = Ck_ * a1
    
    return A0, a0, A1, a1


def forward_OU(N, N_chunk, K, dt, x, y, r1, r2, mu0, a0, a1, R0, InvBoB, Sigma_u, mu_t, R_t, KX, KY, corr_noise):
    if corr_noise == False:
        # leverage the diagonal matrix property for acceleration
        a1_diag = a1.diagonal()
        Sigma_u_diag = Sigma_u.diagonal()
        Sigma_u_diag2 = Sigma_u_diag * np.conj(Sigma_u_diag)

        for i in range(1, N):
            i_chunk = (i-1) % N_chunk
            if i_chunk == 0:
                A1_t = get_A_OU(x[i-1:i-1+N_chunk, :], y[i-1:i-1+N_chunk, :], K, r1, r2, KX, KY)
            
            x0 = x[i - 1, :]
            y0 = y[i - 1, :]
            x1 = x[i, :]
            y1 = y[i, :]
            x_diff = np.mod(x1 - x0 + np.pi, 2 * np.pi) - np.pi # consider periodic boundary conditions
            y_diff = np.mod(y1 - y0 + np.pi, 2 * np.pi) - np.pi # consider periodic boundary conditions
    
            # precompute
            A1 = A1_t[i_chunk, :, :]
            R0_A1_H = R0 @ A1.conj().T
    
            # Update the posterior mean and posterior covariance
            mu = mu0 + (a0 + a1 @ mu0) * dt + R0_A1_H * InvBoB @ (np.hstack((x_diff, y_diff)) - A1 @ mu0 * dt)
            R = R0 + (a1_diag[:,None] * R0 + R0 * a1_diag.conj() + np.diag(Sigma_u_diag2) - R0_A1_H * InvBoB @ R0_A1_H.conj().T) * dt
            mu_t[:, i] = mu
            R_t[:, i] = np.diag(R)
            mu0 = mu
            R0 = R
            
    elif corr_noise == True:
        a1_diag = a1.diagonal()
        Sigma_u_sq = Sigma_u @ Sigma_u.conj().T
    
        for i in range(1, N):
            i_chunk = (i-1) % N_chunk
            if i_chunk == 0:
                A1_t = get_A_OU(x[i-1:i-1+N_chunk, :], y[i-1:i-1+N_chunk, :], K, r1, r2, KX, KY)
            
            x0 = x[i - 1, :]
            y0 = y[i - 1, :]
            x1 = x[i, :]
            y1 = y[i, :]
            x_diff = np.mod(x1 - x0 + np.pi, 2 * np.pi) - np.pi # consider periodic boundary conditions
            y_diff = np.mod(y1 - y0 + np.pi, 2 * np.pi) - np.pi # consider periodic boundary conditions
    
            # precompute
            A1 = A1_t[i_chunk, :, :]
            R0_A1_H = R0 @ A1.conj().T
    
            # Update the posterior mean and posterior covariance
            mu = mu0 + (a0 + a1 @ mu0) * dt + R0_A1_H * InvBoB @ (np.hstack((x_diff, y_diff)) - A1 @ mu0 * dt)
            R = R0 + (a1_diag[:,None] * R0 + R0 * a1_diag.conj() + Sigma_u_sq - R0_A1_H * InvBoB @ R0_A1_H.conj().T) * dt
            mu_t[:, i] = mu
            R_t[:, i] = np.diag(R)
            mu0 = mu
            R0 = R

    return mu_t, R_t


def forward_OU_flow(N, N_chunk, dt, psi1_k_t, mu0, R0, mu_t, R_t, A0_, A1, a0_, a1, F1, F2, Sigma1, Sigma2, sigma1, sigma2):
    boB = np.diag(sigma1 * Sigma1.conj() + sigma2 * Sigma2.conj())
    InvBoB = 1 / (Sigma1 * Sigma1.conj() + Sigma2 * Sigma2.conj())
    InvBoB[0] = 0
    bob = np.diag(sigma1 * sigma1.conj() + sigma2 * sigma2.conj())
    R0_A1_H = R0 * A1.conj()
    R0_a1_H = R0 * a1.conj()
    for i in range(1, N):
        i_chunk = (i-1) % N_chunk
        if i_chunk == 0:
            A0_t, a0_t = get_A_OU_flow(psi1_k_t[i-1:i-1+N_chunk, :], A0_, a0_, F1, F2)

        # precompute
        A0 = A0_t[i_chunk, :]
        a0 = a0_t[i_chunk, :]
        psi1_diff = psi1_k_t[i, :] - psi1_k_t[i-1, :]

        # Update the posterior mean and posterior covariance
        mu = mu0 + (a0 + a1 * mu0) * dt + (boB + R0_A1_H) * InvBoB @ (psi1_diff - (A0 + A1 * mu0) * dt)
        R = R0 + (R0_a1_H.conj().T + R0_a1_H + bob - (boB + R0_A1_H) * InvBoB @ (boB + R0_A1_H).conj().T) * dt

        mu_t[:, i] = mu
        R_t[:, i] = np.diag(R)
        mu0 = mu
        R0 = R

    return mu_t, R_t


def forward_CG(N, N_chunk, dt, K, KX_cut, KY_cut, k_index_map_cut, mu0, R0, InvBoB, sigma_2, mu_t, R_t, kd, beta, kappa, nu, U, psi1_k_t_cut_T, h_k_cut):
    for i in range(1, N):
        i_chunk = (i-1) % N_chunk
        if i_chunk == 0:
            A0_t, a0_t, A1_t, a1_t= get_A_CG(K, KX_cut, KY_cut, k_index_map_cut, kd, beta, kappa, nu, U, psi1_k_t_cut_T[i-1:i-1+N_chunk,:], h_k_cut)
        
        A0 = A0_t[i_chunk, :]
        a0 = a0_t[i_chunk, :]
        A1 = A1_t[i_chunk, :, :]
        a1 = a1_t[i_chunk, :, :]

        # precompute
        a1R0 = a1 @ R0
        sigma_2_sq = sigma_2 * np.conjugate(sigma_2)
        R0A1_H = R0 @ A1.conj().T
        psi1_diff = psi1_k_t_cut_T[i, :] - psi1_k_t_cut_T[i-1, :]
        
        # Update the posterior mean and posterior covariance
        mu = mu0 + (a0 + a1 @ mu0) * dt + (R0 @ A1.conj().T) * InvBoB @ (psi1_diff - (A0 + A1 @ mu0) * dt)
        R = R0 + (a1R0 + a1R0.conj().T + np.diag(sigma_2_sq) - (R0A1_H) * InvBoB @ R0A1_H.conj().T) * dt
        mu_t[:, i] = mu
        R_t[:, i] = np.diag(R)
        mu0 = mu
        R0 = R

    return mu_t, R_t


def mu2psi(mu_t, K, r_cut, style):
    '''
    reshape flattened variables to two modes matrices
    '''
    mu_t_ = mu_t.reshape((mu_t.shape[0] // 2, 2, -1), order='F')
    psi_k = inv_truncate(mu_t_[:,0,:], r_cut, K, style)
    tau_k = inv_truncate(mu_t_[:,1,:], r_cut, K, style)
        
    return psi_k, tau_k


def mu2layer(mu_t, K, r_cut, r1, r2, style):
    '''
    transform eigenmodes to two-layer modes (flattened matrices)
    '''
    mu_t_ = mu_t.reshape((mu_t.shape[0] // 2, 2, -1), order='F')
    psi_k_ = mu_t_[:,0,:]
    tau_k_ = mu_t_[:,1,:]
    psi1_k_ = np.zeros_like(psi_k_)
    psi2_k_ = np.zeros_like(psi_k_)
    r1 = truncate(r1, r_cut, style)
    r2 = truncate(r2, r_cut, style)
    for ik in range(psi_k_.shape[0]):
        eigenmat = np.array([r1[ik, :],r2[ik, :]]).T
        layer = eigenmat @ np.array([psi_k_[ik, :],tau_k_[ik, :]])
        psi1_k_[ik, :] = layer[0, :]
        psi2_k_[ik, :] = layer[1, :]

    return psi1_k_, psi2_k_
    

def back_sampling(N_s, dt, mu_t, R_t, Sigma, a0, a1):
    K_, N = mu_t.shape
    u_t = np.zeros((N_s, K_, N), dtype='complex') 
    
    # skip mode (0,0)
    mask = np.ones(K_, dtype=bool) 
    mask[0] = False
    mask[K_//2] = False
    mask2d = np.ones((K_, K_), dtype=bool)
    mask2d[0, :] = False 
    mask2d[:, 0] = False
    mask2d[K_//2, :] = False 
    mask2d[:, K_//2] = False
    K_mask = np.sum(mask)
    a0 = a0[mask] 
    R_t = R_t[mask, :]
    mu_t = mu_t[mask, :]
    a1 = a1[mask2d].reshape(K_-2, K_-2)
    Sigma = Sigma[mask2d].reshape(K_-2, K_-2)
    Sigma_sq = Sigma @ Sigma.conj().T

    for n_s in range(N_s):
        mu0 = mu_t[:, -1]
        u_t[n_s, mask, -1] = mu0
        R0 = np.diag(R_t[:, -1])
        u_t[n_s, mask, -1] = np.random.multivariate_normal(mu0.real, R0.real / 2) + 1j * np.random.multivariate_normal(mu0.imag, R0.real / 2) # assume for now only variance of R_t is saved

    for n in range(N-2, -1, -1):
        R_inv = np.diag(1 / R_t[:, n])
        noise = np.diag(Sigma) / np.sqrt(2) * (np.random.randn(N_s, K_mask) + 1j*np.random.randn(N_s, K_mask)) * np.sqrt(dt)
        ## if correlated noise, need to add extra terms
        # noise = (Sigma / np.sqrt(2)) @ (noise_r + 1j*noise_i) * np.sqrt(dt)
        u_t[:, mask, n] = u_t[:, mask, n+1] + np.squeeze((-a0[:, None] - a1@u_t[:, mask, n+1][:, :, None]) * dt + Sigma_sq @ R_inv @ (mu_t[:, n][:, None] - u_t[:, mask, n+1][:, :, None])) * dt + noise
        
    return u_t


class Lagrangian_DA_OU:
    def __init__(self, K, r1, r2, f, gamma, omega, sigma, r_cut, style='circle', corr_noise=False, **kargs):
        """
        Parameters:
        - N: int, total number of steps
        - N_chunk: trunk for calculating DA coefficient matrix
        - K: number of Fourier modes along one axis
        - style: truncation style, 'circle' or 'square'
        - psi_k_t: np.array of shape (K, K, N), truth time series of the Fourier eigenmode1 stream function. only to provide DA initial condition
        - tau_k_t: np.array of shape (K, K, N), truth time series of the Fourier eigenmode2 stream function. only to provide DA initial condition
        - r1: eigenvectors1
        - r2: eigenvectors2
        - dt: float, time step
        - sigma_xy: float, standard deviation of the observation noise
        - f: forcing in complex OU process model
        - gamma: damping in complex OU process model
        - omega: phase in complex OU process model
        - sigma: noise standard deviation in complex OU process model
        - xt: observations x of shape (L, N)
        - yt: observations y of shape (L, N)
        - r_cut: modes truncation radius
        - corr_noise: LSMs with correlalted noise
        """

        self.K = K
        self.style = style
        self.r_cut = r_cut
        Kx = np.fft.fftfreq(K) * K
        Ky = np.fft.fftfreq(K) * K
        KX, KY = np.meshgrid(Kx, Ky)
        self.KX = truncate(KX, r_cut, style)
        self.KY = truncate(KY, r_cut, style)
        self.r1 = truncate(r1, r_cut, style)
        self.r2 = truncate(r2, r_cut, style)
        self.f = truncate(f,r_cut, style=style)
        self.gamma = truncate(gamma,r_cut, style=style)
        self.omega = truncate(omega,r_cut, style=style)
        self.sigma = truncate(sigma,r_cut, style=style)
        self.corr_noise = corr_noise
        
        if corr_noise:
            cov = kargs['cov']
            cov = truncate(cov, r_cut, style=style)
            d_cov = cov.shape[0]
            self.Sigma_u[:d_cov, d_cov:] = np.diag(cov)
            self.Sigma_u[d_cov:, :d_cov] = np.diag(cov)  
            self.Sigma_u = np.nan_to_num(self.Sigma_u) # some modes not evaluated are nan
            self.Sigma_u = self.Sigma_u * np.sqrt(2) # normalization term due to real vs complex white noise


    def forward(self, N, N_chunk, dt, tracer=True, **kargs):
        r_cut = self.r_cut
        style = self.style
        r1 = self.r1
        r2 = self.r2
        gamma = self.gamma
        omega = self.omega
        f = self.f
        sigma = self.sigma

        if tracer == True:
            psi_k_t = kargs['psi_k_t']
            tau_k_t = kargs['tau_k_t']
            sigma_xy = kargs['sigma_xy']
            xt = kargs['xt']
            yt = kargs['yt']

            InvBoB = 1 / sigma_xy**2
            mu0 = np.concatenate((truncate(psi_k_t[:,:,0],r_cut, style=style), truncate(tau_k_t[:,:,0],r_cut, style=style))) # assume the initial condition is truth
            K_ = mu0.shape[0]
            # self.R0 = np.eye(K_, dtype='complex')
            R0 = np.zeros((K_, K_), dtype='complex')
            mu_t = np.zeros((K_, N), dtype='complex')  # posterior mean
            mu_t[:, 0] = mu0
            R_t = np.zeros((K_, N), dtype='complex')  # posterior covariance
            R_t[:, 0] = np.diag(R0)  # only save the diagonal elements
            a0 = self.f.flatten(order='F')
            a1 = -np.diag(gamma.flatten(order='F')) + 1j * np.diag(omega.flatten(order='F'))
            Sigma_u = np.diag(self.sigma.flatten(order='F'))

            mu_t, R_t = forward_OU(N, N_chunk, self.K, dt, xt, yt, r1[:, 0], r2[:, 0], mu0, a0, a1, R0, InvBoB, Sigma_u, mu_t, R_t, self.KX, self.KY, self.corr_noise)

        elif tracer == False:
            psi1_k_t = kargs['psi1_k_t']
            psi2_k_t0 = kargs['psi2_k_t0']
            psi1_k_t = truncate(psi1_k_t, r_cut, style)

            mu0 = truncate(psi2_k_t0, r_cut, style) # assume the initial condition is truth
            K_ = mu0.shape[0]
            mu_t = np.zeros((K_, N), dtype='complex') # posterior mean
            mu_t[:, 0] = mu0
            R0 = np.zeros((K_, K_), dtype='complex')
            R_t = np.zeros((K_, N), dtype='complex')  # posterior covariance
            R_t[:, 0] = np.diag(R0) # only save the diagonal elements

            deno = 1 / r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0]
            deno[0] = 0
            A0_ = (r1[:, 0] * r2[:, 1] * (-gamma[:, 0] + 1j * omega[:, 0]) - r1[:, 1] * r2[:, 0] * (-gamma[:, 1] + 1j * omega[:, 1])) * deno
            A1 = (r1[:, 0] * r2[:, 0] * (-(-gamma[:, 0] + gamma[:, 1]) + 1j * (-omega[:, 0] + omega[:, 1]))) * deno
            a0_ = (r1[:, 1] * r2[:, 1] * (-(gamma[:, 0] - gamma[:, 1]) + 1j * (omega[:, 0] - omega[:, 1]))) * deno
            a1 = (-r1[:, 1] * r2[:, 0] * (-gamma[:, 0] + 1j * omega[:, 0]) + r1[:, 0] * r2[:, 1] * (-gamma[:, 1] + 1j * omega[:, 1])) * deno
            F1 = r1[:, 0] * f[:, 0] + r2[:, 0] * f[:, 1]
            F2 = r1[:, 1] * f[:, 0] + r2[:, 1] * f[:, 1]
            Sigma1 = r1[:, 0] * sigma[:, 0]
            Sigma2 = r2[:, 0] * sigma[:, 1]
            sigma1 = r1[:, 1] * sigma[:, 0]
            sigma2 = r2[:, 1] * sigma[:, 1]

            mu_t, R_t = forward_OU_flow(N, N_chunk, dt, psi1_k_t.T, mu0, R0, mu_t, R_t, A0_, A1, a0_, a1, F1, F2, Sigma2, Sigma2, sigma1, sigma2)

        return mu_t, R_t


class Lagrangian_DA_CG:
    def __init__(self, K, sigma_1, sigma_2, kd, beta, kappa, nu, U, h_hat, r_cut, style='circle'):
        """
        Parameters:
        - K: number of Fourier modes along one axis
        - style: truncation style, 'circle' or 'square'
        - psi1_k_t: np.array of shape (K, K, N), truth time series of the Fourier eigenmode1 stream function.
        - psi2_k_t: np.array of shape (K, K, N), only used for the DA initial condition, assumed to be truth 
        - sigma_xy: float, standard deviation of the observation noise
        - r_cut: modes truncation radius
        """

        self.K = K
        Kx = np.fft.fftfreq(K) * K
        Ky = np.fft.fftfreq(K) * K
        KX, KY = np.meshgrid(Kx, Ky)
        self.KX_cut = truncate(KX, r_cut, style)
        self.KY_cut = truncate(KY, r_cut, style)
        self.h_k_cut = truncate(h_hat, r_cut, style)
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.kd = kd
        self.beta = beta
        self.kappa = kappa
        self.nu = nu
        self.U = U
        self.style = style
        self.r_cut = r_cut

        # self.k_index_map_cut = {(KX[iy, ix], KY[iy, ix]): (ix, iy) for ix in range(K) for iy in range(K) if (KX[iy, ix]**2 + KY[iy, ix]**2) <=r_cut**2}

        # Create an empty Numba-typed dictionary with appropriate key and value types
        k_index_map_cut = Dict.empty(
            key_type=types.Tuple([types.float64, types.float64]),  
            value_type=types.Tuple([types.int64, types.int64])   
        )
        # Populate the Numba-typed dictionary using a loop
        for ix in range(K):
            for iy in range(K):
                kx = KX[iy, ix]
                ky = KY[iy, ix]
                if (kx**2 + ky**2) <= r_cut**2:
                    k_index_map_cut[(kx, ky)] = (ix, iy)
        self.k_index_map_cut = k_index_map_cut


    def forward(self, N, N_chunk, dt, N_s=1, **kargs):
        """
        - N: int, total number of steps
        - N_chunk: trunk for calculating DA coefficient matrix
        - dt: float, time step
        - N_s: number of observation trajectories
        """

        # initialize mean and variance
        psi2_k_t0 = kargs['psi2_k_t0']
        psi2_k_t0_cut = truncate(psi2_k_t0, self.r_cut, self.style)
        mu0 = psi2_k_t0_cut # the initial condition posterior mean
        K_ = psi2_k_t0_cut.shape[0] # number of flattened K modes
        # R0 = np.eye(K_, dtype='complex')
        R0 = np.zeros((K_,K_), dtype='complex')
        sigma_1 = self.sigma_1
        sigma_2 = self.sigma_2 * np.ones(K_)        
        InvBoB = 1 / sigma_1**2

        if N_s == 1:
            psi1_k_t = kargs['psi1_k_t']
            psi1_k_t_cut = truncate(psi1_k_t, self.r_cut, self.style)
            psi2_k_t0_cut = truncate(psi2_k_t0, self.r_cut, self.style)
            psi1_k_t_cut = np.transpose(psi1_k_t_cut, axes=(1,0)) # psi_hat should be of shape shape (N,k_left)
            mu_t = np.zeros((K_, N), dtype='complex')  # posterior mean
            mu_t[:, 0] = mu0
            R_t = np.zeros((K_, N), dtype='complex')  # posterior covariance
            R_t[:, 0] = np.diag(R0)  # only save the diagonal elements

            mu_t, R_t = forward_CG(N, N_chunk, dt, self.K, self.KX_cut, self.KY_cut, self.k_index_map_cut, mu0, R0, InvBoB, sigma_2, mu_t, R_t, self.kd, self.beta, self.kappa, self.nu, self.U, psi1_k_t_cut, self.h_k_cut)

            return mu_t, R_t

        elif N_s > 1:
            mu_eigen_t = kargs['mu_eigen_t'] # eigenmodes
            R_eigen_t = kargs['R_eigen_t']
            sigma = kargs['sigma']
            f = kargs['f']
            gamma = kargs['gamma']
            omega = kargs['omega']
            r1 = kargs['r1']
            r2 = kargs['r2']
            Sigma = np.diag(truncate(sigma,self.r_cut, style=self.style).flatten(order='F')) # assume independent noise for now
            a0 = truncate(f,self.r_cut, style=self.style).flatten(order='F')
            a1 = -np.diag(truncate(gamma,self.r_cut, style=self.style).flatten(order='F')) + 1j * np.diag(truncate(omega,self.r_cut, style=self.style).flatten(order='F')) 
            psitau_k_s = back_sampling(N_s, dt, mu_eigen_t, R_eigen_t, Sigma, a0, a1)
            mu_t = np.zeros((N_s, K_, N), dtype='complex')  # posterior mean
            mu_t[:, :, 0] = mu0
            R_t = np.zeros((N_s, K_, N), dtype='complex')  # posterior covariance
            R_t[:, :, 0] = np.diag(R0)  # only save the diagonal elements

            for n_s in range(N_s):
                print(n_s)
                psi1_k_s, _ = mu2layer(psitau_k_s[n_s, :, :], self.K, self.r_cut, r1, r2, self.style)
                psi1_k_s = np.transpose(psi1_k_s, axes=(1,0)) # psi_hat should be of shape shape (N,k_left)

                mu_t[n_s, :, :], R_t[n_s, :, :] = forward_CG(N, N_chunk, dt, self.K, self.KX_cut, self.KY_cut, self.k_index_map_cut, mu0, R0, InvBoB, sigma_2, mu_t[n_s, :, :], R_t[n_s, :, :], self.kd, self.beta, self.kappa, self.nu, self.U, psi1_k_s, self.h_k_cut)


            return mu_t, R_t

    def forward_model(self, N, dt, psi1_k_t0, psi2_k_t0):
        K = self.K
        r_cut = self.r_cut
        style = self.style
        KX_cut = self.KX_cut
        KY_cut = self.KY_cut
        k_index_map_cut = self.k_index_map_cut
        kd = self.kd
        beta = self.beta
        kappa = self.kappa
        nu = self.nu
        U = self.U
        h_k_cut = self.h_k_cut
        psi1_k_t0 = truncate(psi1_k_t0, r_cut, style)
        psi2_k_t0 = truncate(psi2_k_t0, r_cut, style)
        K_ = psi1_k_t0.shape[0]

        psi1_k_t = np.zeros((K_, N), dtype='complex')
        psi2_k_t = np.zeros((K_, N), dtype='complex')
        psi1_k_t[:, 0] = psi1_k_t0
        psi2_k_t[:, 0] = psi2_k_t0

        for i in range(1, N):
            A0, a0, A1, a1= get_A_CG(K, KX_cut, KY_cut, k_index_map_cut, kd, beta, kappa, nu, U, psi1_k_t[:, i-1][None, :], h_k_cut)
            
            A0 = np.squeeze(A0)
            a0 = np.squeeze(a0)
            A1 = np.squeeze(A1)
            a1 = np.squeeze(a1)

            # forward integration
            psi1_k_t[:, i] = psi1_k_t[:, i-1] + (A0 + A1 @ psi2_k_t[:, i-1]) * dt
            psi2_k_t[:, i] = psi2_k_t[:, i-1] + (a0 + a1 @ psi2_k_t[:, i-1]) * dt

        return psi1_k_t, psi2_k_t

                