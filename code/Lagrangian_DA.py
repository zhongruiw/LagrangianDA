import numpy as np
from numba import jit
from scipy import sparse


''' 
Lagrangian DA 
    1. using tracers to recover the two-layer flow field 
    2. the reference (true) flow field is from QG model
    3. the reduced-order model is complex OU process / nonlinear conditional Gaussian model
'''

# @jit(nopython=True)
def truncate(kk, cut):
    '''require the input kk has shape (K,) or (K,K,...)'''
    K = kk.shape[0]

    if kk.ndim == 1:
        index_to_remove = np.zeros(K, dtype=np.bool_)
        index_to_remove[(K//2-cut):(K//2+cut+1)] = True
    elif kk.ndim > 1:
        index_to_remove = np.zeros(kk.shape, dtype=np.bool_)
        index_to_remove[(K//2-cut):(K//2+cut+1), :] = True
        index_to_remove[:, (K//2-cut):(K//2+cut+1)] = True
        new_shape = np.array(kk.shape)
        new_shape[:2] -= (2*cut+1)

    return kk[~index_to_remove].reshape(new_shape)

def inv_truncate(kk_truncated, cut):
    ''' Recovers the original array from a truncated version by filling zeros.
        Parameters:
        - kk_truncated: The truncated array.
        - cut: The number of columns and rows around the center that were removed.
    '''
    K = kk_truncated.shape[0] + 2*cut+1

    if len(kk_truncated.shape) == 1:
        recovered = np.zeros(K, dtype=kk_truncated.dtype)
        recovered[:K//2-cut] = kk_truncated[:K//2-cut]
        recovered[K//2+cut+1:] = kk_truncated[K//2-cut:]

    elif len(kk_truncated.shape) > 1:
        new_shape = np.array(kk_truncated.shape)
        new_shape[:2] += (2*cut+1)
        recovered = np.zeros(new_shape, dtype=kk_truncated.dtype)
        recovered[:K//2-cut, :K//2-cut] = kk_truncated[:K//2-cut, :K//2-cut]
        recovered[K//2+cut+1:, :K//2-cut] = kk_truncated[K//2-cut:, :K//2-cut]
        recovered[:K//2-cut, K//2+cut+1:] = kk_truncated[:K//2-cut, K//2-cut:]
        recovered[K//2+cut+1:, K//2+cut+1:] = kk_truncated[K//2-cut:, K//2-cut:]

    return recovered

# @jit(nopython=True)
def get_A_OU(x,y,K,r1,r2,cut,KX,KY):
    L = x.shape[0]
    E = np.zeros((2*L, (K-2*cut-1)**2), dtype=np.complex_)
    KX_flat = KX.flatten(order='F')
    KY_flat = KY.flatten(order='F')

    exp_term = np.exp(1j * (x[:,None] @ KX_flat[None,:] + y[:,None] @ KY_flat[None,:]))
    E[:L,:] = exp_term * (1j) * KY_flat
    E[L:,:] = exp_term * (-1j) * KX_flat
    # R_bot = sparse.hstack([sparse.diags(r1.flatten(order='F')), sparse.diags(r2.flatten(order='F'))], format="csr")
    R_bot = np.hstack((np.diag(r1.flatten(order='F')), np.diag(r2.flatten(order='F'))))
    A = E @ R_bot / K**2

    return A

# @jit(nopython=True)
def forward_(N, K, dt, x, y, r1, r2, cut, mu0, a0, a1, R0, InvBoB, Sigma_u, mu_t, R_t, KX, KY):
    # leverage the diagonal matrix property for acceleration
    a1_diag = a1.diagonal()
    Sigma_u_diag = Sigma_u.diagonal()

    for i in range(1, N):
        x0 = x[:, i - 1]
        y0 = y[:, i - 1]
        x1 = x[:, i]
        y1 = y[:, i]
        x_diff = np.mod(x1 - x0 + np.pi, 2 * np.pi) - np.pi # consider periodic boundary conditions
        y_diff = np.mod(y1 - y0 + np.pi, 2 * np.pi) - np.pi # consider periodic boundary conditions

        # Matrix for filtering
        A1 = get_A_OU(x0, y0, K, r1, r2, cut, KX, KY)
        R0_A1_H = R0 @ A1.conj().T

        # Update the posterior mean and posterior covariance
        mu = mu0 + (a0 + a1 @ mu0) * dt + R0_A1_H * InvBoB @ (np.hstack((x_diff, y_diff)) - A1 @ mu0 * dt)
        R = R0 + (a1_diag[:,None] * R0 + R0 * a1_diag.conj() + np.diag(Sigma_u_diag**2) - R0_A1_H * InvBoB @ R0_A1_H.conj().T) * dt
        mu_t[:, i] = mu
        R_t[:, i] = np.diag(R)
        mu0 = mu
        R0 = R

    return mu_t, R_t

def mu2psi(mu_t, K, cut):
    '''reshape flattened variables to two modes matrices'''
    n = K-cut*2-1
    mu_t = mu_t.reshape((n,n,2,-1), order='F')
    psi_k = mu_t[:,:,0]
    tau_k = mu_t[:,:,1]
    
    return psi_k, tau_k


def get_A_CG(KX, KY, K_squared, kd, beta, kappa, U, psi1_hat, psi2_hat, h_hat):
    K = KX.shape[0]
    
    # psi_hat, hk should be of shape (N, K, K), if not, reshape them to make it legible
    if psi1_hat.shape[-2:] != KX.shape:
        psi1_hat = np.transpose(psi1_hat, axes=(2,0,1))
    if psi2_hat.shape[-2:] != KX.shape:
        psi2_hat = np.transpose(psi2_hat, axes=(2,0,1))
    if h_hat.shape[-2:] != KX.shape:
        h_hat = np.transpose(h_hat, axes=(2,0,1))

    # make sure \partial psi1_k / \partial x be conjugate symmetric
    psi1_hat[:, :, K//2] = 0

    # define Ck for each wave number
    Ck = 1 / (K_squared * (K_squared + kd**2))
    Ck[K_squared == 0] = 0  # avoid division by zero at k = 0, constant mode

    # linear part for A0, a0, A1 and a1
    linear_A0 = 1j * KX * (((K_squared + kd**2/2)*beta - K_squared**2 * U)*psi1_hat - kd**2/2 * U * h_hat)
    linear_a0 = 1j * KX * ((kd**2/2 * beta - kd**2 * K_squared * U)*psi1_hat - (K_squared + kd**2/2) * U * h_hat)
    linear_A1_diag = 1j * KX * (kd**2/2 * beta + kd**2 * K_squared * U) - (kd**2/2 * kappa * K_squared)
    linear_a1_diag = 1j * KX * ((K_squared + kd**2/2)*beta + K_squared**2 * U) - (K_squared + kd**2/2) * kappa * K_squared
    linear_A1 = np.tile(np.diag(linear_A1_diag.flatten(order='F'))[None,:,:], (psi1_hat.shape[0],1,1)) 
    linear_a1 = np.tile(np.diag(linear_a1_diag.flatten(order='F'))[None,:,:], (psi1_hat.shape[0],1,1)) 
    
    # nonlinear summation part for A0, a0, A1 and a1
    nonlinear_sum_A0 = np.zeros_like(psi1_hat, dtype=complex)
    nonlinear_sum_a0 = np.zeros_like(psi1_hat, dtype=complex)
    nonlinear_sum_A1 = np.zeros_like(linear_A1, dtype=complex)
    nonlinear_sum_a1 = np.zeros_like(linear_a1, dtype=complex)
    k_index_map = {(KX[iy, ix], KY[iy, ix]): (ix, iy) for ix in range(K) for iy in range(K)}
    
    # check 
    det_sum = np.zeros(KX.shape)
    Jacobi_2_diff = np.zeros_like(psi1_hat, dtype=complex)
    
    for imx in range(K):
        for imy in range(K):
            m = [KX[imy, imx], KY[imy, imx]]
            psi1_m = psi1_hat[:, imy, imx]
            psi2_m = psi2_hat[:, imy, imx]

            for inx in range(K):
                for iny in range(K):
                    n = [KX[iny, inx], KY[iny, inx]]
                    psi1_n = psi1_hat[:, iny, inx]
                    psi2_n = psi2_hat[:, iny, inx]
                    h_n = h_hat[iny, inx]
                    
                    # Check if k is within the wave number sets
                    k = (m[0] + n[0], m[1] + n[1])
                    
                    if k in k_index_map:
                        ikx, iky = k_index_map[k]
                        ik_flatten = ikx * K + iky
                        im_flatten = imx * K + imy
                        k_mod = K_squared[iky, ikx]
                        det_mn = np.linalg.det(np.array([m, n]))
                        # nonlinear_sum_A0[:, iky, ikx] -= det_mn * ((k_mod + kd**2 / 2)**2 * psi1_n * psi1_m)
                        # nonlinear_sum_a0[:, iky, ikx] -= det_mn * (kd**2 / 2 * (k_mod + kd**2 / 2) * psi1_n * psi1_m)
                        nonlinear_sum_A1[:, ik_flatten, im_flatten] += det_mn * kd**2 / 2 * (k_mod * psi1_n - h_n)
                        nonlinear_sum_a1[:, ik_flatten, im_flatten] -= det_mn * (k_mod * kd**2 / 2 * psi1_n - (k_mod + kd**2 / 2) * h_n)    
    
                        det_sum[iky,ikx] += det_mn
                        Jacobi_2_diff[:,iky,ikx] -= det_mn *  ((k_mod + kd**2 / 2)**2 * psi2_n * psi2_m)
    # aggregate 
    A0 = linear_A0 + nonlinear_sum_A0
    a0 = linear_a0 + nonlinear_sum_a0
    A1 = linear_A1 + nonlinear_sum_A1
    a1 = linear_a1 + nonlinear_sum_a1
    
    # normalization 
    A0 = Ck * A0
    a0 = Ck * a0
    A1 = np.tile(Ck.flatten(order='F')[:,None], (1,K**2)) * A1
    a1 = np.tile(Ck.flatten(order='F')[:,None], (1,K**2)) * a1
    
    # flatten
    A0 = A0.reshape((A0.shape[0], -1), order='F')
    a0 = a0.reshape((a0.shape[0], -1), order='F')
    
    return A0, a0, A1, a1, np.tile(Ck.flatten(order='F')[:,None], (1,K**2)) * nonlinear_sum_A1, det_sum, Jacobi_2_diff


class Lagrangian_DA_OU:
    def __init__(self, N, K, psi_t_k, tau_t_k, r1, r2, dt, sigma_xy, f, gamma, omega, sigma, xt, yt, cut=0):
        """
        Parameters:
        - N: int, total number of steps
        - K: number of Fourier modes along one axis
        - psi_t_k: np.array of shape (K, K, N), truth time series of the Fourier eigenmode1 stream function.
        - tau_t_k: np.array of shape (K, K, N), truth time series of the Fourier eigenmode2 stream function.
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
        - cut: modes truncation, number of truncated modes = 2*cut+1 from K//2 centerwise
        """
        self.N = N
        self.K = K
        self.r1 = r1
        self.r2 = r2
        self.dt = dt
        self.InvBoB = 1 / sigma_xy**2
        self.mu0 = np.concatenate((truncate(psi_t_k[:,:,0],cut).flatten(order='F'), truncate(tau_t_k[:,:,0],cut).flatten(order='F'))) # assume the initial condition is truth
        self.n = self.mu0.shape[0]
        self.R0 = np.zeros((self.n, self.n), dtype='complex')
        self.mu_t = np.zeros((self.n, N), dtype='complex')  # posterior mean
        self.mu_t[:, 0] = self.mu0
        self.R_t = np.zeros((self.n, N), dtype='complex')  # posterior covariance
        self.R_t[:, 0] = np.diag(self.R0)  # only save the diagonal elements
        self.a0 = truncate(f,cut).flatten(order='F')
        self.a1 = -np.diag(truncate(gamma,cut).flatten(order='F')) + 1j * np.diag(truncate(omega,cut).flatten(order='F'))
        self.Sigma_u = np.diag(truncate(sigma,cut).flatten(order='F'))
        self.x = xt
        self.y = yt
        self.cut = cut

    def forward(self):
        kx = np.fft.fftfreq(self.K) * self.K
        ky = np.fft.fftfreq(self.K) * self.K
        KX, KY = np.meshgrid(kx, ky)
        KX = truncate(KX, self.cut)
        KY = truncate(KY, self.cut)
        self.r1 = truncate(self.r1[:,:,0], self.cut)
        self.r2 = truncate(self.r2[:,:,0], self.cut)

        mu_t, R_t = forward_(self.N, self.K, self.dt, self.x, self.y, self.r1, self.r2, self.cut, self.mu0, self.a0, self.a1, self.R0, self.InvBoB, self.Sigma_u, self.mu_t, self.R_t, KX, KY)

        return mu_t, R_t


class Lagrangian_DA_CG:
    def __init__(self, N, K, psi_t_k, tau_t_k, r1, r2, dt, sigma_xy, f, gamma, omega, sigma, xt, yt, cut=0):
        """
        Parameters:
        - N: int, total number of steps
        - K: number of Fourier modes along one axis
        - psi_t_k: np.array of shape (K, K, N), truth time series of the Fourier eigenmode1 stream function.
        - tau_t_k: np.array of shape (K, K, N), truth time series of the Fourier eigenmode2 stream function.
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
        - cut: modes truncation, number of truncated modes = 2*cut+1 from K//2 centerwise
        """
        self.N = N
        self.K = K
        self.r1 = r1
        self.r2 = r2
        self.dt = dt
        self.InvBoB = 1 / sigma_xy**2
        self.mu0 = np.concatenate((truncate(psi_t_k[:,:,0],cut).flatten(order='F'), truncate(tau_t_k[:,:,0],cut).flatten(order='F'))) # assume the initial condition is truth
        self.n = self.mu0.shape[0]
        self.R0 = np.zeros((self.n, self.n), dtype='complex')
        self.mu_t = np.zeros((self.n, N), dtype='complex')  # posterior mean
        self.mu_t[:, 0] = self.mu0
        self.R_t = np.zeros((self.n, N), dtype='complex')  # posterior covariance
        self.R_t[:, 0] = np.diag(self.R0)  # only save the diagonal elements
        self.a0 = truncate(f,cut).flatten(order='F')
        self.a1 = -np.diag(truncate(gamma,cut).flatten(order='F')) + 1j * np.diag(truncate(omega,cut).flatten(order='F'))
        self.Sigma_u = np.diag(truncate(sigma,cut).flatten(order='F'))
        self.x = xt
        self.y = yt
        self.cut = cut

    def forward(self):
        kx = np.fft.fftfreq(self.K) * self.K
        ky = np.fft.fftfreq(self.K) * self.K
        KX, KY = np.meshgrid(kx, ky)
        KX = truncate(KX, self.cut)
        KY = truncate(KY, self.cut)
        self.r1 = truncate(self.r1[:,:,0], self.cut)
        self.r2 = truncate(self.r2[:,:,0], self.cut)

        mu_t, R_t = forward_(self.N, self.K, self.dt, self.x, self.y, self.r1, self.r2, self.cut, self.mu0, self.a0, self.a1, self.R0, self.InvBoB, self.Sigma_u, self.mu_t, self.R_t, KX, KY)

        return mu_t, R_t