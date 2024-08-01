import numpy as np
from numba import jit
from mode_truc import truncate, inv_truncate


class Lagrange_tracer_model:
    """
    math of the model:
    v(x, t) =\sum_k{psi_hat_{1,k}(t) e^(ik·x) r_k}
    r_k=(ik_2,-ik_1). 

    """
    def __init__(self, N, L, K, psi_hat, dt, sigma_xy, x0, y0, interv=1, t_interv=100, style='square'):
        """
        Parameters:
        - N: int, total number of steps
        - L: int, number of tracers
        - K: number of modes
        - kx: np.array of shape (K,), wave numbers of x direction. K is the number of 1D Fourier modes.
        - ky: np.array of shape (K,), wave numbers of y direction. K is the number of 1D Fourier modes.
        - psi_hat: np.array of shape (N, K, K) or truncated, Fourier time series of the upper layer stream function.
        - dt: float, time step
        - sigma_xy: float, standard deviation of the noise
        - x0:  Initial tracer locations in x of shape (L, 1)
        - y0:  Initial tracer locations in y of shape (L, 1)
        - interv:  int, wave number inverval for calculating u, v field 
        - t_interv: int, time interval for calculate and save u, v field
        - style: truncation style
        """
        self.N = N
        self.L = L
        self.K = K
        self.psi_hat = psi_hat
        self.dt = dt
        self.sigma_xy = sigma_xy
        self.x = np.zeros((L, N))  
        self.y = np.zeros((L, N))
        self.xgrid = np.linspace(-np.pi,np.pi, self.K, endpoint=False)
        self.ygrid = np.linspace(-np.pi,np.pi, self.K, endpoint=False)
        self.ut = np.zeros((self.K//interv, self.K//interv, N//t_interv))  
        self.vt = np.zeros((self.K//interv, self.K//interv, N//t_interv))
        self.x[:,0] = x0
        self.y[:,0] = y0
        self.interv = interv
        self.t_interv = t_interv
        self.style = style

    # @jit(nopython=True)
    def forward(self):
        """
        Integrates tracer locations using forward Euler method.
        
        There are to ways to get u,v from psi_hat.
        1) write the modified Fourier coefficients as psi_hat_{1,k} r_k, do ifft2. Then interpolate velocity on the grid points to (x,y).
        2) use (x,y) when computing the exponential components, then manually sum up.
        
        """
        Kx = np.fft.fftfreq(self.K) * self.K
        Ky = np.fft.fftfreq(self.K) * self.K
        KX, KY = np.meshgrid(Kx, Ky)

        if self.psi_hat.shape[0] < self.K and self.style == 'square':
            r_cut = (self.psi_hat.shape[0] - 1) // 2
            KX_flat = truncate(KX, r_cut, style=self.style)
            KY_flat = truncate(KY, r_cut, style=self.style)
            psi_hat_flat = np.reshape(self.psi_hat, (self.psi_hat.shape[0]**2, -1), order='F')
        elif self.psi_hat.shape[0] == self.K:
            KX_flat = KX.flatten(order='F')
            KY_flat = KY.flatten(order='F')
            psi_hat_flat = np.reshape(self.psi_hat, (self.K**2, -1), order='F')
        else:
            raise Exception("unknown truncation style")

        l = 0
        for i in range(1, self.N):
            exp_term = np.exp(1j * self.x[:, i-1][:,None] @ KX_flat[None,:] + 1j * self.y[:, i-1][:,None] @ KY_flat[None,:])
            uk = (psi_hat_flat[:, i-1] * (1j) * KY_flat)
            vk = (psi_hat_flat[:, i-1] * (-1j) * KX_flat)

            # ensure conjugate symmetric
            if self.style == 'square' and self.psi_hat.shape[0] == self.K:
                uk[self.K//2::self.K] = 0; uk[self.K*self.K//2:self.K*(self.K//2 + 1)] = 0
                vk[self.K//2::self.K] = 0; vk[self.K*self.K//2:self.K*(self.K//2 + 1)] = 0 

            u = np.squeeze(exp_term @ uk[:,None]) / self.K**2
            v = np.squeeze(exp_term @ vk[:,None]) / self.K**2
            
            max_imag_abs = max(np.max(np.abs(np.imag(u))), np.max(np.abs(np.imag(v))))
            if max_imag_abs > 1e-10:
                raise Exception("get significant imaginary parts, check the ifft2")
            else:
                u = np.real(u)
                v = np.real(v)

            self.x[:, i] = self.x[:, i-1] + u * self.dt + np.random.randn(self.L) * self.sigma_xy * np.sqrt(self.dt)
            self.y[:, i] = self.y[:, i-1] + v * self.dt + np.random.randn(self.L) * self.sigma_xy * np.sqrt(self.dt)
            self.x[:, i] = np.mod(self.x[:, i] + np.pi, 2*np.pi) - np.pi  # Periodic boundary conditions
            self.y[:, i] = np.mod(self.y[:, i] + np.pi, 2*np.pi) - np.pi  # Periodic boundary conditions

            if np.mod(i,self.t_interv) == 0:
#                 # manually do ifft2
#                 for jx in range(0,self.K,self.interv):
#                     for jy in range(0,self.K,self.interv):
#                         exp_term = np.exp(1j * self.xgrid[jx] * self.kx.flatten()[None,:] + 1j * self.ygrid[jy] * self.ky.flatten()[None,:])
#                         uk = (self.psi_hat[:, :, i-1] * (1j) * self.ky)
#                         vk = (self.psi_hat[:, :, i-1] * (-1j) * self.kx)
#                         uk[self.K//2, :] = 0; uk[:, self.K//2] = 0; vk[self.K//2, :] = 0; vk[:, self.K//2] = 0 # ensure conjugate symmetric
#                         u = np.squeeze(exp_term @ uk.flatten()[:,None]) / self.K**2
#                         v = np.squeeze(exp_term @ vk.flatten()[:,None]) / self.K**2
#                         max_imag_abs = max(np.max(np.abs(np.imag(u))), np.max(np.abs(np.imag(v))))
#                         if max_imag_abs > 1e-10:
#                             raise Exception("get significant imaginary parts, check the ifft2")
#                         else:
#                             u = np.real(u)
#                             v = np.real(v)

#                         self.ut[jy//self.interv,jx//self.interv,l] = u
#                         self.vt[jy//self.interv,jx//self.interv,l] = v
                if self.psi_hat.shape[0] == self.K:
                    psi_hat_KK = self.psi_hat[:, :, i-1]
                else:
                    psi_hat_KK = inv_truncate(psi_hat_flat[:, i-1][:,None], r_cut, self.K, self.style)[:,:,0]

                # using built-in ifft2
                u_ifft = np.fft.ifft2(psi_hat_KK * 1j * KY)
                u_ifft_shift = np.roll(u_ifft, shift=self.K//2, axis=0) # shift domain from [0,2pi) to [-pi,pi)
                u_ifft_shift = np.roll(u_ifft_shift, shift=self.K//2, axis=1) # shift domain from [0,2pi) to [-pi,pi)
                self.ut[:,:,l] = u_ifft_shift[::self.interv, ::self.interv] # only save the sparsely sampled grids
                v_ifft = np.fft.ifft2(psi_hat_KK * (-1j) * KX)
                v_ifft_shift = np.roll(v_ifft, shift=self.K//2, axis=0) # shift domain from [0,2pi) to [-pi,pi)
                v_ifft_shift = np.roll(v_ifft_shift, shift=self.K//2, axis=1) # shift domain from [0,2pi) to [-pi,pi)
                self.vt[:,:,l] = v_ifft_shift[::self.interv, ::self.interv] # only save the sparsely sampled grids
                        
                l += 1

        return self.x, self.y, self.ut, self.vt