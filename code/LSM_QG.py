import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf, ccf
from conj_symm_tools import avg_conj_symm


def solve_eigen(K, beta, kd, U):
    '''
    compute eigenvalues and eigenvectors of linearized QG
    '''
    omega1 = np.zeros((K,K), dtype=complex)
    omega2 = np.zeros((K,K), dtype=complex)
    r1 = np.zeros((K,K,2), dtype=complex)
    r2 = np.zeros((K,K,2), dtype=complex)
    beta_value = beta; kd_value = kd+0j; U_value = U;
    kx = np.fft.fftfreq(K) * K
    ky = np.fft.fftfreq(K) * K

    for ikx,kx_value in enumerate(kx):
        for iky,ky_value in enumerate(ky):  
            k_mag = kx_value**2 + ky_value**2
            if k_mag == 0:  # Skip the case where k_mag is 0 to avoid division by zero
                continue

            # Compute omega_k1 and omega_k2
            omega_k1 = (kx_value * (beta_value * (kd_value**2 + 2 * k_mag) + np.sqrt(beta_value**2 * kd_value**4 + 4 * k_mag**2 * (-kd_value**4 + k_mag**2) * U_value**2))) / (2 * k_mag * (kd_value**2 + k_mag))
            omega_k2 = kx_value * (beta_value * (kd_value**2 + 2 * k_mag) - np.sqrt(beta_value**2 * kd_value**4 + 4 * k_mag**2 * (-kd_value**4 + k_mag**2) * U_value**2)) / (2 * k_mag * (kd_value**2 + k_mag))
            omega1[iky,ikx] = omega_k1
            omega2[iky,ikx] = omega_k2

            # Compute r_k1 and r_k2
            r_k1 = np.array([
                [- (2 * U_value * k_mag**2 - np.sqrt(beta_value**2 * kd_value**4 + 4 * k_mag**2 * (-kd_value**4 + k_mag**2) * U_value**2)) / (kd_value**2 * (beta_value - 2 * k_mag * U_value))],
                [1]
            ], dtype=complex)
            r_k2 = np.array([
                [- (2 * U_value * k_mag**2 + np.sqrt(beta_value**2 * kd_value**4 + 4 * k_mag**2 * (-kd_value**4 + k_mag**2) * U_value**2)) / (kd_value**2 * (beta_value - 2 * k_mag * U_value))],
                [1]
            ], dtype=complex)
            r1[iky,ikx,:] = np.squeeze(r_k1)
            r2[iky,ikx,:] = np.squeeze(r_k2)
            
    eigens = {
        'omega1': omega1,
        'omega2': omega2,
        'r1': r1,
        'r2': r2
    }
            
    return eigens


def CCF(data, gamma, omega):
    '''
    Ansatz of cross-correlation between real and imaginary parts 
    '''
    return np.exp(-gamma*data) * np.sin(omega*data)


def ACF(data, gamma, omega):
    '''
    Ansatz of auto-correlation of real part
    '''
    return np.exp(-gamma*data) * np.cos(omega*data)


def calibrate_OU(r1, r2, psi1_hat_t, psi2_hat_t, K, dt, Lag, r_cut, style):
    '''
    calibration of complex OU process modeling two eigenmodes
    - Lag: int, lag for computing the ACF
    '''
    
    tt = np.linspace(0, Lag*dt, num=Lag+1, endpoint=True) # time interval to plot the ACF or cross-correlation function
    gamma_est = np.zeros((K,K,2))
    omega_est = np.zeros((K,K,2))
    omega_est_ccf = np.zeros((K,K,2))
    f_est = np.zeros((K,K,2), dtype='complex')
    sigma_est = np.zeros((K,K,2))
    kx = np.fft.fftfreq(K) * K
    ky = np.fft.fftfreq(K) * K

    for ikx,kx_value in enumerate(kx):
        for iky,ky_value in enumerate(ky): 
            if (kx_value == 0 and ky_value==0):  # Skip the case where k_mag is 0 and truncation
                continue
            elif style == 'square' and abs(kx_value) > r_cut or abs(ky_value) > r_cut:
                continue
            elif style == 'circle' and (kx_value**2 + ky_value**2) > r_cut**2:
                continue
            else:
                eigenmat_inv = np.linalg.inv(np.array([r1[iky,ikx,:],r2[iky,ikx,:]]).T)
                eigenmode = eigenmat_inv @ np.array([psi1_hat_t[iky,ikx,:],psi2_hat_t[iky,ikx,:]])
                psi1_hat = eigenmode[0,:] 
                psi2_hat = eigenmode[1,:]
                acf_psi1 = acf(np.real(psi1_hat), nlags=Lag, fft=True) 
                acf_psi2 = acf(np.real(psi2_hat), nlags=Lag, fft=True) 
                ccf_psi1 = -ccf(np.real(psi1_hat), np.imag(psi1_hat), adjusted=False)[:Lag+1]
                ccf_psi2 = -ccf(np.real(psi2_hat), np.imag(psi2_hat), adjusted=False)[:Lag+1]

                # Fitting the cross-correlation function
                x0 = [0.5, 0.5]
                x1, _ = curve_fit(ACF, tt, acf_psi1, p0=x0, check_finite=True, maxfev=2000)
                x1_, _ = curve_fit(CCF, tt, ccf_psi1, p0=x0, check_finite=True, maxfev=2000)
                # Estimated values
                gamma1_est = x1[0]
                omega1_est = x1[1]
                omega1_est_ccf = x1_[1]
                m1 = np.mean(psi1_hat)
                E1 = np.var(psi1_hat)
                T1 = gamma1_est / (gamma1_est**2 + omega1_est**2)
                theta1 = omega1_est / (gamma1_est**2 + omega1_est**2)
                f1_est = m1 * (T1 - 1j*theta1) / (T1**2 + theta1**2)
                sigma1_est = np.sqrt(2*E1*T1 / (T1**2 + theta1**2))
                gamma_est[iky,ikx,0] = gamma1_est; omega_est[iky,ikx,0] = omega1_est; f_est[iky,ikx,0] = f1_est; sigma_est[iky,ikx,0] = sigma1_est;
                omega_est_ccf[iky,ikx,0] = omega1_est_ccf;

                x2, _ = curve_fit(ACF, tt, acf_psi2, p0=x0, check_finite=True, maxfev=2000)
                x2_, _ = curve_fit(CCF, tt, ccf_psi2, p0=x0, check_finite=True, maxfev=2000)
                # Estimated values
                gamma2_est = x2[0]
                omega2_est = x2[1]
                omega2_est_ccf = x2_[1]
                m2 = np.mean(psi2_hat)
                E2 = np.var(psi2_hat)
                T2 = gamma2_est / (gamma2_est**2 + omega2_est**2)
                theta2 = omega2_est / (gamma2_est**2 + omega2_est**2)
                f2_est = m2 * (T2 - 1j*theta2) / (T2**2 + theta2**2)
                sigma2_est = np.sqrt(2*E2*T2 / (T2**2 + theta2**2))
                gamma_est[iky,ikx,1] = gamma2_est; omega_est[iky,ikx,1] = omega2_est; f_est[iky,ikx,1] = f2_est; sigma_est[iky,ikx,1] = sigma2_est;
                omega_est_ccf[iky,ikx,1] = omega2_est_ccf;
            
    # ensure conjugate symmetry by average
    gamma_est = avg_conj_symm(gamma_est, r1).real    
    omega_est_ccf = (avg_conj_symm(1j * omega_est_ccf, r1) / 1j).real
    f_est = avg_conj_symm(f_est, r1)
    sigma_est = avg_conj_symm(sigma_est, r1).real 

    # calibrate the sign of ACF-estimated omega_k according to CCF-estimated omega_k
    sign1 = np.sign(omega_est)
    sign2 = np.sign(omega_est_ccf)
    sign2_corrected = np.where((sign1 == 0) & (sign2 == 0), 0, sign2)
    omega_est_ca = np.abs(omega_est) * sign2_corrected

    # ensure conjugate symmetry by average
    omega_est_ca = (avg_conj_symm(1j * omega_est_ca, r1) / 1j).real

    est_params = {
        'gamma': gamma_est,
        'omega_ccf': omega_est_ccf,
        'omega_acf': omega_est,
        'omega': omega_est_ca,
        'f': f_est,
        'sigma': sigma_est,
        'r_cut': r_cut
    }

    return est_params


def run_OU(psi_k0, tau_k0, K, N, dt, r_cut, r1, r2, gamma, omega, f, sigma, style='circle', s_rate=1):
    N_sub = int(N//s_rate)
    psi_k = np.zeros((K, K, N_sub), dtype=complex)
    tau_k = np.zeros((K, K, N_sub), dtype=complex)
    psi_k[:,:,0] = psi_k0
    tau_k[:,:,0] = tau_k0
    psi_k1 = np.zeros((K,K), dtype=complex)
    tau_k1 = np.zeros((K,K), dtype=complex)
    Kx = np.fft.fftfreq(K) * K
    Ky = np.fft.fftfreq(K) * K
    
    for ikx,kx_value in enumerate(Kx):
        for iky,ky_value in enumerate(Ky):   
            if (kx_value == 0 and ky_value==0):  # Skip the case where k_mag is 0 and truncation
                continue
            elif style == 'square' and abs(kx_value) > r_cut or abs(ky_value) > r_cut:
                continue
            elif style == 'circle' and (kx_value**2 + ky_value**2) > r_cut**2:
                continue
            elif kx_value > 0 or ((kx_value == 0 or kx_value==-K/2) and ky_value > 0): # half of modes
                for i in range(1, N):
                    noise1 = np.random.randn() + 1j * np.random.randn()
                    noise2 = np.random.randn() + 1j * np.random.randn()
                    if r1[iky,ikx,0].imag == 0:
                        psi_k1[iky,ikx] = psi_k0[iky,ikx] + (-gamma[iky,ikx,0] + 1j * omega[iky,ikx,0]) * psi_k0[iky,ikx] * dt + f[iky,ikx,0] * dt + sigma[iky,ikx,0]/np.sqrt(2) * np.sqrt(dt) * noise1
                        psi_k1[-iky,-ikx] = psi_k1[iky,ikx].real - 1j * psi_k1[iky,ikx].imag
                        tau_k1[iky,ikx] = tau_k0[iky,ikx] + (-gamma[iky,ikx,1] + 1j * omega[iky,ikx,1]) * tau_k0[iky,ikx] * dt + f[iky,ikx,1] * dt + sigma[iky,ikx,1]/np.sqrt(2) * np.sqrt(dt) * noise2
                        tau_k1[-iky,-ikx] = tau_k1[iky,ikx].real - 1j * tau_k1[iky,ikx].imag
                    else:
                        psi_k1[iky,ikx] = psi_k0[iky,ikx] + (-gamma[iky,ikx,0] + 1j * omega[iky,ikx,0]) * psi_k0[iky,ikx] * dt + f[iky,ikx,0] * dt + sigma[iky,ikx,0]/np.sqrt(2) * np.sqrt(dt) * noise1
                        tau_k1[-iky,-ikx] = psi_k1[iky,ikx].real - 1j * psi_k1[iky,ikx].imag
                        psi_k1[-iky,-ikx] = psi_k0[-iky,-ikx] + (-gamma[-iky,-ikx,0] + 1j * omega[-iky,-ikx,0]) * psi_k0[-iky,-ikx] * dt + f[-iky,-ikx,0] * dt + sigma[-iky,-ikx,0]/np.sqrt(2) * np.sqrt(dt) * noise2
                        tau_k1[iky,ikx] = psi_k1[-iky,-ikx].real - 1j * psi_k1[-iky,-ikx].imag
                    psi_k0 = psi_k1
                    tau_k0 = tau_k1
                    
                    if i % s_rate == 0:
                        i_sub = int(i / s_rate)
                        psi_k[:, :, i_sub] = psi_k1
                        tau_k[:, :, i_sub] = tau_k1

    return psi_k, tau_k


def eigen2layer(K,r_cut,r1,r2,psi_k,tau_k, style='circle'):
    N = psi_k.shape[-1]
    Kx = np.fft.fftfreq(K) * K
    Ky = np.fft.fftfreq(K) * K
    psi1_k = np.zeros((K,K,N), dtype='complex')
    psi2_k = np.zeros((K,K,N), dtype='complex')
    
    for ikx,kx_value in enumerate(Kx):
        for iky,ky_value in enumerate(Ky): 
            if (kx_value == 0 and ky_value==0):  # Skip the case where k_mag is 0 and truncation
                continue
            elif style == 'square' and abs(kx_value) > r_cut or abs(ky_value) > r_cut:
                continue
            elif style == 'circle' and (kx_value**2 + ky_value**2) > r_cut**2:
                continue
            else:
                eigenmat = np.array([r1[iky,ikx,:],r2[iky,ikx,:]]).T
                layer = eigenmat @ np.array([psi_k[iky,ikx,:],tau_k[iky,ikx,:]])
                psi1_k[iky,ikx,:] = layer[0,:] 
                psi2_k[iky,ikx,:] = layer[1,:] 
        
    return psi1_k, psi2_k


def layer2eigen(K,r_cut,r1,r2,psi1_k,psi2_k, style='circle'):
    N = psi1_k.shape[-1]
    Kx = np.fft.fftfreq(K) * K
    Ky = np.fft.fftfreq(K) * K
    psi_k = np.zeros((K,K,N), dtype='complex')
    tau_k = np.zeros((K,K,N), dtype='complex')

    for ikx,kx_value in enumerate(Kx):
        for iky,ky_value in enumerate(Ky): 
            if (kx_value == 0 and ky_value==0):  # Skip the case where k_mag is 0 and truncation
                continue
            elif style == 'square' and abs(kx_value) > r_cut or abs(ky_value) > r_cut:
                continue
            elif style == 'circle' and (kx_value**2 + ky_value**2) > r_cut**2:
                continue
            else:
                eigenmat_inv = np.linalg.inv(np.array([r1[iky,ikx,:],r2[iky,ikx,:]]).T)
                eigenmode = eigenmat_inv @ np.array([psi1_k[iky,ikx,:N],psi2_k[iky,ikx,:N]])
                psi_k[iky,ikx,:] = eigenmode[0,:] 
                tau_k[iky,ikx,:] = eigenmode[1,:]
            
    return psi_k, tau_k


