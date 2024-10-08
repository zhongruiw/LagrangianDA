''' energy spectrum'''
import numpy as np


def adjust_ik(ik, cut, K):
    if ik < K//2 - cut:
        return ik
    else:
        return ik -cut*2-1

def trunc2full(psi_k, K, cut):
    N = psi_k.shape[-1]
    psi_k_full = np.zeros((K,K,N), dtype='complex')
    kx = np.fft.fftfreq(K) * K
    ky = np.fft.fftfreq(K) * K

    for ikx,kx_value in enumerate(kx):
        for iky,ky_value in enumerate(ky): 
            if (kx_value == 0 and ky_value==0) or abs(kx_value) >= (K//2-cut) or abs(ky_value) >= (K//2-cut):  # Skip the case where k_mag is 0 and truncation
                continue
            adj_ikx = adjust_ik(ikx,cut,K); adj_iky = adjust_ik(iky,cut,K)
            psi_k_full[iky,ikx,:] = psi_k[adj_iky,adj_ikx,:]

    return psi_k_full


def ene_spectrum(psi_hat, K, kd, topo, r_cut):
    '''
    psi_hat should be of shape (N, K, K, 2)
    '''
    N = psi_hat.shape[0]
    hk = np.fft.fft2(topo)
    K_half = K // 2
    E_mode = np.zeros((N,K_half + 1,2))
    KE = np.zeros((N,K_half + 1))
    APE = np.zeros_like(KE)
    E = np.zeros_like(KE)
    ETP = np.zeros_like(KE)
    kx = np.fft.fftfreq(K) * K
    ky = np.fft.fftfreq(K) * K
    KX, KY = np.meshgrid(kx, ky)

    for jj in range(K):
        for ii in range(K):
            k = np.sqrt(KX[ii, jj]**2 + KY[ii, jj]**2)
            if np.ceil(k) <= K_half and np.ceil(k) <= r_cut:
                r = k - np.floor(k)
                floor_k = int(np.floor(k))
                ceil_k = int(np.ceil(k))

                q_hat = np.zeros((N,2), dtype=complex)
                q_hat[:, 0] = -k**2 * psi_hat[:, ii, jj, 0] + kd**2/2*(psi_hat[:, ii, jj, 1]-psi_hat[:, ii, jj, 0])
                q_hat[:, 1] = -k**2 * psi_hat[:, ii, jj, 1] + kd**2/2*(psi_hat[:, ii, jj, 0]-psi_hat[:, ii, jj, 1]) + hk[ii, jj]

                KE[:, floor_k] += (1 - r) * (k**2) * (np.abs(psi_hat[:, ii, jj, 0])**2 + np.abs(psi_hat[:, ii, jj, 1])**2)
                APE[:, floor_k] += (1 - r) * (.5 * kd**2) * np.abs(psi_hat[:, ii, jj, 0] - psi_hat[:, ii, jj, 1])**2
                E[:, floor_k] += (1 - r) * (KE[:, floor_k] + APE[:, floor_k])
                ETP[:, floor_k] += (1 - r) * (np.abs(q_hat[:, 0])**2 + np.abs(q_hat[:, 1])**2)
                E_mode[:, floor_k, :] += (1 - r) * (np.abs(psi_hat[:, ii, jj, :])**2)                                
                
                if ceil_k != floor_k:  # Only update if ceil(k) and floor(k) are different
                    KE[:, ceil_k] += r * (k**2) * (np.abs(psi_hat[:, ii, jj, 0])**2 + np.abs(psi_hat[:, ii, jj, 1])**2)
                    APE[:, ceil_k] += r * (.5 * kd**2) * np.abs(psi_hat[:, ii, jj, 0] - psi_hat[:, ii, jj, 1])**2
                    E[:, ceil_k] += r * (KE[:, ceil_k] + APE[:, ceil_k])
                    ETP[:, ceil_k] += r * (np.abs(q_hat[:, 0])**2 + np.abs(q_hat[:, 1])**2)
                    E_mode[:, ceil_k, :] += r * (np.abs(psi_hat[:, ii, jj, :])**2)                                

    KE = .5 * KE / (K**4)
    APE = .5 * APE / (K**4)
    E = .5 * E / (K**4)
    ETP = .5 * ETP / (K**4)
    E_mode = E_mode / (K**4)

    KE = np.mean(KE, axis=0)
    APE = np.mean(APE, axis=0)
    E = np.mean(E, axis=0)
    ETP = np.mean(ETP, axis=0)
    E_mode = np.mean(E_mode, axis=0)
    
    return KE, APE, E, ETP, E_mode
    
# compute difference
def ene_spectrum1(psi_hat, K, kd, topo, r1, r2):
    hk = np.fft.fft2(topo)
    K_half = K // 2
    E_mode_res = np.zeros((K_half + 1,2), dtype='complex')
    kx = np.fft.fftfreq(K) * K
    ky = np.fft.fftfreq(K) * K
    KX, KY = np.meshgrid(kx, ky)

    for jj in range(K):
        for ii in range(K):
            k = np.sqrt(KX[ii, jj]**2 + KY[ii, jj]**2)
            if np.ceil(k) <= K_half:
                r = k - np.floor(k)
                floor_k = int(np.floor(k))
                ceil_k = int(np.ceil(k))

                E_mode_res[floor_k,:] += (1 - r) * r1[ii,jj,:]*r2[ii,jj,:] * (-2 * (psi_hat[ii, jj, 0].real * psi_hat[ii, jj, 1].real + psi_hat[ii, jj, 0].imag * psi_hat[ii, jj, 1].imag)) 
                                                  #+2 * (psi_hat[ii, jj, 0].real * psi_hat[ii, jj, 0].imag + psi_hat[ii, jj, 1].real * psi_hat[ii, jj, 1].imag)) 
                
                if ceil_k != floor_k:  # Only update if ceil(k) and floor(k) are different
                    E_mode_res[ceil_k,:] += r * r1[ii,jj,:]*r2[ii,jj,:] * (-2 * (psi_hat[ii, jj, 0].real * psi_hat[ii, jj, 1].real + psi_hat[ii, jj, 0].imag * psi_hat[ii, jj, 1].imag)) 
                                               #+2 * (psi_hat[ii, jj, 0].real * psi_hat[ii, jj, 0].imag + psi_hat[ii, jj, 1].real * psi_hat[ii, jj, 1].imag))                               

    E_mode_res = E_mode_res / (K**4)
    
    return E_mode_res.real
