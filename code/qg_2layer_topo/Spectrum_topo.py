
import numpy as np

def spectrum_topo(q_hat, p):
    N = p['N']
    kd = p['kd']
    kb = p['kb']
    U = p['U']
    hk = p['hk']

    k = np.concatenate((np.arange(0, N//2 + 1), np.arange(-N//2 + 1, 0)))
    KX, KY = np.meshgrid(k, k)
    dX = 1j * np.tile(k[None,:,None], (N, 1, 2))
    dY = 1j * np.tile(k[:,None,None], (1, N, 2))
    Laplacian = dX[:,:,0]**2 + dY[:,:,0]**2
    InvBT = 1 / Laplacian
    InvBT[0, 0] = 0
    InvBC = 1 / (Laplacian - kd**2)
    InvBC[0, 0] = 0
    k = np.concatenate((np.arange(0, N//2), [0], np.arange(-N//2 + 1, 0)))
    dX = 1j * np.tile(k[None,:,None], (N, 1, 2))
    dY = 1j * np.tile(k[:,None,None], (1, N, 2))

    mu1 = -kb**2 / U - kd**2
    mu2 = kb**2 / U - kd**2

    # Invert for psi
    q_bt = 0.5 * (q_hat[:,:,0] + q_hat[:,:,1])
    q_bc = 0.5 * (q_hat[:,:,0] - q_hat[:,:,1])
    psi_bt = InvBT * (q_bt - 0.5 * hk)
    psi_bc = InvBC * (q_bc + 0.5 * hk)
    psi_hat = np.zeros_like(q_hat)
    psi_hat[:,:,1] = psi_bt - psi_bc
    psi_hat[:,:,0] = psi_bt + psi_bc
    DX_psi_hat = dX * psi_hat
    DY_psi_hat = dY * psi_hat

    # Calculate KE and APE
    KE = np.zeros(N//2 + 1)
    APE = np.zeros_like(KE)
    for jj in range(N):
        for ii in range(N):
            k = np.sqrt(KX[ii,jj]**2 + KY[ii,jj]**2)
            if np.ceil(k) <= N//2:
                r = k - np.floor(k)
                floor_k = int(np.floor(k))
                ceil_k = int(np.ceil(k))
                KE[floor_k] += (1-r) * k**2 * (np.abs(psi_hat[ii,jj,0])**2 + np.abs(psi_hat[ii,jj,1])**2)
                APE[floor_k] += (1-r) * 0.5 * kd**2 * np.abs(psi_hat[ii,jj,0] - psi_hat[ii,jj,1])**2
                KE[ceil_k] += r * k**2 * (np.abs(psi_hat[ii,jj,0])**2 + np.abs(psi_hat[ii,jj,1])**2)
                APE[ceil_k] += r * 0.5 * kd**2 * np.abs(psi_hat[ii,jj,0] - psi_hat[ii,jj,1])**2

    KE *= 0.5 / (N**4)
    APE *= 0.5 / (N**4)

    return KE, APE
