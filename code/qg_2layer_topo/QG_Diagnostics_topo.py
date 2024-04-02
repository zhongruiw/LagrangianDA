
import numpy as np
from numpy.fft import ifft2

def qg_diagnostics_topo(q_hat, p):
    N = p['N']
    kd = p['kd']
    kb = p['kb']
    U = p['U']
    hk = p['hk']

    # Initialization of dX, dY, Laplacian, InvBT, InvBC
    k = np.concatenate((np.arange(0, N//2 + 1), np.arange(-N//2 + 1, 0)))
    dX = 1j * np.tile(k[None,:], (N, 1))
    dY = 1j * np.tile(k[:,None], (1, N))
    Laplacian = dX**2 + dY**2
    InvBT = 1 / Laplacian
    InvBT[0, 0] = 0
    InvBC = 1 / (Laplacian - kd**2)
    InvBC[0, 0] = 0

    mu1 = -kb**2 / U - kd**2
    mu2 = kb**2 / U - kd**2

    # Invert for psi
    q_bt = 0.5 * (q_hat[:,:,0] + q_hat[:,:,1])
    q_bc = 0.5 * (q_hat[:,:,0] - q_hat[:,:,1])
    psi_bt = InvBT * (q_bt - 0.5 * hk)
    psi_bc = InvBC * (q_bc + 0.5 * hk)

    # Real-Space quantities
    vt = np.real(ifft2(dX * psi_bt))
    vc = np.real(ifft2(dX * psi_bc))
    psic = np.real(ifft2(psi_bc))
    ut = np.real(ifft2(-dY * psi_bt))
    uc = np.real(ifft2(-dY * psi_bc))
    q1 = np.real(ifft2(q_hat[:,:,0]))
    q2 = np.real(ifft2(q_hat[:,:,1]))

    # Outputs
    vb = ((2 * np.pi * kd / N)**2) * np.sum(vt * psic)
    utz = np.mean(ut, axis=1)
    e = (2 * np.pi / N)**2 * np.sum(ut**2 + vt**2 + uc**2 + ut**2 + kd**2 * psic**2)
    etp = 0.5 * (2 * np.pi / N)**2 * np.sum((1 / mu1 * q1**2 + 1 / mu2 * q2**2))

    return vb, utz, e, etp
