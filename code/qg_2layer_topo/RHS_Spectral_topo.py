
import numpy as np
from numpy.fft import fft2, ifft2

# Assuming 'hk', 'Ut', and 'p' (parameters) are defined elsewhere in the code
# The translation assumes that 'p' is a dictionary-like object containing the necessary parameters

def rhs_spectral_topo(q_hat, p, Ut):
    N = p['N']
    kd = p['kd']
    kb = p['kb']
    r = p['r']
    hk = p['hk']

    # Initialization of variables
    k = np.concatenate((np.arange(0, N//2 + 1), np.arange(-N//2 + 1, 0)))
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
    # For the dealiased jacobian
    k = np.concatenate((np.arange(0, 0.75*N), [0], np.arange(-0.75*N + 1, 0)))
    DX = 1j * np.tile(k[None,:,None], (int(1.5 * N), 1, 2))
    DY = 1j * np.tile(k[:,None,None], (1, int(1.5 * N), 2))

    # Invert for psi
    q_bt = 0.5 * (q_hat[:,:,0] + q_hat[:,:,1])
    q_bc = 0.5 * (q_hat[:,:,0] - q_hat[:,:,1])
    psi_bt = InvBT * (q_bt - 0.5 * hk)
    psi_bc = InvBC * (q_bc + 0.5 * hk)
    psi_hat = np.zeros_like(q_hat)
    psi_hat[:,:,1] = psi_bt - psi_bc
    psi_hat[:,:,0] = psi_bt + psi_bc

    # Calculate Ekman plus beta plus mean shear
    RHS = np.zeros_like(q_hat)
    RHS[:,:,0] = -Ut * dX[:,:,0] * q_hat[:,:,0] - (kb**2 + Ut * kd**2) * dX[:,:,0] * psi_hat[:,:,0]
    RHS[:,:,1] = Ut * dX[:,:,0] * q_hat[:,:,1] - (kb**2 - Ut * kd**2) * dX[:,:,0] * psi_hat[:,:,1] - (r * Laplacian * psi_hat[:,:,1] - Ut * dX[:,:,0] * hk)

    # For using a 3/2-rule dealiased jacobian
    Psi_hat = np.zeros((int(1.5 * N), int(1.5 * N), 2))
    Q_hat = np.zeros((int(1.5 * N), int(1.5 * N), 2))

    Psi_hat[:N//2, :N//2, :] = (9/4) * psi_hat[:N//2, :N//2, :]
    Psi_hat[:N//2, -N//2:, :] = (9/4) * psi_hat[:N//2, N//2:, :]
    Psi_hat[-N//2:, :N//2, :] = (9/4) * psi_hat[N//2:, :N//2, :]
    Psi_hat[-N//2:, -N//2:, :] = (9/4) * psi_hat[N//2:, N//2:, :]

    Q_hat[:N//2, :N//2, :] = (9/4) * q_hat[:N//2, :N//2, :]
    Q_hat[:N//2, -N//2:, :] = (9/4) * q_hat[:N//2, N//2:, :]
    Q_hat[-N//2:, :N//2+1, :] = (9/4) * q_hat[N//2:, :N//2+1, :]
    Q_hat[-N//2:, -N//2:, :] = (9/4) * q_hat[N//2:, N//2:, :]

    # Calculate u.gradq on 3/2 grid
    u = np.real(ifft2(-DY * Psi_hat))
    v = np.real(ifft2(DX * Psi_hat))
    qx = np.real(ifft2(DX * Q_hat))
    qy = np.real(ifft2(DY * Q_hat))
    jaco_real = u * qx + v * qy

    # FFT, 3/2 grid; factor of (4/9) scales fft
    Jaco_hat = (4/9) * fft2(jaco_real)

    # Reduce to normal grid
    jaco_hat = np.zeros((N, N, 2))
    jaco_hat[:N//2, :N//2, :] = Jaco_hat[:N//2, :N//2, :]
    jaco_hat[:N//2, N//2:, :] = Jaco_hat[:N//2, -N//2:, :]
    jaco_hat[N//2:, :N//2, :] = Jaco_hat[-N//2:, :N//2, :]
    jaco_hat[N//2:, N//2:, :] = Jaco_hat[-N//2:, -N//2:, :]
    
    # Put it all together
    RHS -= jaco_hat  # Assuming 'jaco_hat' is defined in the above commented part

    return RHS, psi_hat
