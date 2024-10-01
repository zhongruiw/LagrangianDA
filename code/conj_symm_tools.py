import numpy as np


def verify_conjugate_symmetry(Y):
    """
    Checks if the matrix Y is conjugate symmetric based on the given conditions.
    
    Parameters:
    - Y: np.ndarray, a numpy array representing the matrix.
    
    Returns:
    - bool, True if the matrix Y is conjugate symmetric, False otherwise.
    """
    # Condition 1: Check if Y[0, 1:] is conjugate symmetric
    condition1 = np.allclose(Y[0, 1:], np.conj(Y[0, -1:0:-1]))
    
    # Condition 2: Check if Y[1:, 0] is conjugate symmetric
    condition2 = np.allclose(Y[1:, 0], np.conj(Y[-1:0:-1, 0]))
    
    # Condition 3: Check if Y[1:, 1:] is conjugate centrosymmetric
    condition3 = np.allclose(Y[1:, 1:], np.conj(Y[-1:0:-1, -1:0:-1]))
    
    # Return True if all conditions are met, False otherwise
    return condition1 and condition2 and condition3


def find_non_conjugate_pairs(Y):
    """
    Identifies pairs of entries that are not complex conjugate if the matrix Y does not pass the verification.
    
    Parameters:
    - Y: np.ndarray, a numpy array representing the matrix.
    
    Returns:
    - list of tuple, pairs of indices that are not complex conjugate.
    """
    non_conjugate_pairs = []
    
    # Check condition 1
    for i in range(1, Y.shape[1]):
        if not np.isclose(Y[0, i], np.conj(Y[0, -i])):
            non_conjugate_pairs.append(((0, i), (0, -i)))
    
    # Check condition 2
    for i in range(1, Y.shape[0]):
        if not np.isclose(Y[i, 0], np.conj(Y[-i, 0])):
            non_conjugate_pairs.append(((i, 0), (-i, 0)))
    
    # Check condition 3
    for i in range(1, Y.shape[0]):
        for j in range(1, Y.shape[1]):
            if not np.isclose(Y[i, j], np.conj(Y[-i, -j])):
                non_conjugate_pairs.append(((i, j), (-i, -j)))
    
    return non_conjugate_pairs


def avg_conj_symm(matrix, r1):
    '''
    matrix1, matrix2: psi_k, tau_k
    modify matrix1(2) to conjugate symmetric if omega_k or r_k is real-value, i.e., psi_k = psi_{-k}
    otherwise modfify matrix1_k and matrix2_{-k} to be conjugate symmetric, i.e.,psi_{k} = tau_{-k}
    by averaging the magnitude of real and imag parts of conjugate pairs'''
    # if input matrix is not complex, convert it to be complex
    if matrix.dtype != 'complex128':
        matrix = matrix + 0j
        
    K = matrix.shape[0]
    kx = np.fft.fftfreq(K) * K
    ky = np.fft.fftfreq(K) * K

    for ikx, kx_value in enumerate(kx):
        for iky, ky_value in enumerate(ky):
            if np.mod(kx_value,K/2)==0 and np.mod(ky_value,K/2)==0: # enforce K/2 mode to be real-value if K is even
                matrix[iky, ikx] = matrix[iky, ikx].real
            elif kx_value>0 or ((kx_value == 0 or kx_value==-K/2) and ky_value > 0): # half of modes
                if r1[iky,ikx,0].imag == 0:
                    real_mean = (matrix[iky, ikx, :].real + matrix[-iky, -ikx, :].real) / 2
                    imag_mean = (matrix[iky, ikx, :].imag - matrix[-iky, -ikx, :].imag) / 2
                    matrix[iky, ikx, :] = real_mean + 1j*imag_mean
                    matrix[-iky, -ikx, :] = real_mean - 1j*imag_mean
                else:
                    real_mean = (matrix[iky, ikx, 0].real + matrix[-iky, -ikx, 1].real) / 2
                    imag_mean = (matrix[iky, ikx, 0].imag - matrix[-iky, -ikx, 1].imag) / 2
                    matrix[iky, ikx, 0] = real_mean + 1j*imag_mean
                    matrix[-iky, -ikx, 1] = real_mean - 1j*imag_mean
                    
                    real_mean = (matrix[-iky, -ikx, 0].real + matrix[iky, ikx, 1].real) / 2
                    imag_mean = (matrix[-iky, -ikx, 0].imag - matrix[iky, ikx, 1].imag) / 2
                    matrix[-iky, -ikx, 0] = real_mean + 1j*imag_mean
                    matrix[iky, ikx, 1] = real_mean - 1j*imag_mean

    return matrix


def map_conj_symm(matrix, r1):
    '''
    matrix1, matrix2: psi_k, tau_k
    modify matrix1(2) to conjugate symmetric if omega_k or r_k is real-value, i.e., psi_k = psi_{-k}
    otherwise modfify matrix1_k and matrix2_{-k} to be conjugate symmetric, i.e.,psi_{k} = tau_{-k}
    by mapping the magnitude of real and imag parts from one half to the other halof of conjugate pairs'''
    # if input matrix is not complex, convert it to be complex
    if matrix.dtype != 'complex128':
        matrix = matrix + 0j
        
    K = matrix.shape[0]
    kx = np.fft.fftfreq(K) * K
    ky = np.fft.fftfreq(K) * K

    for ikx, kx_value in enumerate(kx):
        for iky, ky_value in enumerate(ky):
            if np.mod(kx_value,K/2)==0 and np.mod(ky_value,K/2)==0: # enforce K/2 mode to be real-value if K is even
                matrix[iky, ikx] = matrix[iky, ikx].real
            elif kx_value>0 or ((kx_value == 0 or kx_value==-K/2) and ky_value > 0): # half of modes
                if r1[iky,ikx,0].imag == 0:
                    matrix[-iky, -ikx] = matrix[iky, ikx].conj()
                else:
                    matrix[-iky, -ikx, 1] = matrix[iky, ikx, 0].conj()
                    matrix[iky, ikx, 1] = matrix[-iky, -ikx, 0].conj()

    return matrix