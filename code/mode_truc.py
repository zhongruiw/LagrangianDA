import numpy as np


def truncate(kk, r, style='circle'):
    '''
    1. require the input kk has shape (K,) or (K,K,...)
    2. r is the radius to be preserved.
    3. style = 'circle' or 'square', default is 'circle'
    4. return flattened modes with 'F' order with shape (k_left,...)
    '''
    K = kk.shape[0]

    if kk.ndim == 1:
        index_to_remove = np.zeros(K, dtype=bool)
        index_to_remove[r+1:-r] = True
        new_shape = np.array(kk.shape)
        new_shape[:2] = min((2*r+1), K)
        kk_cut = kk[~index_to_remove].reshape(new_shape)
        
    elif kk.ndim > 1:
        if style == 'square':
            index_to_remove = np.zeros((K, K), dtype=bool)
            index_to_remove[r+1:-r, :] = True
            index_to_remove[:, r+1:-r] = True
            k_left = min((2*r+1), K)**2
            
        elif style == 'circle':
            index_to_remove = np.ones((K, K), dtype=bool)
            k_left = 0
            for ix in range(K):
                for iy in range(K):
                    r2_xy = min((ix**2 + iy**2), ((ix-K)**2 + iy**2), ((iy-K)**2 + ix**2), ((ix-K)**2 + (iy-K)**2))
                    if r2_xy <= r**2:
                        index_to_remove[ix,iy] = False  
                        k_left += 1

        else:
            raise Exception("unknown style key input")
                                    
        # To retrieve elements in Fortran ('F') order:
        axes = np.arange(kk.ndim)
        axes[0], axes[1] = axes[1], axes[0]
        kk_T = np.transpose(kk, axes)
        kk_cut = kk_T[~index_to_remove.T]

        # Returned flattened truncatation modes with order 'F' 
        new_shape = np.array(kk.shape[1:])
        new_shape[0] = k_left
        kk_cut = kk_cut.reshape(new_shape, order='F')

    return kk_cut

# kx = np.fft.fftfreq(K) * K
# # truncate(kx, 8, style='circle')
# truncate(np.tile(KY[:,:,None],(1,1,23)), 3, style='circle')[:,0]


def inv_truncate(kk_truncated, r, K, style='circle'):
    ''' Recovers the original array from a truncated version by filling zeros.
        Parameters:
        - kk_truncated: The truncated array of shape (k_left,...). *if the original array has dim>1, kk_truncated should have dim>1
        - r: radius to be preserved.
        - style: 'circle' or 'square'.
    '''
    if kk_truncated.ndim == 1:
        recovered = np.zeros(K, dtype=kk_truncated.dtype)
        recovered[:r+1] = kk_truncated[:r+1]
        recovered[-r:] = kk_truncated[-r:]

    elif kk_truncated.ndim > 1:
        k_left = kk_truncated.shape[0]
        new_shape = [K] + list(kk_truncated.shape)
        new_shape[1] = K
        recovered = np.zeros(new_shape, dtype=kk_truncated.dtype)
        if style == 'square':
            k_cut = int(np.sqrt(k_left))
            shape_cut = [k_cut, k_cut] + list(kk_truncated.shape[1:])
            kk_temp = np.reshape(kk_truncated, shape_cut, order='F')
            recovered[:r+1, :r+1] = kk_temp[:r+1, :r+1]
            recovered[-r:, :r+1] = kk_temp[-r:, :r+1]
            recovered[:r+1, -r:] = kk_temp[:r+1, -r:]
            recovered[-r:, -r:] = kk_temp[-r:, -r:]
            
        elif style == 'circle':
            kx = np.fft.fftfreq(K) * K
            ky = np.fft.fftfreq(K) * K
            KX, KY = np.meshgrid(kx, ky)
            k_index_map = {(KX[iy, ix], KY[iy, ix]): (ix, iy) for ix in range(K) for iy in range(K) if (KX[iy, ix]**2 + KY[iy, ix]**2) <=r**2}
            for ik_, (k, ik) in enumerate(k_index_map.items()):
                ikx, iky = ik
                recovered[iky, ikx] = kk_truncated[ik_]

        else:
            raise Exception("unknown style key input")
            
    return recovered

# K = 16 
# kx = np.fft.fftfreq(K) * K
# kx_cut = truncate(kx, 3)
# kx_rec = inv_truncate(kx_cut, 3, K)
# print(kx_cut, kx_rec)