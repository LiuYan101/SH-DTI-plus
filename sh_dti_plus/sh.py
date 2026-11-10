"""Second-order real SH preprocessing with the training-time basis convention."""
import numpy as np
from scipy.special import sph_harm
from .io import load_gradients


def real_sh_basis(bvecs):
    theta = np.arccos(np.clip(bvecs[:,2], -1, 1))
    phi = np.mod(np.arctan2(bvecs[:,1], bvecs[:,0]), 2*np.pi)
    basis = np.zeros((len(bvecs), 6), dtype=np.float32)
    offset = 0
    for order in [0, 2]:
        m = np.arange(-order, order+1)
        values = sph_harm(m, order, phi[:,None], theta[:,None])
        if order:
            values[:,:order] = np.sqrt(2)*(-1.)**np.arange(order,0,-1)*values[:,:order:-1].imag
            values[:,order+1:] = np.sqrt(2)*(-1.)**np.arange(1,order+1)*values[:,order+1:].real
        basis[:,offset:offset+2*order+1] = values.real
        offset += 2*order+1
    return basis


def sh_inverse(bvals_path, bvecs_path):
    bvals, bvecs = load_gradients(bvals_path, bvecs_path)
    shell = np.flatnonzero(bvals>100)
    if shell.size < 6 or np.unique(bvals[shell]).size != 1:
        raise ValueError('Expected a single shell with at least six DW directions.')
    basis = real_sh_basis(bvecs[shell])
    if np.linalg.matrix_rank(basis) < 6:
        raise ValueError('Gradient directions do not support full-rank second-order SH.')
    # Preserve the shuffled Gram-Schmidt transform used by the trained model.
    np.random.seed(1234)
    inverse_sum = np.zeros_like(basis.T)
    for _ in range(1000):
        order = [0] + list(1+np.random.permutation(5))
        inverse = np.zeros_like(basis.T)
        for row in range(6):
            inverse[row] = basis[:,order[row]]
            for previous in range(row):
                norm = np.sum(inverse[previous]**2)
                if norm>1e-8:
                    inverse[row] -= np.sum(inverse[row]*inverse[previous])/(norm+1e-8)*inverse[previous]
            inverse[row] /= np.sqrt(np.sum(inverse[row]**2))
        inverse_sum += inverse[np.argsort(order)]
    return inverse_sum/1000/np.sqrt((basis[:,:1].T@basis[:,:1])[0,0]), bvals, shell


def fit_sh(dwi, mask, inverse, bvals, shell, reference_b=1000.):
    if dwi.shape[:-1] != mask.shape or dwi.shape[-1] != bvals.size:
        raise ValueError('DWI, mask and gradient dimensions differ.')
    b0_indices = np.flatnonzero(bvals<=100)
    if not len(b0_indices):
        raise ValueError('At least one b0 volume is required.')
    # The original pipeline uses the first b0 volume.
    b0 = dwi[...,b0_indices[0]:b0_indices[0]+1]
    ratio = np.clip(dwi[...,shell]/(b0+1e-10), 0, 1)
    if reference_b is not None:
        if reference_b<=0:
            raise ValueError('Reference b-value must be positive.')
        if not np.all(bvals[shell]==reference_b):
            ratio = ratio**(reference_b/bvals[shell])
    coefficients = np.einsum('cd,...d->...c', inverse, ratio)
    return (coefficients*(mask>0)[...,None]).astype(np.float32)
