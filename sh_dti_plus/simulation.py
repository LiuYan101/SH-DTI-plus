"""Simulate DW signal from a physical diffusion tensor."""
import numpy as np


def simulate_dw(tensor, s0, bvals, bvecs, mask=None):
    # Tensor order: Dxx, Dyy, Dzz, Dxy, Dxz, Dyz; units: mm^2/s.
    tensor = np.asarray(tensor, dtype=np.float64)
    s0 = np.asarray(s0, dtype=np.float64)
    if tensor.shape != s0.shape + (6,):
        raise ValueError('Expected XYZx6 tensor and XYZ S0.')
    gx, gy, gz = bvecs.T
    design = np.column_stack([gx*gx, gy*gy, gz*gz, 2*gx*gy, 2*gx*gz, 2*gy*gz])
    adc = np.einsum('...c,nc->...n', tensor, design)
    signal = s0[...,None] * np.exp(-adc*bvals)
    if mask is not None:
        if mask.shape != s0.shape:
            raise ValueError('Mask and tensor shapes differ.')
        signal *= (mask>0)[...,None]
    if not np.isfinite(signal).all():
        raise ValueError('Simulated signal is non-finite; check tensor units.')
    return signal.astype(np.float32)


def rician_noise(signal, sigma, rng):
    if sigma < 0:
        raise ValueError('Noise sigma must be nonnegative.')
    return np.sqrt((signal + sigma*rng.normal(size=signal.shape))**2
                   + (sigma*rng.normal(size=signal.shape))**2)
