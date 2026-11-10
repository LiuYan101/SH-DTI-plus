"""Generate noise-free or Rician-noisy DW images from a GT tensor."""
import argparse
import numpy as np
from sh_dti_plus.io import load_image, save_image, check_geometry, load_gradients
from sh_dti_plus.simulation import simulate_dw, rician_noise


def main():
    p = argparse.ArgumentParser(description='SH-DTI++ tensor-to-DW simulation')
    for name in ['tensor','s0','mask','bvals','bvecs','output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--s0-volume', type=int, default=0)
    p.add_argument('--noise-sigma', type=float, default=0.)
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()
    tensor, reference = load_image(a.tensor)
    s0, s0_image = load_image(a.s0)
    mask, mask_image = load_image(a.mask)
    check_geometry(reference, s0_image)
    check_geometry(reference, mask_image)
    if s0.ndim==4:
        s0 = s0[...,a.s0_volume]
    bvals, bvecs = load_gradients(a.bvals,a.bvecs)
    signal = simulate_dw(tensor,s0,bvals,bvecs,mask)
    if a.noise_sigma:
        signal = rician_noise(signal,a.noise_sigma,np.random.RandomState(a.seed))
    save_image(signal,a.output,reference)


if __name__=='__main__':
    main()
