"""Convert one DW volume to six SH coefficient maps."""
import argparse
from sh_dti_plus.io import load_image, save_image, check_geometry
from sh_dti_plus.sh import sh_inverse, fit_sh


def main():
    p = argparse.ArgumentParser(description='SH-DTI++ SH and b-value preprocessing')
    for name in ['dwi','mask','bvals','bvecs','output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--reference-b', type=float, default=1000.)
    p.add_argument('--no-b-normalization', action='store_true')
    a = p.parse_args()
    dwi, reference = load_image(a.dwi)
    mask, mask_image = load_image(a.mask)
    check_geometry(reference,mask_image)
    inverse,bvals,shell = sh_inverse(a.bvals,a.bvecs)
    sh = fit_sh(dwi,mask,inverse,bvals,shell,None if a.no_b_normalization else a.reference_b)
    save_image(sh,a.output,reference)


if __name__=='__main__':
    main()
