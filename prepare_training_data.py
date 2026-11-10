"""Create paired 64x64 SH/tensor patches from selected subjects."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from sh_dti_plus.io import load_image, check_geometry, load_gradients
from sh_dti_plus.sh import sh_inverse, fit_sh
from sh_dti_plus.simulation import simulate_dw, rician_noise
from sh_dti_plus.patches import pad_xy, patch_starts


def main():
    p = argparse.ArgumentParser(description='SH-DTI++ training data generation')
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--bvals', required=True)
    p.add_argument('--bvecs', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--crop', nargs=6, type=int, metavar=('X0','X1','Y0','Y1','Z0','Z1'))
    p.add_argument('--stride', type=int, default=32)
    p.add_argument('--noise-levels', nargs='+', type=float, default=list(np.linspace(.005,.045,9)))
    p.add_argument('--reference-b', type=float, default=1000.)
    p.add_argument('--no-b-normalization', action='store_true')
    a = p.parse_args()
    if a.stride<=0 or a.stride>64 or min(a.noise_levels)<0:
        p.error('Use stride 1..64 and nonnegative noise levels.')
    inverse,bvals,shell = sh_inverse(a.bvals,a.bvecs)
    _,bvecs = load_gradients(a.bvals,a.bvecs)
    cases = json.loads(a.cases.read_text(encoding='utf-8'))
    if not cases:
        raise ValueError('Case manifest is empty.')
    if len({r['id'] for r in cases}) != len(cases):
        raise ValueError('Case IDs must be unique.')
    for name in ['sh_coe','gt_tensor']:
        (a.output/name).mkdir(parents=True,exist_ok=True)
    records=[]
    for case in cases:
        def image(name):
            path = Path(case[name])
            return load_image(path if path.is_absolute() else a.cases.parent/path)
        tensor,reference = image('tensor')
        mask,mask_image = image('mask')
        check_geometry(reference,mask_image)
        if tensor.shape != mask.shape+(6,):
            raise ValueError('Expected XYZx6 tensor and XYZ mask.')
        if 'dwi' in case:
            dwi,dwi_image = image('dwi')
            check_geometry(reference,dwi_image)
        else:
            s0,s0_image = image('s0')
            check_geometry(reference,s0_image)
            if s0.ndim==4:
                s0=s0[...,case.get('s0_volume',0)]
            dwi=simulate_dw(tensor,s0,bvals,bvecs,mask)
        if a.crop:
            x0,x1,y0,y1,z0,z1=a.crop
            if not (0<=x0<x1<=mask.shape[0] and 0<=y0<y1<=mask.shape[1] and 0<=z0<z1<=mask.shape[2]):
                raise ValueError('Requested crop is outside the input volume.')
            crop=(slice(x0,x1),slice(y0,y1),slice(z0,z1))
            dwi,tensor,mask=dwi[crop],tensor[crop],mask[crop]
        else:
            x0=y0=z0=0
        h,w,depth=mask.shape
        if min(h,w,depth)<1 or dwi.shape != mask.shape+(bvals.size,):
            raise ValueError('Expected nonempty spatial dimensions and one image per b-value.')
        dwi,tensor,mask=pad_xy(dwi),pad_xy(tensor),pad_xy(mask)
        hp,wp=mask.shape[:2]
        locations=[(x,y) for x in patch_starts(hp,a.stride) for y in patch_starts(wp,a.stride)]
        for z in range(depth):
            # Assign configured noise levels to the patches of each slice.
            levels=np.asarray(a.noise_levels)[np.random.choice(len(a.noise_levels),len(locations),replace=len(locations)!=len(a.noise_levels))]
            for (x,y),sigma in zip(locations,levels):
                sl=(slice(x,x+64),slice(y,y+64),z)
                noisy=rician_noise(dwi[sl],float(sigma),np.random)
                sh=fit_sh(noisy,mask[sl],inverse,bvals,shell,None if a.no_b_normalization else a.reference_b)
                index=len(records)
                for name,data in [('sh_coe',sh),('gt_tensor',tensor[sl])]:
                    prefix='sph' if name=='sh_coe' else 'tensor'
                    target=a.output/name/f'{prefix}_{index:06d}.nii.gz'
                    if target.exists():
                        raise FileExistsError(target)
                    nib.save(nib.Nifti1Image(data.astype(np.float32),np.eye(4)),str(target))
                records.append(dict(index=index,subject=case['id'],x=x+x0,y=y+y0,z=z+z0,noise_sigma=float(sigma)))
        print(f"{case['id']}: {len(locations)*depth} training pairs",flush=True)
    (a.output/'patch_index.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    (a.output/'preprocessing.json').write_text(json.dumps(dict(
        method='SH-DTI++',tensor_order=['Dxx','Dyy','Dzz','Dxy','Dxz','Dyz'],
        tensor_units='mm^2/s',patch_size=64,stride=a.stride,crop=a.crop,
        reference_b=None if a.no_b_normalization else a.reference_b,
        noise_levels=a.noise_levels,bvals=bvals.tolist(),bvecs=bvecs.tolist(),
        subjects=[r['id'] for r in cases],seed=1234),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
