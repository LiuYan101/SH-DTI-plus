"""Estimate a complete tensor volume with SH-DTI++."""
import argparse
from pathlib import Path
import numpy as np
import torch
from sh_dti_plus.io import load_image,save_image,check_geometry,load_config
from sh_dti_plus.model import SHDTIPlus
from sh_dti_plus.sh import sh_inverse,fit_sh
from sh_dti_plus.diffusion import beta_schedule,load_weights,sample
from sh_dti_plus.patches import patch_starts


def tensor_metrics(tensor):
    matrix=np.zeros(tensor.shape[:-1]+(3,3),dtype=np.float64)
    matrix[...,0,0],matrix[...,1,1],matrix[...,2,2]=[tensor[...,i] for i in range(3)]
    matrix[...,0,1]=matrix[...,1,0]=tensor[...,3]
    matrix[...,0,2]=matrix[...,2,0]=tensor[...,4]
    matrix[...,1,2]=matrix[...,2,1]=tensor[...,5]
    values=np.linalg.eigvalsh(matrix)
    md=values.mean(-1)
    fa=np.sqrt(1.5*np.sum((values-md[...,None])**2,axis=-1)/np.maximum(np.sum(values**2,axis=-1),1e-20))
    return md,fa


def main():
    p=argparse.ArgumentParser(description='Test SH-DTI++ on a complete volume')
    inp=p.add_mutually_exclusive_group(required=True)
    inp.add_argument('--dwi')
    inp.add_argument('--sh')
    p.add_argument('--mask',required=True)
    p.add_argument('--bvals')
    p.add_argument('--bvecs')
    p.add_argument('--reference-b',type=float,default=1000.)
    p.add_argument('--no-b-normalization',action='store_true')
    p.add_argument('--config',default='configs/sh_dti_plus.yaml')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=1244)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-ema',action='store_true')
    a=p.parse_args()
    if a.stride<=0 or a.stride>64 or a.batch_size<1:
        p.error('Use stride 1..64 and a positive batch size.')
    data,reference=load_image(a.dwi or a.sh)
    mask,mask_image=load_image(a.mask)
    check_geometry(reference,mask_image)
    if mask.ndim!=3 or not np.any(mask>0):
        raise ValueError('Expected a nonempty 3-D brain mask.')
    h,w,depth=mask.shape
    locations=[(z,x,y) for z in range(depth) for x in patch_starts(h,a.stride) for y in patch_starts(w,a.stride)]
    mask=mask>0
    if a.dwi:
        if not a.bvals or not a.bvecs:p.error('--dwi requires --bvals and --bvecs.')
        inverse,bvals,shell=sh_inverse(a.bvals,a.bvecs)
        sh=fit_sh(data,mask,inverse,bvals,shell,None if a.no_b_normalization else a.reference_b)
    else:
        sh=data
    if sh.shape!=mask.shape+(6,):
        raise ValueError('Expected XYZx6 SH coefficients on the mask voxel grid.')
    c=load_config(a.config)
    if c.data.image_size!=64 or not c.model.use_se:
        raise ValueError('Use the 64x64 SH-DTI++ configuration with SE enabled.')
    np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    model=SHDTIPlus(c).to(a.device)
    load_weights(model,a.checkpoint,a.device,not a.no_ema)
    model.eval()
    betas=beta_schedule(c,a.device)
    prediction=np.zeros((h,w,depth,6),dtype=np.float64)
    weights=np.zeros((h,w,depth),dtype=np.float64)
    window=np.maximum(np.outer(np.hanning(64),np.hanning(64)),1e-3)
    for start in range(0,len(locations),a.batch_size):
        batch=locations[start:start+a.batch_size]
        patches=np.stack([sh[x:x+64,y:y+64,z].transpose(2,0,1) for z,x,y in batch])
        generated=sample(model,torch.from_numpy(patches.copy()).to(a.device),betas)
        generated=generated.cpu().numpy().transpose(0,2,3,1)
        for (z,x,y),patch in zip(batch,generated):
            prediction[x:x+64,y:y+64,z]+=patch*window[...,None]
            weights[x:x+64,y:y+64,z]+=window
        print(f'SH-DTI++ patches {min(start+a.batch_size,len(locations))}/{len(locations)}',flush=True)
    tensor=prediction/weights[...,None]
    tensor/=np.array([1000,1000,1000,10000,10000,10000])
    tensor*=mask[...,None]
    md,fa=tensor_metrics(tensor)
    a.output.mkdir(parents=True,exist_ok=True)
    for name,array in [('tensor',tensor),('md',md),('fa',fa)]:
        save_image(array,a.output/f'{name}.nii.gz',reference)


if __name__=='__main__':
    main()
