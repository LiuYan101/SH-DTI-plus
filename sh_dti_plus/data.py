"""Paired SH coefficient and tensor training patches."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .io import load_image


class PatchDataset(Dataset):
    def __init__(self,root):
        root=Path(root)
        self.sh=sorted((root/'sh_coe').glob('sph_*.nii.gz'))
        self.tensor=[]
        for sh in self.sh:
            target=root/'gt_tensor'/sh.name.replace('sph_','tensor_',1)
            if not target.is_file():
                raise FileNotFoundError(target)
            self.tensor.append(target)
        if not self.sh or len(self.tensor)!=len(list((root/'gt_tensor').glob('tensor_*.nii.gz'))):
            raise ValueError('Empty or unmatched training patch directories.')

    def __len__(self):
        return len(self.sh)

    def __getitem__(self,index):
        sh,_=load_image(self.sh[index])
        tensor,_=load_image(self.tensor[index])
        if sh.shape!=(64,64,6) or tensor.shape!=sh.shape:
            raise ValueError('Expected paired 64x64x6 SH and tensor patches.')
        tensor=tensor*np.array([1000,1000,1000,10000,10000,10000],dtype=np.float32)
        return (torch.from_numpy(sh.transpose(2,0,1).copy()),
                torch.from_numpy(tensor.transpose(2,0,1).copy()))
