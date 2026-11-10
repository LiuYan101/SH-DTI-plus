"""Shared spatial tiling for training and full-volume inference."""
import numpy as np


def pad_xy(array,patch_size=64):
    if array.ndim<2 or min(array.shape[:2])<1:
        raise ValueError('Spatial dimensions must be positive.')
    padding=[(0,max(0,patch_size-array.shape[0])),
             (0,max(0,patch_size-array.shape[1]))]+[(0,0)]*(array.ndim-2)
    return np.pad(array,padding)


def patch_starts(size,stride,patch_size=64):
    if size<patch_size or not 0<stride<=patch_size:
        raise ValueError('Pad small inputs first and use stride 1..patch_size.')
    return sorted(set(range(0,size-patch_size+1,stride))|{size-patch_size})
