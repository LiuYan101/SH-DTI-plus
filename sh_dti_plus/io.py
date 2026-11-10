"""NIfTI and configuration helpers."""
from pathlib import Path
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import yaml


def load_image(path):
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f'Non-finite image values: {path}')
    return data, image


def save_image(data, path, reference, affine=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1, 0)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32),
                          reference.affine if affine is None else affine, header)
    nib.save(image, str(path))


def check_geometry(reference, other):
    if reference.shape[:3] != other.shape[:3] or not np.allclose(reference.affine, other.affine, atol=1e-5):
        raise ValueError('Images must already share a voxel grid and affine.')


def load_config(path):
    def convert(value):
        return SimpleNamespace(**{k: convert(v) for k,v in value.items()}) if isinstance(value, dict) else value
    with open(path, encoding='utf-8') as handle:
        return convert(yaml.safe_load(handle))


def load_gradients(bvals_path, bvecs_path):
    bvals = np.asarray(np.loadtxt(bvals_path), dtype=np.float64).reshape(-1)
    bvecs = np.asarray(np.loadtxt(bvecs_path), dtype=np.float64)
    if bvecs.shape == (3, bvals.size):
        bvecs = bvecs.T
    if bvecs.shape != (bvals.size, 3) or not np.isfinite(bvecs).all() or not np.isfinite(bvals).all():
        raise ValueError('Expected N b-values and Nx3 or 3xN b-vectors.')
    if np.any(bvals < 0):
        raise ValueError('b-values must be nonnegative.')
    dw = bvals > 100
    norms = np.linalg.norm(bvecs[dw], axis=1)
    if np.any(norms < 1e-8):
        raise ValueError('A DW gradient has zero length.')
    bvecs = bvecs.copy()
    bvecs[dw] /= norms[:,None]
    bvecs[~dw] = 0
    return bvals, bvecs
