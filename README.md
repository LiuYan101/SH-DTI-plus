# SH-DTI++

SH-DTI++ estimates diffusion tensors from second-order spherical harmonic (SH) coefficients using a conditional denoising diffusion model with squeeze-and-excitation (SE) modules.

This repository contains tensor-to-DW simulation, SH preprocessing with optional b-value normalization, paired training-data generation, SH-DTI++ training and full-volume inference. It includes the original 6-, 9-, 15- and 22-direction gradient tables and the MGH-HCP/CHCP subject lists.

## Installation

Use Python 3.10 or later. Install the PyTorch build appropriate for your CUDA environment, then install the remaining requirements:

```bash
pip install -r requirements.txt
```

## Data convention

All NIfTI arrays use `X × Y × Z × channels`. Tensor channels are ordered as
`[Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]` in **mm²/s**. Tensor, S0, DW image and brain mask must already be on the same voxel grid. The code does not register or resample images.

Tensor targets are stored in physical units. The training loader scales the three diagonal channels by 1000 and the off-diagonal channels by 10000. Inference reverses this scaling.

## 1. Simulate DW images from a GT tensor

The signal model is `S(b,g) = S0 exp(-b gᵀDg)`. The simulation uses the supplied tensor directly, without eigenvalue correction, smoothing or optimization.

```bash
python simulate_dw.py \
  --tensor data/MGH_1001/tensor.nii.gz \
  --s0 data/MGH_1001/s0.nii.gz \
  --mask data/MGH_1001/mask.nii.gz \
  --bvals gradients/bval_15.txt --bvecs gradients/bvec_15.txt \
  --output data/MGH_1001/dwi_clean.nii.gz
```

`--s0-volume 0` selects the S0 volume if the source file is 4-D. Optional `--noise-sigma 0.02` adds Rician magnitude noise to all volumes, including b0. Sigma has the same intensity units as S0; the original simulation used normalized S0 images. The default is noise-free simulation.

## 2. SH conversion and b-value normalization

The six SH channels follow `l=0`, then `l=2, m=-2…2`, using the original real-basis and shuffled Gram-Schmidt transform. This transform is retained for checkpoint compatibility; it is not replaced by an ordinary least-squares pseudoinverse.

The DW signal is divided by the first b0 and clipped to `[0,1]`. By default, single-shell signals are mapped to `b_ref=1000 s/mm²`:

`r_ref = clip(S(b)/S0, 0, 1) ** (b_ref/b)`.

This follows the exponential attenuation model. It is applied to the normalized signal before SH conversion. The standard 1000-shell training protocol is unchanged by this step. The implementation expects one nonzero shell and at least six independent directions.

```bash
python prepare_sh.py \
  --dwi data/MGH_1001/dwi_clean.nii.gz \
  --mask data/MGH_1001/mask.nii.gz \
  --bvals gradients/bval_15.txt --bvecs gradients/bvec_15.txt \
  --reference-b 1000 --output data/MGH_1001/sh.nii.gz
```

For acquisition b-values other than 1000, provide the corresponding b-value file. Use `--no-b-normalization` to retain the acquired b-value. Do not apply the normalization again to an already normalized signal.

## 3. Generate paired training data

List the training cases in `examples/train_cases.json`, with the tensor, mask and either S0 or noise-free DW image paths. Paths are relative to the manifest.

Input volumes may have different spatial dimensions. The code automatically extracts overlapping `64 × 64` axial patches, covers the image boundaries and zero-pads inputs smaller than a patch. No fixed volume size or manual crop is required.

```bash
python prepare_training_data.py \
  --cases examples/train_cases.json \
  --bvals gradients/bval_15.txt --bvecs gradients/bvec_15.txt \
  --output data/train_patches
```

The script adds Rician noise using the configured noise levels and saves paired SH/tensor patches in `sh_coe/` and `gt_tensor/`. Patch overlap can be adjusted with `--stride`. Keep subjects separate between training and testing.

## 4. Train SH-DTI++

```bash
python train.py --data data/train_patches \
  --config configs/sh_dti_plus.yaml --output checkpoints/sh_dti_plus
```

The configuration preserves the main model settings: 12 input channels (six SH coefficients and six noisy tensor components), six output noise channels, base width 128, 1000 diffusion steps, batch size 4, Adam learning rate `2e-5`, EMA rate 0.999, and 1,200,000 training steps. Four SE blocks operate on channel groups of sizes 1, 5, 3 and 3. The loss is the per-sample summed squared noise error, averaged over the batch. SH coefficients and scaled tensors receive no additional `[-1,1]` intensity transform.

Resume a trusted local checkpoint with:

```bash
python train.py --data data/train_patches \
  --output checkpoints/sh_dti_plus --resume checkpoints/sh_dti_plus/ckpt.pth
```

Checkpoints retain the original `[model, optimizer, epoch, step, EMA]` structure and parameter names. The SE forward path is active independently of any SH scaling setting.

## 5. Test a complete case

Inference accepts full volumes of different spatial sizes, starting from DW images or precomputed SH coefficients:

```bash
python test.py \
  --dwi data/MGH_1001/dwi_clean.nii.gz \
  --mask data/MGH_1001/mask.nii.gz \
  --bvals gradients/bval_15.txt --bvecs gradients/bvec_15.txt \
  --checkpoint checkpoints/sh_dti_plus/ckpt_1200000.pth \
  --output predictions/MGH_1001
```

Alternatively, replace `--dwi`, `--bvals` and `--bvecs` with `--sh data/MGH_1001/sh.nii.gz`. Precomputed SH maps must already follow the normalization and basis conventions of the model.

The input is automatically divided into overlapping `64 × 64` axial patches, with padding where needed. Predictions are blended and padding is removed to restore the original volume size. Outputs are `tensor.nii.gz`, `md.nii.gz` and `fa.nii.gz`, with the input spatial coordinates preserved. Tensor and MD units are mm²/s; FA is dimensionless. `--batch-size` and `--stride` control inference execution.

## Included cohorts and gradients

- `gradients/`: original tables for 6, 9, 15 and 22 DW directions, each including one leading b0 at b=0 and DW entries at b=1000.
- `data_lists/mgh_hcp.txt`: MGH_1001–MGH_1032.
- `data_lists/chcp.txt`: 3008, 3009, 3010, 3012, 3013, 3016, 3025, 3031, 3036, 3037, 3038, 3040, 3048, 3063, 3120, 3121, 3122, 3132, 3137, 3138.

Subject lists identify the cohorts and do not define a train/test split. Images and pretrained weights are not bundled. Only the SH-DTI++ pipeline is included; comparison methods, ablation experiments and result-editing scripts are outside this repository.

## Acknowledgment

The denoising network and diffusion routines build on the original Fast-DDPM/DDPM implementation lineage. The upstream MIT license is retained in `LICENSE`; see `NOTICE` for attribution.
