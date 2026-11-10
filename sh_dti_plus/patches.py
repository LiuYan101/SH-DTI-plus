"""Shared spatial tiling for training and full-volume inference."""
def patch_starts(size,stride,patch_size=64):
    if size<patch_size:
        raise ValueError(f'Input slice dimensions must be at least {patch_size}; padding is not used.')
    if not 0<stride<=patch_size:
        raise ValueError('Use stride 1..patch_size.')
    # Shift the final patch inward so its end coincides with the image boundary.
    return sorted(set(range(0,size-patch_size+1,stride))|{size-patch_size})
