import argparse
from typing import Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates

from convexAdam.convex_adam_utils import validate_image


def apply_convex(
    disp: Union[torch.Tensor, np.ndarray, sitk.Image],
    moving: Union[torch.Tensor, np.ndarray, sitk.Image],
) -> np.ndarray:
    # convert to numpy, if not already
    moving = validate_image(moving).numpy()
    disp = validate_image(disp).numpy()

    d1, d2, d3, _ = disp.shape
    identity = np.meshgrid(np.arange(d1), np.arange(d2), np.arange(d3), indexing='ij')
    warped_image = map_coordinates(moving, disp.transpose(3, 0, 1, 2) + identity, order=1)
    return warped_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_field", dest="input_field", help="input convex displacement field (.nii.gz) full resolution", default=None, required=True)
    parser.add_argument("--input_moving", dest="input_moving",  help="input moving scan (.nii.gz)", default=None, required=True)
    parser.add_argument("--output_warped", dest="output_warped",  help="output waroed scan (.nii.gz)", default=None, required=True)
    args = parser.parse_args()

    moving = nib.load(args.input_moving)
    disp = nib.load(args.input_field)

    warped_image = apply_convex(
        disp=disp.get_fdata().astype('float32'),
        moving=moving.get_fdata().astype('float32'),
    )

    warped_image = nib.Nifti1Image(warped_image, affine=None, header=moving.header)
    nib.save(warped_image, args.output_warped)
