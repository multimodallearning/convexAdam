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


def apply_convex_original_moving(
    disp: Union[torch.Tensor, np.ndarray, sitk.Image],
    moving_image_original: sitk.Image,
    fixed_image_original: sitk.Image,
    fixed_image_resampled: sitk.Image,
):
    """Apply displacement field to the moving image without resampling the moving image"""
    # convert to numpy, if not already
    disp = validate_image(disp).numpy()

    # resample the displacement field to the physical space of the original moving image
    channels_resampled = []
    for i in range(3):
        displacement_field_channel = sitk.GetImageFromArray(disp[:, :, :, i])
        displacement_field_channel.CopyInformation(fixed_image_resampled)

        # set up the resampling filter
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving_image_original)
        resampler.SetInterpolator(sitk.sitkLinear)

        # apply resampling
        displacement_field_resampled = resampler.Execute(displacement_field_channel)

        # append to list of channels
        channels_resampled.append(displacement_field_resampled)

    # combine channels
    displacement_field_resampled = sitk.JoinSeries(channels_resampled)
    displacement_field_resampled = np.moveaxis(sitk.GetArrayFromImage(displacement_field_resampled), 0, -1)

    # find the rotation between the direction of the moving image and the direction of the fixed image
    fixed_direction = np.array(fixed_image_original.GetDirection()).reshape(3, 3)
    moving_direction = np.array(moving_image_original.GetDirection()).reshape(3, 3)
    rotation = np.dot(np.linalg.inv(fixed_direction), moving_direction)

    # rotate the vectors in the displacement field (the z, y, x components are in the last dimension)
    displacement_field_resampled = displacement_field_resampled[..., ::-1]  # make the order x, y, z
    displacement_field_rotated = np.dot(displacement_field_resampled, rotation)
    displacement_field_rotated = displacement_field_rotated[..., ::-1]  # make the order z, y, x

    # adapt the displacement field to the original moving image, which has a different spacing
    scaling_factor = np.array(fixed_image_resampled.GetSpacing()) / np.array(moving_image_original.GetSpacing())
    displacement_field_rescaled = displacement_field_rotated * list(scaling_factor)[::-1]

    moving_image_warped = apply_convex(
        disp=displacement_field_rescaled,
        moving=moving_image_original,
    )
    moving_image_warped = sitk.GetImageFromArray(moving_image_warped.astype(np.float32))
    moving_image_warped.CopyInformation(moving_image_original)
    return moving_image_warped


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_field", dest="input_field", help="input convex displacement field (.nii.gz) full resolution", default=None, required=True)
    parser.add_argument("--input_moving", dest="input_moving",  help="input moving scan (.nii.gz)", default=None, required=True)
    parser.add_argument("--output_warped", dest="output_warped",  help="output warped scan (.nii.gz)", default=None, required=True)
    args = parser.parse_args()

    moving = nib.load(args.input_moving)
    disp = nib.load(args.input_field)

    warped_image = apply_convex(
        disp=disp.get_fdata().astype('float32'),
        moving=moving.get_fdata().astype('float32'),
    )

    warped_image = nib.Nifti1Image(warped_image, affine=None, header=moving.header)
    nib.save(warped_image, args.output_warped)
