import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def apply_convex(
    disp: np.ndarray,
    moving: np.ndarray,
    header: nib.nifti1.Nifti1Header,
):
    H, W, D, _ = disp.shape
    identity = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    warped = map_coordinates(moving, disp.transpose(3,0,1,2) + identity, order=1)
    warped_image = nib.Nifti1Image(warped, affine=None, header=header)
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
        header=moving.header,
    )

    nib.save(warped_image, args.output_warped)
