from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk

from convexAdam.convex_adam_translation import (
    apply_translation, convex_adam_translation,
    index_translation_to_world_translation)
from convexAdam.convex_adam_utils import resample_moving_to_fixed


def translate_along_image_directions(image: sitk.Image, translation: Iterable[float]):
    """
    Translate an image along its image directions (not physical directions).

    Args:
        image: The image to translate.
        translation (x, y, z): The translation in the image directions (mm).
    """
    # Convert physical translation to index space (voxel units)
    world_translation = index_translation_to_world_translation(translation, direction=image.GetDirection())

    # Create the transformation
    dimension = image.GetDimension()
    transform = sitk.TranslationTransform(dimension, world_translation)

    # Apply the transformation
    resampled_image = sitk.Resample(image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())

    return resampled_image


def test_translation_precision(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    (output_dir / patient_id).mkdir(exist_ok=True, parents=True)

    # move moving image a multiple of the voxel size
    spacing = np.array(moving_image.GetSpacing())
    nvoxels = 5
    translation = spacing * nvoxels
    moving_image = translate_along_image_directions(image=moving_image, translation=translation)
    sitk.WriteImage(moving_image, str(output_dir / patient_id / f"{subject_id}_t2w_translation.mha"))

    # move moving image back
    moving_image = apply_translation(moving_image=moving_image, translation_ijk=-translation)
    sitk.WriteImage(moving_image, str(output_dir / patient_id / f"{subject_id}_t2w_translation_back.mha"))

    # compare images
    moving_image = resample_moving_to_fixed(moving=moving_image, fixed=fixed_image)
    arr_fixed = sitk.GetArrayFromImage(fixed_image)
    arr_moving = sitk.GetArrayFromImage(moving_image)

    # crop (to avoid edge effects from translation)
    arr_fixed = arr_fixed[nvoxels:-nvoxels, nvoxels:-nvoxels, nvoxels:-nvoxels]
    arr_moving = arr_moving[nvoxels:-nvoxels, nvoxels:-nvoxels, nvoxels:-nvoxels]

    np.testing.assert_allclose(
        arr_fixed,
        arr_moving,
        atol=2.0
    )


def test_convex_adam_translation(
    input_dir = Path("tests/input"),
    subject_id = "10000_1000000",
    use_mask: bool = True,
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    if use_mask:
        segmentation = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_prostate_seg.nii.gz"))

    # move moving image
    translation = [10, 10, 0]
    moving_image = translate_along_image_directions(moving_image, translation)

    # apply convex adam translation
    translation_xyz, moving_image, _ = convex_adam_translation(
        fixed_image=fixed_image,
        moving_image=moving_image,
        segmentation=segmentation,
    )

    # check translation
    np.testing.assert_allclose(
        -np.array(translation),
        translation_xyz,
        atol=1.0
    )


if __name__ == "__main__":
    test_translation_precision()
    test_convex_adam_translation()
    print("All tests passed")
