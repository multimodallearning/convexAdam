from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
from helper_functions import resample_img, resample_moving_to_fixed

from convexAdam.convex_adam_MIND import convex_adam_pt
from convexAdam.convex_adam_utils import rescale_displacement_field


def world_translation_to_index_translation(world_translation: Iterable[float], direction: Iterable[float]):
    """
    Convert a translation in world coordinates to a translation in index coordinates.

    Args:
        world_translation (x, y, z): The translation in world coordinates (mm).
        direction: The direction of the image.
    Returns:
        index_translation (x, y, z): The translation in index coordinates (mm).
    """
    dimension = int(np.sqrt(len(direction)))
    direction_matrix = np.array(direction).reshape((dimension, dimension))
    index_translation = direction_matrix @ np.array(world_translation)
    return index_translation


def index_translation_to_world_translation(index_translation: Iterable[float], direction: Iterable[float]):
    """
    Convert a translation in index coordinates to a translation in world coordinates.
    NOT YET TESTED

    Args:
        index_translation (x, y, z): The translation in index coordinates (mm).
        direction: The direction of the image.
    Returns:
        world_translation (x, y, z): The translation in world coordinates (mm).
    """
    dimension = int(np.sqrt(len(direction)))
    direction_matrix = np.array(direction).reshape((dimension, dimension))
    world_translation = np.linalg.inv(direction_matrix) @ np.array(index_translation)
    return world_translation


def translate_along_image_directions(image: sitk.Image, translation: Iterable[float]):
    """
    Translate an image along its image directions (not physical directions).

    Args:
        image: The image to translate.
        translation (x, y, z): The translation in the image directions (mm).
    """
    # Convert physical translation to index space (voxel units)
    index_translation = world_translation_to_index_translation(translation, direction=image.GetDirection())

    # Create the transformation
    dimension = image.GetDimension()
    transform = sitk.TranslationTransform(dimension, index_translation)

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

    # move moving image a multiple of the voxel size
    spacing = np.array(moving_image.GetSpacing())
    translation = spacing * 5
    moving_image = translate_along_image_directions(moving_image, translation)
    sitk.WriteImage(moving_image, str(output_dir / patient_id / f"{subject_id}_t2w_translation.mha"))

    # move moving image back
    moving_image = translate_along_image_directions(moving_image, -translation)
    sitk.WriteImage(moving_image, str(output_dir / patient_id / f"{subject_id}_t2w_translation_back.mha"))

    # compare images
    arr_fixed = sitk.GetArrayFromImage(fixed_image)
    arr_moving = sitk.GetArrayFromImage(moving_image)

    # crop (to avoid edge effects from translation)
    crop = 5
    arr_fixed = arr_fixed[crop:-crop, crop:-crop, crop:-crop]
    arr_moving = arr_moving[crop:-crop, crop:-crop, crop:-crop]

    np.testing.assert_allclose(
        arr_fixed,
        arr_moving,
        atol=2.0
    )



def test_convex_adam_translation(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
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
    moving_image = translate_along_image_directions(moving_image, [10, 10, 0])
    sitk.WriteImage(moving_image, str(output_dir / patient_id / f"{subject_id}_t2w_translation.mha"))

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)

    # run convex adam
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

    if use_mask:
        # resample segmentation to the same spacing as the displacement field
        segmentation = resample_moving_to_fixed(moving=segmentation, fixed=fixed_image_resampled)
        seg_arr = sitk.GetArrayFromImage(segmentation)
        seg_arr = (seg_arr > 0)  # above resampling is with linear interpolation, so we need to threshold

    # convert displacement field to translation only
    spacing_zyx = np.array(list(moving_image.GetSpacing())[::-1])
    if use_mask:
        translation_zyx = np.mean(displacementfield[seg_arr], axis=0)
    else:
        translation_zyx = np.mean(displacementfield, axis=(0, 1, 2))
    translation_ijk = translation_zyx / spacing_zyx
    translation_ijk_voxels = np.round(translation_ijk, decimals=0)
    translation_ijk_mm = translation_ijk_voxels * spacing_zyx
    translation_xyz = tuple(list(translation_ijk_mm[::-1]))

    # apply translation to moving image
    moving_image = translate_along_image_directions(moving_image, translation_xyz)

    # save moved image
    sitk.WriteImage(moving_image, str(output_dir / patient_id / f"{subject_id}_t2w_translation_warped.mha"))


if __name__ == "__main__":
    test_translation_precision()
    test_convex_adam_translation()
    print("All tests passed")
