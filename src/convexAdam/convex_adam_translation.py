import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import SimpleITK as sitk

from convexAdam.convex_adam_MIND import convex_adam_pt
from convexAdam.convex_adam_utils import resample_img, resample_moving_to_fixed


def index_translation_to_world_translation(
    index_translation: Iterable[float],
    direction: Iterable[float]
) -> np.ndarray:
    """
    Convert a translation along the image grid to a translation in world coordinates.

    Args:
        index_translation (i, j, k): The translation in index coordinates (mm).
        direction: The direction of the image.
    Returns:
        world_translation (x, y, z): The translation in world coordinates (mm).
    """
    dimension = int(np.sqrt(len(direction)))
    direction_matrix = np.array(direction).reshape((dimension, dimension))
    index_translation = direction_matrix @ np.array(index_translation)
    return index_translation


def apply_translation(
    moving_image: sitk.Image,
    translation_ijk: Iterable[float] = (0, 0, 0),
) -> sitk.Image:
    """
    Apply a translation to an image, with the translation in mm along the image grid.

    Args:
        moving_image: The image to translate.
        translation_ijk: The translation in mm along the image grid.

    Returns:
        The translated image.
    """
    # copy image
    moving_image = sitk.Image(moving_image)

    # apply translation to moving image
    translation_xyz = index_translation_to_world_translation(translation_ijk, moving_image.GetDirection()[0:9])
    origin = list(moving_image.GetOrigin())
    origin[0:3] -= translation_xyz
    moving_image.SetOrigin(tuple(origin))

    return moving_image


def convex_adam_translation(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    segmentation: Optional[sitk.Image] = None,
    co_moving_images: Optional[Iterable[sitk.Image]] = None,
) -> tuple[tuple[float], sitk.Image, Optional[Iterable[sitk.Image]]]:
    """
    Apply convex Adam translation to an image.

    Args:
        fixed_image: The fixed image.
        moving_image: The moving image.
        segmentation: The segmentation.
        co_moving_images: The co-moving images.

    Returns:
        translation_xyz: The translation in mm.
        moving_image: The moved image.
        co_moving_images: The moved co-moving images.
    """

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)

    # run convex adam
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

    # convert displacement field to translation only
    if segmentation is not None:
        # resample segmentation to the same spacing as the displacement field
        segmentation = resample_moving_to_fixed(moving=segmentation, fixed=fixed_image_resampled)
        seg_arr = sitk.GetArrayFromImage(segmentation)
        seg_arr = (seg_arr > 0)  # above resampling is with linear interpolation, so we need to threshold
        translation_zyx = np.mean(displacementfield[seg_arr], axis=0)
    else:
        translation_zyx = np.mean(displacementfield, axis=(0, 1, 2))

    # transform translation into the number of pixels to move in image space
    spacing_zyx = np.array(list(moving_image.GetSpacing())[::-1])
    translation_ijk = translation_zyx / spacing_zyx
    translation_ijk_voxels = np.round(translation_ijk, decimals=0)
    translation_ijk_mm = translation_ijk_voxels * spacing_zyx
    translation_xyz = tuple(list(translation_ijk_mm[::-1]))

    # apply translation to moving image
    moving_image = apply_translation(moving_image=moving_image, translation_ijk=translation_xyz)

    # apply translation to co-moving images
    if co_moving_images is not None:
        for i, co_moving_image in enumerate(co_moving_images):
            co_moving_image = apply_translation(moving_image=co_moving_image, translation_ijk=translation_xyz)
            co_moving_images[i] = co_moving_image

    return translation_xyz, moving_image, co_moving_images


def convex_adam_translation_from_file(
    fixed_path: Path = Path("/input/fixed.mha"),
    moving_path: Path = Path("/input/moving.mha"),
    segmentation_path: Optional[Path] = Path("/input/segmentation.nii.gz"),
    moving_output_path: Path = Path("/output/moving_warped.mha"),
    co_moving_paths: Optional[Iterable[Path]] = None,
    co_moving_output_paths: Optional[Iterable[Path]] = None,
):
    # paths
    fixed_image = sitk.ReadImage(str(fixed_path))
    moving_image = sitk.ReadImage(str(moving_path))
    segmentation = sitk.ReadImage(str(segmentation_path)) if segmentation_path is not None else None

    translation_xyz, moving_image, co_moving_images = convex_adam_translation(
        fixed_image=fixed_image,
        moving_image=moving_image,
        segmentation=segmentation,
        co_moving_images=[sitk.ReadImage(str(path)) for path in co_moving_paths] if co_moving_paths is not None else None,
    )

    # save moved image
    sitk.WriteImage(moving_image, str(moving_output_path))

    # save co-moving images
    if co_moving_images is not None:
        for co_moving_image, co_moving_output_path in zip(co_moving_images, co_moving_output_paths):
            sitk.WriteImage(co_moving_image, str(co_moving_output_path))

    return translation_xyz


if __name__ == "__main__":
    # command line interface
    parser = argparse.ArgumentParser(description="Apply convex Adam translation to an image.")
    parser.add_argument("--fixed_path", type=Path, help="Path to the fixed image.")
    parser.add_argument("--moving_path", type=Path, help="Path to the moving image.")
    parser.add_argument("--segmentation_path", type=Path, help="Path to the segmentation.")
    parser.add_argument("--moving_output_path", type=Path, help="Path to the output moving image.")
    parser.add_argument("--co_moving_paths", type=Path, nargs="+", help="Paths to the co-moving images.")
    parser.add_argument("--co_moving_output_paths", type=Path, nargs="+", help="Paths to the output co-moving images.")
    args = parser.parse_args()

    convex_adam_translation_from_file(
        fixed_path=args.fixed_path,
        moving_path=args.moving_path,
        segmentation_path=args.segmentation_path,
        moving_output_path=args.moving_output_path,
        co_moving_paths=args.co_moving_paths,
        co_moving_output_paths=args.co_moving_output_paths,
    )
