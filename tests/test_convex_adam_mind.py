from pathlib import Path
from typing import Optional

import nibabel
import numpy as np
import SimpleITK as sitk

from convexAdam.apply_convex import apply_convex
from convexAdam.convex_adam_MIND import convex_adam_pt


def resample_img(img: sitk.Image, spacing: tuple[float, float, float]):
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(spacing)
    resample.SetSize([int(sz * spc / new_spc + 0.5) for sz, spc, new_spc in zip(img.GetSize(), img.GetSpacing(), spacing)])
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)  # value for regions without source (zero-padding)
    resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(img)


def resample_moving_to_fixed(fixed: sitk.Image, moving: sitk.Image):
    """Resample moving image to the same grid as the fixed image"""
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(fixed.GetSpacing())
    resample.SetSize(fixed.GetSize())
    resample.SetOutputDirection(fixed.GetDirection())
    resample.SetOutputOrigin(fixed.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)  # value for regions without source (zero-padding)
    resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(moving)


def test_convex_adam_identity(
    images_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(images_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(images_dir / patient_id / f"{subject_id}_t2w.mha"))

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)

    # workaround
    (output_dir / patient_id).mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(fixed_image_resampled, str(output_dir / patient_id / "fixed_image_resampled.nii.gz"))
    sitk.WriteImage(moving_image_resampled, str(output_dir / patient_id / "moving_image_resampled.nii.gz"))

    # run convex adam
    print("Running convex adam")
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

    # test that the displacement field is performing identity transformation
    assert np.allclose(displacementfield, np.zeros_like(displacementfield), atol=0.1)


def test_convex_adam(
    images_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(images_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(images_dir / patient_id / f"{subject_id}_adc.mha"))

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)

    # run convex adam
    print("Running convex adam")
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

    # apply displacement field
    print("Applying displacement field")
    moving_image_resampled_warped = apply_convex(
        disp=displacementfield,
        moving=moving_image_resampled,
    )

    # convert to SimpleITK image
    moving_image_resampled_warped = sitk.GetImageFromArray(moving_image_resampled_warped)
    moving_image_resampled_warped.CopyInformation(moving_image_resampled)

    # save warped image
    print("Saving warped image")
    output_dir.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(moving_image_resampled_warped, str(output_dir / f"{subject_id}_adc_warped.mha"))


if __name__ == "__main__":
    # test_convex_adam_identity()
    test_convex_adam()
