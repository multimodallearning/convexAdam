from pathlib import Path

import numpy as np
import SimpleITK as sitk
from helper_functions import (rotate_image_around_center_affine,
                              rotate_image_around_center_resample)

from convexAdam.apply_convex import apply_convex, apply_convex_original_moving
from convexAdam.convex_adam_MIND import convex_adam_pt
from convexAdam.convex_adam_utils import (resample_img,
                                          resample_moving_to_fixed,
                                          rescale_displacement_field)


def test_convex_adam_rotated_and_shifted_anisotropic(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    (output_dir / patient_id).mkdir(exist_ok=True, parents=True)

    # translate the moving image
    translation = 20
    affine = sitk.AffineTransform(3)
    affine.SetTranslation([translation, 0, 0])
    moving_image = sitk.Resample(moving_image, affine)

    # rotate the moving image twice: once by updating the direction cosines and once by resampling the image
    angle = np.pi / 4.0
    moving_image = rotate_image_around_center_resample(moving_image, angle)
    rotate_image_around_center_affine(moving_image, angle)
    sitk.WriteImage(moving_image, str(output_dir / patient_id / f"{subject_id}_moving_rotated_and_shifted.mha"))
    # note: the moving image, when viewed in ITK-SNAP, is now moved 20 mm to the left (patient's right)

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)
    sitk.WriteImage(fixed_image_resampled, str(output_dir / patient_id / f"{subject_id}_fixed_resampled.mha"))
    sitk.WriteImage(moving_image_resampled, str(output_dir / patient_id / f"{subject_id}_moving_rotated_and_shifted_resampled.mha"))

    # run convex adam
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

    disp = sitk.GetImageFromArray(displacementfield.astype(np.float32))
    disp.CopyInformation(fixed_image_resampled)
    sitk.WriteImage(disp, str(output_dir / patient_id / f"{subject_id}_displacementfield.mha"))

    # apply displacement field
    moving_image_resampled_warped = apply_convex(
        disp=displacementfield,
        moving=moving_image_resampled,
    )

    # convert to SimpleITK image
    moving_image_resampled_warped = sitk.GetImageFromArray(moving_image_resampled_warped.astype(np.float32))
    moving_image_resampled_warped.CopyInformation(moving_image_resampled)

    # save warped image
    output_dir.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(moving_image_resampled_warped, str(output_dir / patient_id / f"{subject_id}_moving_rotated_and_shifted_resampled_warped.mha"))

    # apply displacement field to the moving image without resampling the moving image
    displacement_field_rescaled = rescale_displacement_field(
        displacement_field=displacementfield,
        moving_image=moving_image,
        fixed_image=fixed_image,
        fixed_image_resampled=fixed_image_resampled,
    )

    moving_image_warped = apply_convex(
        disp=displacement_field_rescaled,
        moving=moving_image,
    )
    moving_image_warped = sitk.GetImageFromArray(moving_image_warped.astype(np.float32))
    moving_image_warped.CopyInformation(moving_image)
    sitk.WriteImage(moving_image_warped, str(output_dir / patient_id / f"{subject_id}_moving_rotated_and_shifted_warped.mha"))


def test_convex_adam_anisotropic(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_adc.mha"))
    (output_dir / patient_id).mkdir(exist_ok=True, parents=True)

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)

    # run convex adam
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

    # apply displacement field to the moving image without resampling the moving image
    moving_image_warped = apply_convex_original_moving(
        disp=displacementfield,
        moving_image_original=moving_image,
        fixed_image_original=fixed_image,
        fixed_image_resampled=fixed_image_resampled,
    )
    sitk.WriteImage(moving_image_warped, str(output_dir / patient_id / f"{subject_id}_moving_warped.mha"))


if __name__ == "__main__":
    test_convex_adam_rotated_and_shifted_anisotropic()
    test_convex_adam_anisotropic()
    print("All tests passed")
