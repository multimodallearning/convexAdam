from pathlib import Path
import torch

import numpy as np
import SimpleITK as sitk
from helper_functions import (rotate_image_around_center_affine,
                              rotate_image_around_center_resample,
                              ssim3D)

from convexAdam.apply_convex import apply_convex
from convexAdam.convex_adam_MIND import convex_adam_pt
from convexAdam.convex_adam_utils import (resample_img,
                                          resample_moving_to_fixed,
                                          rescale_displacement_field)


##For testing
torch.backends.cuda.matmul.allow_tf32 = False 
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

def test_convex_adam_identity(
    input_dir = Path("tests/input"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)

    # run convex adam
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

    # test that the displacement field is performing identity transformation
    assert np.allclose(displacementfield, np.zeros_like(displacementfield), atol=0.1)


def test_convex_adam(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    output_expected_dir = Path("tests/output-expected"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_adc.mha"))
    moving_image_reference = sitk.ReadImage(str(output_expected_dir / patient_id / f"{subject_id}_adc_warped.mha"))
    (output_dir / patient_id).mkdir(exist_ok=True, parents=True)

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)

    # run convex adam
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

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
    sitk.WriteImage(moving_image_resampled_warped, str(output_dir / patient_id / f"{subject_id}_adc_warped.mha"))

    # compare results with SSIM metric
    arr1 = torch.from_numpy(sitk.GetArrayFromImage(moving_image_resampled_warped)[np.newaxis, np.newaxis, ...])
    arr2 = torch.from_numpy(sitk.GetArrayFromImage(moving_image_reference)[np.newaxis, np.newaxis, ...])
    assert ssim3D(arr1, arr2) > 0.95

def test_convex_adam_translation(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    (output_dir / patient_id).mkdir(exist_ok=True, parents=True)

    # set direction to unity (this is important for the test)
    # doing this aligns the image axes with the world axes
    fixed_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    moving_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)

    # move moving image
    affine = sitk.AffineTransform(3)
    affine.SetTranslation([10, 10, 10])
    moving_image_resampled = sitk.Resample(moving_image_resampled, affine)

    # run convex adam
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

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
    sitk.WriteImage(moving_image_resampled_warped, str(output_dir / patient_id / f"{subject_id}_t2w_translation_warped.mha"))

    # compare with reference (displacement field should be within 1 mm of the translation for at least 90% of the voxels in the center)
    s = displacementfield.shape[0] // 10
    displacementfield_center = displacementfield[s:-s, s:-s, s:-s]
    assert (np.abs(displacementfield_center + 10) < 1).mean() > 0.90


def test_convex_adam_identity_rotated_direction(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    (output_dir / patient_id).mkdir(exist_ok=True, parents=True)

    # set center and direction to unity
    moving_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    moving_image.SetOrigin([0, 0, 0])
    fixed_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    fixed_image.SetOrigin([0, 0, 0])
    sitk.WriteImage(fixed_image, str(output_dir / patient_id / f"{subject_id}_fixed_unity.mha"))

    # rotate the moving image twice: once by updating the direction cosines and once by resampling the image
    angle = np.pi / 4.0
    moving_image = rotate_image_around_center_resample(moving_image, angle)
    rotate_image_around_center_affine(moving_image, angle)

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)
    sitk.WriteImage(fixed_image_resampled, str(output_dir / patient_id / f"{subject_id}_fixed_resampled.mha"))

    # run convex adam
    displacementfield = convex_adam_pt(
        img_fixed=fixed_image_resampled,
        img_moving=moving_image_resampled,
    )

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
    sitk.WriteImage(moving_image_resampled_warped, str(output_dir / patient_id / f"{subject_id}_moving_rotation_warped.mha"))

    # test that the displacement field is performing identity transformation
    d1, d2, d3 = np.array(displacementfield.shape[0:3]) // 3
    disp_center = displacementfield[d1:-d1, d2:-d2, d3:-d3]
    assert np.allclose(disp_center, np.zeros_like(disp_center), atol=0.3)


def test_convex_adam_identity_rotated_and_shifted(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    (output_dir / patient_id).mkdir(exist_ok=True, parents=True)

    # set center and direction to unity
    moving_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    moving_image.SetOrigin([0, 0, 0])
    fixed_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    fixed_image.SetOrigin([0, 0, 0])
    sitk.WriteImage(fixed_image, str(output_dir / patient_id / f"{subject_id}_fixed_unity.mha"))

    # rotate the moving image twice: once by updating the direction cosines and once by resampling the image
    angle = np.pi / 4.0
    moving_image = rotate_image_around_center_resample(moving_image, angle)
    rotate_image_around_center_affine(moving_image, angle)

    # translate the moving image
    affine = sitk.AffineTransform(3)
    affine.SetTranslation([20, 0, 0])
    moving_image = sitk.Resample(moving_image, affine)
    sitk.WriteImage(moving_image, str(output_dir / patient_id / f"{subject_id}_moving_rotated_and_shifted.mha"))
    # note: the moving image, when viewed in ITK-SNAP, is now moved 20 mm to the left (patient's right)

    # resample images to specified spacing and the field of view of the fixed image
    fixed_image_resampled = resample_img(fixed_image, spacing=(1.0, 1.0, 1.0))
    moving_image_resampled = resample_moving_to_fixed(fixed_image_resampled, moving_image)
    sitk.WriteImage(fixed_image_resampled, str(output_dir / patient_id / f"{subject_id}_fixed_resampled.mha"))

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
    sitk.WriteImage(moving_image_resampled_warped, str(output_dir / patient_id / f"{subject_id}_moving_rotated_and_shifted_warped.mha"))

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
    sitk.WriteImage(moving_image_warped, str(output_dir / patient_id / f"{subject_id}_original_moving_rotated_and_shifted_warped.mha"))


if __name__ == "__main__":
    test_convex_adam_identity()
    test_convex_adam()
    test_convex_adam_translation()
    test_convex_adam_identity_rotated_direction()
    test_convex_adam_identity_rotated_and_shifted()
    print("All tests passed")
