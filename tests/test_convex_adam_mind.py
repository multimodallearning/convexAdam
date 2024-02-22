from pathlib import Path

import numpy as np
import SimpleITK as sitk
from helper_functions import (resample_img, resample_moving_to_fixed,
                              rotate_image_around_center_affine,
                              rotate_image_around_center_resample)

from convexAdam.apply_convex import apply_convex
from convexAdam.convex_adam_MIND import convex_adam_pt


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

    # compare with reference
    assert np.allclose(
        sitk.GetArrayFromImage(moving_image_resampled_warped),
        sitk.GetArrayFromImage(moving_image_reference),
    )


def test_convex_adam_translation(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))

    # move moving image
    affine = sitk.AffineTransform(3)
    affine.SetTranslation([10, 10, 10])
    moving_image = sitk.Resample(moving_image, affine)

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
    sitk.WriteImage(moving_image_resampled_warped, str(output_dir / patient_id / f"{subject_id}_t2w_translation_warped.mha"))

    # compare with reference (displacement field should be within 1 mm of the translation for at least 90% of the voxels)
    assert (np.abs(displacementfield + 10) < 1).mean() > 0.9


def test_convex_adam_identity_rotated_direction(
    input_dir = Path("tests/input"),
    output_dir = Path("tests/output"),
    subject_id = "10000_1000000",
):
    # paths
    patient_id = subject_id.split("_")[0]
    fixed_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))
    moving_image = sitk.ReadImage(str(input_dir / patient_id / f"{subject_id}_t2w.mha"))

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

    # apply displacement field to the moving image without resampling
    channels_resampled = []
    for i in range(3):
        displacement_field_channel = sitk.GetImageFromArray(displacementfield[:, :, :, i])
        displacement_field_channel.CopyInformation(fixed_image_resampled)

        # set up the resampling filter
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving_image)
        resampler.SetInterpolator(sitk.sitkLinear)

        # apply resampling
        displacement_field_resampled = resampler.Execute(displacement_field_channel)

        # append to list of channels
        channels_resampled.append(displacement_field_resampled)

    # combine channels
    displacement_field_resampled = sitk.JoinSeries(channels_resampled)
    displacement_field_resampled = np.moveaxis(sitk.GetArrayFromImage(displacement_field_resampled), 0, -1)

    # find the rotation between the direction of the moving image and the direction of the fixed image
    fixed_direction = np.array(fixed_image.GetDirection()).reshape(3, 3)
    moving_direction = np.array(moving_image.GetDirection()).reshape(3, 3)
    rotation = np.dot(np.linalg.inv(fixed_direction), moving_direction)

    # rotate the vectors in the displacement field (the z, y, x components are in the last dimension)
    displacement_field_resampled = displacement_field_resampled[..., ::-1]  # make the order x, y, z
    displacement_field_rotated = np.dot(displacement_field_resampled, rotation)
    displacement_field_rotated = displacement_field_rotated[..., ::-1]  # make the order z, y, x

    # adapt the displacement field to the original moving image, which has a different spacing
    scaling_factor = np.array(fixed_image_resampled.GetSpacing()) / np.array(moving_image.GetSpacing())
    displacement_field_rescaled = displacement_field_rotated * list(scaling_factor)[::-1]

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
