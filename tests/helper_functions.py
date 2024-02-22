from pathlib import Path

import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

from convexAdam.apply_convex import apply_convex
from convexAdam.convex_adam_MIND import convex_adam_pt


def resample_img(img: sitk.Image, spacing: tuple[float, float, float]) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(spacing)
    resample.SetSize([int(sz * spc / new_spc + 0.5) for sz, spc, new_spc in zip(img.GetSize(), img.GetSpacing(), spacing)])
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)  # value for regions without source (zero-padding)
    resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(img)


def resample_moving_to_fixed(fixed: sitk.Image, moving: sitk.Image) -> sitk.Image:
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


def rotate_image_around_center_affine(image: sitk.Image, angle: float) -> None:
    """
    Rotate the given image around its center by the specified angle.

    Parameters:
        moving_image (sitk.image): The image to be rotated.
        angle (float): The angle of rotation in radians.
    """
    # Calculate the physical center of the image
    physical_center = image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0)

    # For a 3D image rotation around the z-axis, the rotation matrix is:
    direction = image.GetDirection()
    axis_angle = (direction[2], direction[5], direction[8], angle)
    rotation_matrix = matrix_from_axis_angle(axis_angle)

    # Compute the new origin after rotation
    new_origin = np.dot(rotation_matrix, -np.array(physical_center)) + np.array(physical_center)

    # Get the current direction of the image
    direction = np.array(image.GetDirection()).reshape((3, 3))

    # Compute the new direction cosines by multiplying the current direction by the rotation matrix
    new_direction_cosines = np.dot(rotation_matrix, direction)

    # Update the image with the new direction and origin
    image.SetDirection(new_direction_cosines.flatten())
    image.SetOrigin(new_origin)


def rotate_image_around_center_resample(image: sitk.Image, angle: float) -> sitk.Image:
    """
    Rotate the given image around its center by the specified angle. The rotation is around the z-axis.

    Parameters:
        moving_image (sitk.image): The image to be rotated.
        angle (float): The angle of rotation in radians.
    """
    scale_factor = 1.0
    translation = (0, 0, 0)
    rotation_center = image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0)
    direction = image.GetDirection()
    axis = (direction[2], direction[5], direction[8])

    # rotate moving image
    similarity_transform = sitk.Similarity3DTransform(
        scale_factor, axis, angle, translation, rotation_center
    )

    image = sitk.Resample(image, similarity_transform)

    return image


# This function is from https://github.com/rock-learning/pytransform3d/blob/7589e083a50597a75b12d745ebacaa7cc056cfbd/pytransform3d/rotations.py#L302
def matrix_from_axis_angle(a):
    """ Compute rotation matrix from axis-angle.
    This is called exponential map or Rodrigues' formula.
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(theta) +
    #      (1.0 - np.cos(theta)) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(theta))

    return R

def resample(image, transform):
    """
    This function resamples (updates) an image using a specified transform
    :param image: The sitk image we are trying to transform
    :param transform: An sitk transform (ex. resizing, rotation, etc.
    :return: The transformed sitk image
    """
    reference_image = image
    interpolator = sitk.sitkLinear
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2)),
                                              int(np.ceil(depth/2))))


def rotation3d(image, theta_z, show=False):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :param image: An sitk MRI image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param show: Boolean, whether or not the user wants to see the result of the rotation
    :return: The rotated image
    """
    theta_z = np.deg2rad(theta_z)
    euler_transform = sitk.Euler3DTransform()
    print(euler_transform.GetMatrix())
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)

    direction = image.GetDirection()
    axis_angle = (direction[2], direction[5], direction[8], theta_z)
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    print(np_rot_mat)
    euler_transform.SetMatrix(np_rot_mat.flatten().tolist())
    resampled_image = resample(image, euler_transform)
    if show:
        slice_num = 15#int(input("Enter the index of the slice you would like to see"))
        plt.imshow(sitk.GetArrayFromImage(resampled_image)[slice_num], cmap='gray')
        plt.show()
    return resampled_image
