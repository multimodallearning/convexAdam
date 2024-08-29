import numpy as np
import SimpleITK as sitk


def rotate_image_around_center_affine(image: sitk.Image, angle: float) -> None:
    """
    Rotate the given image around its center by the specified angle.

    Parameters:
        moving_image (sitk.image): The image to be rotated.
        angle (float): The angle of rotation in radians.
    """
    original_origin = np.array(image.GetOrigin())
    image.SetOrigin([0, 0, 0])

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
    image.SetOrigin(new_origin + original_origin)


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
