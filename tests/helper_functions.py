import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.autograd import Variable

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


# The following code has been copied/adapted from https://github.com/jinh0park/pytorch-ssim-3D
# Thanks to the author for providing this resource
def gaussian(window_size, sigma):
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    return gauss / gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


