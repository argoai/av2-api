# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Common coordinate system conversions."""


import numpy as np
from scipy.spatial.transform import Rotation

from av2.utils.typing import NDArrayFloat


def cartesian_to_spherical(coordinates_cartesian_m: NDArrayFloat) -> NDArrayFloat:
    """Convert Cartesian coordinates into spherical coordinates.

    This function converts a set of points in R^3 to its spherical representation in R^3.

    NOTE: Ellipses indicate any number of proceeding dimensions allowed for input.

    Args:
        coordinates_cartesian_m: (...,3) Array of points (x,y,z) in Cartesian space.

    Returns:
        (...,3) Array in spherical space. [Order: (azimuth, inclination, radius)].
    """
    coordinates_x_m = coordinates_cartesian_m[..., 0]
    coordinates_y_m = coordinates_cartesian_m[..., 1]
    coordinates_z_m = coordinates_cartesian_m[..., 2]

    hypot_xy = np.hypot(coordinates_x_m, coordinates_y_m)
    coordinates_azimuth = np.arctan2(coordinates_y_m, coordinates_x_m)
    coordinates_inclination = np.arctan2(coordinates_z_m, hypot_xy)
    coordinates_radius = np.hypot(hypot_xy, coordinates_z_m)

    coordinates_spherical: NDArrayFloat = np.stack(
        (coordinates_azimuth, coordinates_inclination, coordinates_radius), axis=-1
    )
    return coordinates_spherical


def spherical_to_cartesian(coordinates_spherical: NDArrayFloat) -> NDArrayFloat:
    """Convert Cartesian coordinates into spherical coordinates.

    This function converts a set of points in R^3 to its spherical representation in R^3.
    NOTE: Ellipses indicate any number of proceeding dimensions allowed for input.

    Args:
        coordinates_spherical: (...,3) Array of points (az,inc,rad) in spherical coordinates.

    Returns:
        (...,3) Array in Cartesian coordinates.
    """
    coordinates_azimuth = coordinates_spherical[..., 0]
    coordinates_inclination = coordinates_spherical[..., 1]
    coordinates_radius = coordinates_spherical[..., 2]

    r_cos_inc = coordinates_radius * np.cos(coordinates_inclination)
    coordinates_x_m = r_cos_inc * np.cos(coordinates_azimuth)
    coordinates_y_m = r_cos_inc * np.sin(coordinates_azimuth)
    coordinates_z_m = coordinates_radius * np.sin(coordinates_inclination)
    coordinates_cartesian: NDArrayFloat = np.stack((coordinates_x_m, coordinates_y_m, coordinates_z_m), axis=-1)
    return coordinates_cartesian


def cartesian_to_homogeneous(coordinates_cartesian_m: NDArrayFloat) -> NDArrayFloat:
    """Convert Cartesian coordinates into Homogenous coordinates.

    This function converts a set of points in R^N to its homogeneous representation in R^(N+1).

    Args:
        coordinates_cartesian_m: (M,N) Array of points in Cartesian space in meters.

    Returns:
        (M,N+1) Array in Homogeneous space.
    """
    num_coordinates, num_dimensions = coordinates_cartesian_m.shape
    coordinates_homogeneous: NDArrayFloat = np.ones((num_coordinates, num_dimensions + 1))
    coordinates_homogeneous[:, :num_dimensions] = coordinates_cartesian_m
    return coordinates_homogeneous


def homogeneous_to_cartesian(coordinates_homogeneous: NDArrayFloat) -> NDArrayFloat:
    """Convert Homogenous coordinates into Cartesian coordinates.

    This function converts a set of points in R^(N+1) to its Cartesian representation in R^N.

    Args:
        coordinates_homogeneous: (M,N+1) Array of points in Homogeneous space.

    Returns:
        (M,N) Array in Cartesian space.
    """
    num_dimensions = coordinates_homogeneous.shape[1] - 1
    coordinates_cartesian: NDArrayFloat = (
        coordinates_homogeneous[:, :num_dimensions] / coordinates_homogeneous[:, num_dimensions : num_dimensions + 1]
    )
    return coordinates_cartesian


def mat_to_xyz(mat: NDArrayFloat) -> NDArrayFloat:
    """Convert a 3D rotation matrix to a sequence of _extrinsic_ rotations.

    In other words, 3D rotation matrix and returns a sequence of Tait-Bryan angles
    representing the transformation.

    Reference: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    Reference: https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles_2

    Args:
        mat: (...,3,3) Rotation matrix.

    Returns:
        (...,3) Tait-Bryan angles (in radians) formulated for a sequence of extrinsic rotations.
    """
    xyz_rad: NDArrayFloat = Rotation.from_matrix(mat).as_euler("xyz", degrees=False)
    return xyz_rad


def xyz_to_mat(xyz_rad: NDArrayFloat) -> NDArrayFloat:
    """Convert a sequence of rotations about the (x,y,z) axes to a 3D rotation matrix.

    In other words, this function takes in a sequence of Tait-Bryan angles and
    returns a 3D rotation matrix which represents the sequence of rotations.

    Computes:
        R = Rz(z) * Ry(y) * Rx(x)

    Reference: https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles_2
    Reference: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    Args:
        xyz_rad: (...,3) Tait-Bryan angles (in radians) of extrinsic rotations.

    Returns:
        (...,3,3) 3D Rotation matrix.
    """
    mat: NDArrayFloat = Rotation.from_euler("xyz", xyz_rad, degrees=False).as_matrix()
    return mat


def xy_to_uv(xy: NDArrayFloat, width: int, height: int) -> NDArrayFloat:
    """Convert coordinates in R^2 (x,y) to texture coordinates (u,v) in R^2.

         (x,y) coordinates             (u,v) coordinates
                      (+y)             (0,0) - - - - - (+u)
                       |                 |
                       |        ->       |
                       |                 |
        (+x) - - - - (0,0)              (+v)

    The xy to uv coordinate transformation is shown above. We model pixel coordinates
    using the uv texture mapping convention.

    NOTE: Ellipses indicate any number of proceeding dimensions allowed for input.

    Args:
        xy: (...,2) array of coordinates in R^2 (x,y).
        width: Texture grid width.
        height: Texture grid height.

    Returns:
        (...,2) array of texture / pixel coordinates.
    """
    x = xy[..., 0]
    y = xy[..., 1]

    u = width - x - 1
    v = height - y - 1
    return np.stack((u, v), axis=-1)


def quat_to_mat(quat_wxyz: NDArrayFloat) -> NDArrayFloat:
    """Convert a quaternion to a 3D rotation matrix.

    NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
        we use the scalar FIRST convention.

    Args:
        quat_wxyz: (...,4) array of quaternions in scalar first order.

    Returns:
        (...,3,3) 3D rotation matrix.
    """
    # Convert quaternion from scalar first to scalar last.
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    mat: NDArrayFloat = Rotation.from_quat(quat_xyzw).as_matrix()
    return mat


def mat_to_quat(mat: NDArrayFloat) -> NDArrayFloat:
    """Convert a 3D rotation matrix to a scalar _first_ quaternion.

    NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
        we use the scalar FIRST convention.

    Args:
        mat: (...,3,3) 3D rotation matrices.

    Returns:
        (...,4) Array of scalar first quaternions.
    """
    # Convert quaternion from scalar first to scalar last.
    quat_xyzw: NDArrayFloat = Rotation.from_matrix(mat).as_quat()
    quat_wxyz: NDArrayFloat = quat_xyzw[..., [3, 0, 1, 2]]
    return quat_wxyz
