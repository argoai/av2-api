# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Geometric utilities for manipulation point clouds, rigid objects, and vector geometry."""

from typing import Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from av2.utils.constants import PI
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt


def wrap_angles(angles: NDArrayFloat, period: float = PI) -> NDArrayFloat:
    """Map angles (in radians) from domain [-∞, ∞] to [0, π).

    Args:
        angles: (N,) array of angles
        period: Length of the domain.

    Returns:
        Angles (in radians) mapped to the interval [0, π).
    """
    # Map angles to [0, ∞].
    angles = np.abs(angles)

    # Calculate floor division and remainder simultaneously.
    divs, mods = np.divmod(angles, period)

    # Select angles which exceed specified period.
    angle_complement_mask = np.nonzero(divs)

    # Take set complement of `mods` w.r.t. the set [0, π].
    # `mods` must be nonzero, thus the image is the interval [0, π).
    angles[angle_complement_mask] = period - mods[angle_complement_mask]
    return angles


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


def cart_to_sph(xyz: NDArrayFloat) -> NDArrayFloat:
    """Convert Cartesian coordinates into spherical coordinates.

    This function converts a set of points in R^3 to its spherical representation in R^3.

    NOTE: Ellipses indicate any number of proceeding dimensions allowed for input.

    Args:
        xyz: (...,3) Array of points (x,y,z) in Cartesian space.

    Returns:
        (...,3) Array in spherical space. [Order: (azimuth, inclination, radius)].
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    hypot_xy = np.hypot(x, y)
    radius = np.hypot(hypot_xy, z)
    inclination = np.arctan2(z, hypot_xy)
    azimuth = np.arctan2(y, x)

    return np.stack((azimuth, inclination, radius), axis=-1)


def cart_to_hom(cart: NDArrayFloat) -> NDArrayFloat:
    """Convert Cartesian coordinates into Homogenous coordinates.

    This function converts a set of points in R^N to its homogeneous representation in R^(N+1).

    Args:
        cart: (M,N) Array of points in Cartesian space.

    Returns:
        NDArrayFloat: (M,N+1) Array in Homogeneous space.
    """
    M, N = cart.shape
    hom: NDArrayFloat = np.ones((M, N + 1))
    hom[:, :N] = cart
    return hom


def hom_to_cart(hom: NDArrayFloat) -> NDArrayFloat:
    """Convert Homogenous coordinates into Cartesian coordinates.

    This function converts a set of points in R^(N+1) to its Cartesian representation in R^N.

    Args:
        hom: (M,N+1) Array of points in Homogeneous space.

    Returns:
        NDArrayFloat: (M,N) Array in Cartesian space.
    """
    N = hom.shape[1] - 1
    cart: NDArrayFloat = hom[:, :N] / hom[:, N : N + 1]
    return cart


def crop_points(
    points: Union[NDArrayFloat, NDArrayInt],
    lower_bound_inclusive: Tuple[float, ...],
    upper_bound_exclusive: Tuple[float, ...],
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Crop points to a lower and upper boundary.

    NOTE: Ellipses indicate any number of proceeding dimensions allowed for input.

    Args:
        points: (...,n) n-dimensional array of points.
        lower_bound_inclusive: (n,) Coordinates lower bound (inclusive).
        upper_bound_exclusive: (n,) Coordinates upper bound (exclusive).

    Raises:
        ValueError: If dimensions between xyz and the provided bounds don't match.

    Returns:
        (...,n) Tuple of cropped points and the corresponding boolean mask.
    """
    # Gather dimensions.
    n_dim = points.shape[-1]
    lb_dim = len(lower_bound_inclusive)
    ub_dim = len(upper_bound_exclusive)

    # Ensure that the logical operations will broadcast.
    if n_dim != lb_dim or n_dim != ub_dim:
        raise ValueError(f"Dimensions n_dim {n_dim} must match both lb_dim {lb_dim} and ub_dim {ub_dim}")

    # Ensure that the lower bound less than or equal to the upper bound for each dimension.
    if not all(lb < ub for lb, ub in zip(lower_bound_inclusive, upper_bound_exclusive)):
        raise ValueError("Lower bound must be less than or equal to upper bound for each dimension")

    # Lower bound mask.
    lb_mask = np.greater_equal(points, lower_bound_inclusive)

    # Upper bound mask.
    ub_mask = np.less(points, upper_bound_exclusive)

    # Bound mask.
    is_valid_points = np.logical_and(lb_mask, ub_mask).all(axis=-1)
    return points[is_valid_points], is_valid_points


def compute_interior_points_mask(points_xyz: NDArrayFloat, cuboid_vertices: NDArrayFloat) -> NDArrayBool:
    r"""Compute the interior points mask for the cuboid.

    Reference: https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d

            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

    Args:
        points_xyz: (N,3) Array representing a point cloud in Cartesian coordinates (x,y,z).
        cuboid_vertices: (8,3) Array representing 3D cuboid vertices, ordered as shown above. 
   
    Returns:
        (N,) An array of boolean flags indicating whether the points are interior to the cuboid.
    """
    # Get three corners of the cuboid vertices.
    vertices: NDArrayFloat = np.stack((cuboid_vertices[6], cuboid_vertices[3], cuboid_vertices[1]))  # (3,3)

    # Choose reference vertex.
    # vertices and choice of ref_vertex are coupled.
    ref_vertex = cuboid_vertices[2]  # (3,)

    # Compute orthogonal edges of the cuboid.
    uvw = ref_vertex - vertices  # (3,3)

    # Compute signed values which are proportional to the distance from the vector.
    sim_uvw_points = points_xyz @ uvw.transpose()  # (N,3)
    sim_uvw_ref = uvw @ ref_vertex  # (3,)

    # Only care about the diagonal.
    sim_uvw_vertices: NDArrayFloat = np.diag(uvw @ vertices.transpose())  # (3,)

    # Check 6 conditions (2 for each of the 3 orthogonal directions).
    # Refer to the linked reference for additional information.
    constraint_a = np.logical_and(sim_uvw_ref <= sim_uvw_points, sim_uvw_points <= sim_uvw_vertices)
    constraint_b = np.logical_and(sim_uvw_ref >= sim_uvw_points, sim_uvw_points >= sim_uvw_vertices)
    is_interior: NDArrayBool = np.logical_or(constraint_a, constraint_b).all(axis=1)
    return is_interior
