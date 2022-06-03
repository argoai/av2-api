# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Geometric utilities for manipulation point clouds, rigid objects, and vector geometry."""

from typing import Tuple, Union

import numpy as np

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
    sim_uvw_vertices: NDArrayFloat = np.diag(uvw @ vertices.transpose())  # type: ignore # (3,)

    # Check 6 conditions (2 for each of the 3 orthogonal directions).
    # Refer to the linked reference for additional information.
    constraint_a = np.logical_and(sim_uvw_ref <= sim_uvw_points, sim_uvw_points <= sim_uvw_vertices)
    constraint_b = np.logical_and(sim_uvw_ref >= sim_uvw_points, sim_uvw_points >= sim_uvw_vertices)
    is_interior: NDArrayBool = np.logical_or(constraint_a, constraint_b).all(axis=1)
    return is_interior
