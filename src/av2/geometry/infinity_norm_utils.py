# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Utilities for querying distance to an object by infinity norm."""

import numpy as np

from av2.utils.typing import NDArrayFloat


def has_pts_in_infinity_norm_radius(points: NDArrayFloat, window_center: NDArrayFloat, window_sz: float) -> bool:
    """Check if a map entity has points within a search radius from a single query point.

    Note: Distance is measured by the infinity norm.

    Args:
        points: (N,2) or (N,3) array, representing map entity coordinates.
        window_center: (1,2), or (2,) array, representing query point.
        window_sz: search radius, in meters.

    Returns:
        boolean array of shape (N,) indicating which points lie within the search radius.

    Raises:
        ValueError: If `points` is not in R^2 or R^3.
    """
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"Input points array must have shape (N,2) or (N,3) - received {points.shape}.")

    if points.shape[1] == 3:
        # take only x,y dimensions
        points = points[:, :2]

    if window_center.ndim == 2:
        window_center = window_center.squeeze()
    if window_center.size == 3:
        window_center = window_center[:2]

    # reshape just in case was given column vector
    window_center = window_center.reshape(1, 2)

    dists = np.linalg.norm(points - window_center, ord=np.inf, axis=1)
    return bool(dists.min() < window_sz)
