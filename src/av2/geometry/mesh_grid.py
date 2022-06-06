# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Mesh grid utility functions."""

import math

import numpy as np

from av2.utils.typing import NDArrayFloat


def get_mesh_grid_as_point_cloud(
    min_x: int, max_x: int, min_y: int, max_y: int, downsample_factor: float = 1.0
) -> NDArrayFloat:
    """Sample regular grid and return the (x, y) coordinates.

    For a (2,2) grid, 9 sampled points will be returned (representing the unique corners
    of every grid cell).

    Args:
        min_x: Minimum x-coordinate of 2D grid
        max_x: Maximum x-coordinate of 2D grid
        min_y: Minimum y-coordinate of 2D grid
        max_y: Maximum y-coordinate of 2D grid
        downsample_factor: the resolution of the grid. Defaults to 1 m resolution.
            For example, to sample just the 4 corners of a 3x3 grid (@ 3 meter resolution),
            the downsample_factor should be set to 3.

    Returns:
        Array of shape (N, 2) and type float64 representing 2d (x,y) coordinates.
    """
    nx = max_x - min_x
    ny = max_y - min_y
    x = np.linspace(min_x, max_x, math.ceil(nx / downsample_factor) + 1)
    y = np.linspace(min_y, max_y, math.ceil(ny / downsample_factor) + 1)
    x_grid, y_grid = np.meshgrid(x, y)

    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    x_grid = x_grid[:, np.newaxis]
    y_grid = y_grid[:, np.newaxis]

    pts: NDArrayFloat = np.hstack([x_grid, y_grid])
    return pts
