# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""N-dimensional grid."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import numpy as np

from av2.geometry.geometry import crop_points
from av2.rendering.color import GRAY_BGR
from av2.rendering.rasterize import align_points_center, draw_points_xy_in_img
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt, NDArrayNumber


@dataclass(frozen=True)
class NDGrid:
    """Models an N-dimensional grid.

    Args:
        min_range_m: (N,) Minimum coordinates (in meters).
        max_range_m: (N,) Maximum coordinates (in meters).
        resolution_m_per_cell: (N,) Ratio of meters to cell in each dimension.
    """

    min_range_m: Tuple[float, ...]
    max_range_m: Tuple[float, ...]
    resolution_m_per_cell: Tuple[float, ...]

    def __post_init__(self) -> None:
        """Check validity of variables.

        Raises:
            ValueError: If the minimum range is greater than the maximum range (in any dimension) or
                the resolution is not positive.
        """
        if not all(x < y for x, y in zip(self.min_range_m, self.max_range_m)):
            raise ValueError("All minimum ranges must be less than their corresponding max ranges!")
        if not all(x > 0 for x in self.resolution_m_per_cell):
            raise ValueError("Resolution per cell must be positive!")

    @cached_property
    def dims(self) -> Tuple[int, ...]:
        """Size of the grid _after_ bucketing."""
        range_m: NDArrayFloat = np.array(self.range_m)
        dims = self.scale_and_quantize_points(range_m)
        return tuple(dims.tolist())

    @cached_property
    def range_m(self) -> Tuple[float, ...]:
        """Size of the grid _before_ bucketing."""
        range_m = np.subtract(self.max_range_m, self.min_range_m)
        return tuple(range_m.tolist())

    def scale_points(self, points: NDArrayNumber) -> NDArrayNumber:
        """Scale points by the (1/`resolution_m_per_cell`).

        Args:
            points: (N,D) list of points.

        Returns:
            (N,D) list of scaled points.
        """
        scaled_points: NDArrayNumber = np.divide(points, self.resolution_m_per_cell)
        return scaled_points

    def quantize_points(self, points: NDArrayNumber) -> NDArrayInt:
        """Quantize the points to integer coordinates.

        Args:
            points: (N,D) Array of points.

        Returns:
            (N,D) The array of quantized points.
        """
        # Add half-bucket offset.
        centered_points = align_points_center(points)
        quantized_points: NDArrayInt = np.floor(centered_points).astype(int)
        return quantized_points

    def scale_and_quantize_points(self, points: NDArrayNumber) -> NDArrayInt:
        """Scale and quantize the points.

        Args:
            points: (N,D) Array of points.

        Returns:
            (N,D) The array of quantized points.
        """
        scaled_points = self.scale_points(points)
        quantized_points: NDArrayInt = self.quantize_points(scaled_points)
        return quantized_points

    def transform_to_grid_coordinates(self, points_m: NDArrayFloat) -> NDArrayInt:
        """Transform points to grid coordinates (in meters).

        Args:
            points_m: (N,D) list of points.

        Returns:
            (N,D) list of quantized grid coordinates.
        """
        offset_m = np.abs(self.min_range_m)

        # points_m * (# grid cells / resolution_m_per_cell) = points_grid
        quantized_points_grid = self.scale_and_quantize_points(points_m + offset_m)
        return quantized_points_grid


@dataclass(frozen=True)
class BEVGrid(NDGrid):
    """Models an 2-dimensional grid in the Bird's-eye view.

    Args:
        min_range_m: (2,) Minimum coordinates in the (x,y) axes (in meters).
        max_range_m: (2,) Minimum coordinates in the (x,y) axes (in meters).
        resolution_m_per_cell: (2,) Bucket size in the (x,y) axes (in meters).
    """

    def points_to_bev_img(
        self, points: NDArrayFloat, color: Tuple[int, int, int] = GRAY_BGR, diameter: int = 2
    ) -> NDArrayByte:
        """Convert a set of points in Cartesian space to a bird's-eye-view image.

        Args:
            points: (N,D) List of points in R^D.
            color: RGB color.
            diameter: Point diameter for the drawn points.

        Returns:
            (H,W,3) RGB image representing a BEV projection onto the xy plane.

        Raises:
            ValueError: If points are less than 2-dimensional.
        """
        D = points.shape[-1]
        if D < 2:
            raise ValueError("Points must be at least 2d!")

        points_xy = points[..., :2].copy()  # Prevent modifying input.
        indices_int = self.transform_to_grid_coordinates(points_xy)
        indices, _ = crop_points(indices_int, lower_bound_inclusive=(0.0, 0.0), upper_bound_exclusive=self.dims)

        # Construct uv coordinates.
        H, W = (self.dims[0], self.dims[1])
        uv = indices[..., :2].astype(int)

        C = len(color)
        shape = (H, W, C)
        img: NDArrayByte = np.zeros(shape, dtype=np.uint8)

        colors: NDArrayByte = np.array([color for _ in range(len(points_xy))], dtype=np.uint8)
        img = draw_points_xy_in_img(img, uv, colors, diameter=diameter)
        return img
