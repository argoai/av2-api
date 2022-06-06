# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Polyline related utilities."""

import datetime
import math
from typing import Final, Tuple

import matplotlib.pyplot as plt
import numpy as np

import av2.geometry.interpolate as interp_utils
from av2.utils.typing import NDArrayBool, NDArrayFloat

AVG_LANE_WIDTH_M: Final[float] = 3.8
EPS: Final[float] = 1e-10


def get_polyline_length(polyline: NDArrayFloat) -> float:
    """Calculate the length of a polyline.

    Args:
        polyline: Numpy array of shape (N,3)

    Returns:
        The length of the polyline as a scalar.

    Raises:
        RuntimeError: If `polyline` doesn't have shape (N,2) or (N,3).
    """
    if polyline.shape[1] not in [2, 3]:
        raise RuntimeError("Polyline must have shape (N,2) or (N,3)")
    offsets = np.diff(polyline, axis=0)
    return float(np.linalg.norm(offsets, axis=1).sum())


def interp_polyline_by_fixed_waypt_interval(polyline: NDArrayFloat, waypt_interval: float) -> Tuple[NDArrayFloat, int]:
    """Resample waypoints of a polyline so that waypoints appear roughly at fixed intervals from the start.

    Args:
        polyline: array pf shape (N,2) or (N,3) representing a polyline.
        waypt_interval: space interval between waypoints, in meters.

    Returns:
        interp_polyline: array of shape (N,2) or (N,3) representing a resampled/interpolated polyline.
        num_waypts: number of computed waypoints.

    Raises:
        RuntimeError: If `polyline` doesn't have shape (N,2) or (N,3).
    """
    if polyline.shape[1] not in [2, 3]:
        raise RuntimeError("Polyline must have shape (N,2) or (N,3)")

    # get the total length in meters of the line segment
    len_m = get_polyline_length(polyline)

    # count number of waypoints to get the desired length
    # add one for the extra endpoint
    num_waypts = math.floor(len_m / waypt_interval) + 1
    interp_polyline = interp_utils.interp_arc(t=num_waypts, points=polyline)
    return interp_polyline, num_waypts


def get_double_polylines(polyline: NDArrayFloat, width_scaling_factor: float) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Treat any polyline as a centerline, and extend a narrow strip on both sides.

    Dimension is preserved (2d->2d, and 3d->3d).

    Args:
        polyline: array of shape (N,2) or (N,3) representing a polyline.
        width_scaling_factor: controls the spacing between the two polylines representing the lane boundary,
            e.g. for a "DOUBLE_SOLID" marking.

    Returns:
        left: array of shape (N,2) or (N,3) representing left polyline.
        right: array of shape (N,2) or (N,3) representing right polyline.
    """
    double_line_polygon = centerline_to_polygon(centerline=polyline, width_scaling_factor=width_scaling_factor)
    num_pts = double_line_polygon.shape[0]

    # split index -- polygon from right boundary, left boundary, then close it w/ 0th vertex of right
    # we swap left and right since our polygon is generated about a boundary, not a centerline
    k = num_pts // 2
    left = double_line_polygon[:k]
    right = double_line_polygon[k:-1]  # throw away the last point, since it is just a repeat

    return left, right


def swap_left_and_right(
    condition: NDArrayBool, left_centerline: NDArrayFloat, right_centerline: NDArrayFloat
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Swap points in left and right centerline according to condition.

    Args:
        condition: Numpy array of shape (N,) of type boolean. Where true, swap the values in the left
            and right centerlines.
        left_centerline: The left centerline, whose points should be swapped with the right centerline.
        right_centerline: The right centerline.

    Returns:
        Left and right centerlines.
    """
    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices
    return left_centerline, right_centerline


def centerline_to_polygon(
    centerline: NDArrayFloat, width_scaling_factor: float = 1.0, visualize: bool = False
) -> NDArrayFloat:
    """Convert a lane centerline polyline into a rough polygon of the lane's area.

    The input polyline may be 2d or 3d. Centerline height will be propagated to the two new
    polylines.

    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m on each side.
    We use this as the length of the hypotenuse of a right triangle, and compute the
    other two legs to find the scaled x and y displacement.

    Args:
        centerline: Numpy array of shape (N,3) or (N,2).
        width_scaling_factor: Multiplier that scales avg. lane width to get a new lane width.
        visualize: Save a figure showing the the output polygon.

    Returns:
        Numpy array of shape (2N+1,2) or (2N+1,3), with duplicate first and last vertices.
    """
    # eliminate duplicates
    _, inds = np.unique(centerline, axis=0, return_index=True)
    # does not return indices in sorted order
    inds = np.sort(inds)
    centerline = centerline[inds]

    grad: NDArrayFloat = np.gradient(centerline, axis=0)
    dx = grad[:, 0]
    dy = grad[:, 1]

    # compute the normal at each point
    slopes = dy / (dx + EPS)
    inv_slopes = -1.0 / (slopes + EPS)

    thetas = np.arctan(inv_slopes)
    x_disp = AVG_LANE_WIDTH_M * width_scaling_factor / 2.0 * np.cos(thetas)
    y_disp = AVG_LANE_WIDTH_M * width_scaling_factor / 2.0 * np.sin(thetas)

    displacement: NDArrayFloat = np.hstack([x_disp[:, np.newaxis], y_disp[:, np.newaxis]])

    # preserve z coordinates.
    right_centerline = centerline.copy()
    right_centerline[:, :2] += displacement

    left_centerline = centerline.copy()
    left_centerline[:, :2] -= displacement

    # right centerline position depends on sign of dx and dy
    subtract_cond1 = np.logical_and(dx > 0, dy < 0)
    subtract_cond2 = np.logical_and(dx > 0, dy > 0)
    subtract_cond = np.logical_or(subtract_cond1, subtract_cond2)
    left_centerline, right_centerline = swap_left_and_right(subtract_cond, left_centerline, right_centerline)

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(neg_disp_cond, left_centerline, right_centerline)

    if visualize:
        plt.scatter(centerline[:, 0], centerline[:, 1], 20, marker=".", color="b")
        plt.scatter(right_centerline[:, 0], right_centerline[:, 1], 20, marker=".", color="r")
        plt.scatter(left_centerline[:, 0], left_centerline[:, 1], 20, marker=".", color="g")
        fname = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")
        plt.savefig(f"polygon_unit_tests/{fname}.png")
        plt.close("all")

    # return the polygon
    return convert_lane_boundaries_to_polygon(right_centerline, left_centerline)


def convert_lane_boundaries_to_polygon(right_lane_bounds: NDArrayFloat, left_lane_bounds: NDArrayFloat) -> NDArrayFloat:
    """Convert lane boundaries to a polygon.

    Given left and right boundaries of a lane segment, provide the exterior vertices of the
        2d or 3d lane segment polygon.

    NOTE: We chain the right segment with a reversed left segment, and then repeat the first vertex. In other words,
    the first and last vertex are identical.

    L _________
      .       .
      .       .
    R _________

    Args:
        right_lane_bounds: Array of shape (K,2) or (K,3) representing right lane boundary points.
        left_lane_bounds: Array of shape (M,2) or (M,3) representing left lane boundary points.

    Returns:
        Numpy array of shape (K+M+1,2) or (K+M+1,3)

    Raises:
        RuntimeError: If the last dimension of the left and right boundary polylines do not match.
    """
    if not right_lane_bounds.shape[-1] == left_lane_bounds.shape[-1]:
        raise RuntimeError("Last dimension of left and right boundary polylines must match.")

    polygon: NDArrayFloat = np.vstack([right_lane_bounds, left_lane_bounds[::-1], right_lane_bounds[0]])
    if not polygon.ndim == 2 or polygon.shape[1] not in [2, 3]:
        raise RuntimeError("Polygons must be Nx2 or Nx3 in shape.")
    return polygon
