# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Utilities for interpolating polylines or poses."""

from typing import Final, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from av2.geometry.se3 import SE3
from av2.utils.typing import NDArrayFloat, NDArrayInt

# For a single line segment
NUM_CENTERLINE_INTERP_PTS: Final[int] = 10


def compute_lane_width(left_even_pts: NDArrayFloat, right_even_pts: NDArrayFloat) -> float:
    """Compute the width of a lane, given an explicit left and right boundary.

    Requires an equal number of waypoints on each boundary. For 3d polylines, this incorporates
    the height difference between the left and right polyline into the lane width as a hypotenuse
    of triangle formed by lane width in a flat plane, and the height difference.

    Args:
        left_even_pts: Numpy array of shape (N,2) or (N,3)
        right_even_pts: Numpy array of shape (N,2) or (N,3)

    Raises:
        ValueError: If the shapes of left_even_pts and right_even_pts don't match.

    Returns:
        float representing average width of a lane
    """
    if left_even_pts.shape != right_even_pts.shape:
        raise ValueError(
            f"Shape of left_even_pts {left_even_pts.shape} did not match right_even_pts {right_even_pts.shape}"
        )
    lane_width = float(np.mean(np.linalg.norm(left_even_pts - right_even_pts, axis=1)))
    return lane_width


def compute_mid_pivot_arc(single_pt: NDArrayFloat, arc_pts: NDArrayFloat) -> Tuple[NDArrayFloat, float]:
    """Compute an arc by pivoting around a single point.

    Given a line of points on one boundary, and a single point on the other side,
    produce the middle arc we get by pivoting around the single point.

    Occurs when mapping cul-de-sacs.

    Args:
        single_pt: Numpy array of shape (2,) or (3,) representing a single 2d or 3d coordinate.
        arc_pts: Numpy array of shape (N,2) or (N,3) representing a 2d or 3d polyline.

    Returns:
        centerline_pts: Numpy array of shape (N,3)
        lane_width: average width of the lane.
    """
    num_pts = len(arc_pts)
    # form ladder with equal number of vertices on each side
    single_pt_tiled = np.tile(single_pt, (num_pts, 1))
    # compute midpoint for each rung of the ladder
    centerline_pts = (single_pt_tiled + arc_pts) / 2.0
    lane_width = compute_lane_width(single_pt_tiled, arc_pts)
    return centerline_pts, lane_width


def compute_midpoint_line(
    left_ln_boundary: NDArrayFloat,
    right_ln_boundary: NDArrayFloat,
    num_interp_pts: int = NUM_CENTERLINE_INTERP_PTS,
) -> Tuple[NDArrayFloat, float]:
    """Compute the midpoint line from left and right lane segments.

    Interpolate n points along each lane boundary, and then average the left and right waypoints.

    Note that the number of input waypoints along the left and right boundaries
    can be vastly different -- consider cul-de-sacs, for example.

    Args:
        left_ln_boundary: Numpy array of shape (M,2)
        right_ln_boundary: Numpy array of shape (N,2)
        num_interp_pts: number of midpoints to compute for this lane segment,
            except if it is a cul-de-sac, in which case the number of midpoints
            will be equal to max(M,N).

    Returns:
        centerline_pts: Numpy array of shape (N,2) representing centerline of ladder.

    Raises:
        ValueError: If the left and right lane boundaries aren't a list of 2d or 3d waypoints.
    """
    if left_ln_boundary.ndim != 2 or right_ln_boundary.ndim != 2:
        raise ValueError("Left and right lane boundaries must consist of a sequence of 2d or 3d waypoints.")

    dim = left_ln_boundary.shape[1]
    if dim not in [2, 3]:
        raise ValueError("Left and right lane boundaries must be 2d or 3d.")

    if left_ln_boundary.shape[1] != right_ln_boundary.shape[1]:
        raise ValueError("Left ")

    if len(left_ln_boundary) == 1:
        centerline_pts, lane_width = compute_mid_pivot_arc(single_pt=left_ln_boundary, arc_pts=right_ln_boundary)
        return centerline_pts[:, :2], lane_width

    if len(right_ln_boundary) == 1:
        centerline_pts, lane_width = compute_mid_pivot_arc(single_pt=right_ln_boundary, arc_pts=left_ln_boundary)
        return centerline_pts[:, :2], lane_width

    # fall back to the typical case.
    left_even_pts = interp_arc(num_interp_pts, points=left_ln_boundary)
    right_even_pts = interp_arc(num_interp_pts, points=right_ln_boundary)

    centerline_pts = (left_even_pts + right_even_pts) / 2.0

    lane_width = compute_lane_width(left_even_pts, right_even_pts)
    return centerline_pts, lane_width


def interp_arc(t: int, points: NDArrayFloat) -> NDArrayFloat:
    """Linearly interpolate equally-spaced points along a polyline, either in 2d or 3d.

    We use a chordal parameterization so that interpolated arc-lengths
    will approximate original polyline chord lengths.
        Ref: M. Floater and T. Surazhsky, Parameterization for curve
            interpolation. 2005.
            https://www.mathworks.com/matlabcentral/fileexchange/34874-interparc

    For the 2d case, we remove duplicate consecutive points, since these have zero
    distance and thus cause division by zero in chord length computation.

    Args:
        t: number of points that will be uniformly interpolated and returned
        points: Numpy array of shape (N,2) or (N,3), representing 2d or 3d-coordinates of the arc.

    Returns:
        Numpy array of shape (N,2)

    Raises:
        ValueError: If `points` is not in R^2 or R^3.
    """
    if points.ndim != 2:
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen: NDArrayFloat = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc: NDArrayFloat = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: NDArrayInt = np.digitize(eq_spaced_points, bins=cumarc).astype(int)

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: NDArrayFloat = anchors + offsets

    return points_interp


def linear_interpolation(
    key_timestamps: Tuple[int, int], key_translations: Tuple[NDArrayFloat, NDArrayFloat], query_timestamp: int
) -> NDArrayFloat:
    """Given two 3d positions at specific timestamps, interpolate an intermediate position at a given timestamp.

    Args:
        key_timestamps: pair of integer-valued nanosecond timestamps (representing t0 and t1).
        key_translations: pair of (3,) arrays, representing 3d positions.
        query_timestamp: interpolate the position at this timestamp.

    Returns:
        interpolated translation (3,).

    Raises:
        ValueError: If query_timestamp does not fall within [t0,t1].
    """
    t0, t1 = key_timestamps
    if query_timestamp < t0 or query_timestamp > t1:
        raise ValueError("Query timestamp must be witin the interval [t0,t1].")

    interval = t1 - t0
    t = (query_timestamp - t0) / interval

    vec = key_translations[1] - key_translations[0]
    translation_interp = key_translations[0] + vec * t
    return translation_interp


def interpolate_pose(key_timestamps: Tuple[int, int], key_poses: Tuple[SE3, SE3], query_timestamp: int) -> SE3:
    """Given two SE(3) poses at specific timestamps, interpolate an intermediate pose at a given timestamp.

    Note: we use a straight line interpolation for the translation, while still using interpolate (aka "slerp")
    for the rotational component.

    Other implementations are possible, see:
    https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Pose3.h#L129
    https://github.com/borglab/gtsam/blob/744db328e7ae537e71329e04cc141b3a28b0d6bd/gtsam/base/Lie.h#L327

    Args:
        key_timestamps: list of timestamps, representing timestamps of the keyframes.
        key_poses: list of poses, representing the keyframes.
        query_timestamp: interpolate the pose at this timestamp.

    Returns:
        Inferred SE(3) pose at the query time.

    Raises:
        ValueError: If query_timestamp does not fall within [t0,t1].
    """
    t0, t1 = key_timestamps
    if query_timestamp < t0 or query_timestamp > t1:
        raise ValueError("Query timestamp must be witin the interval [t0,t1].")

    # Setup the fixed keyframe rotations and times
    key_rots = Rotation.from_matrix(np.array([kp.rotation for kp in key_poses]))
    slerp = Slerp(key_timestamps, key_rots)

    # Interpolate the rotations at the given time:
    R_interp = slerp(query_timestamp).as_matrix()

    key_translations = (key_poses[0].translation, key_poses[1].translation)
    t_interp = linear_interpolation(key_timestamps, key_translations=key_translations, query_timestamp=query_timestamp)
    pose_interp = SE3(rotation=R_interp, translation=t_interp)
    return pose_interp
