# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Provides utilities for working with and visualizing ego-view depth maps."""

from typing import Final

import cv2
import matplotlib.pyplot as plt
import numpy as np

from av2.utils.typing import NDArrayByte, NDArrayFloat

MIN_DISTANCE_AWAY_M: Final[float] = 30.0  # assume max noise starting at this distance (meters)
MAX_ALLOWED_NOISE_M: Final[float] = 3.0  # meters


def compute_allowed_noise_per_point(points_cam: NDArrayFloat) -> NDArrayFloat:
    """Compute allowed noise values when comparing depth maps.

    For example, when rendering only objects that are visible within a frame, one can compare
    z-order against a known depth map. However, depth maps can be noisy, and we wish to be
    robust to this. Accordingly, we assume noise is proportional to distance away from the camera,
    and thus we compensate for noisier depth values at range.

    Args:
        points_cam: array of shape (N,3) representing 3d points in the camera coordinate frame.
            We extract z-values from the camera frame coordinates.

    Returns:
        array of shape (N,) representing allowed amount of noise, in meters, for entries in a depth map.
    """
    dists_away: NDArrayFloat = np.linalg.norm(points_cam, axis=1)

    max_dist_away = dists_away.max()
    max_dist_away = max(max_dist_away, MIN_DISTANCE_AWAY_M)

    allowed_noise: NDArrayFloat = (dists_away / max_dist_away) * MAX_ALLOWED_NOISE_M
    return allowed_noise


def vis_depth_map(
    img_rgb: NDArrayByte, depth_map: NDArrayFloat, interp_depth_map: bool, num_dilation_iters: int = 10
) -> None:
    """Visualize a depth map using Matplotlib's `inferno` colormap side by side with an RGB image.

    Args:
        img_rgb: array of shape (H,W,3) representing an RGB image.
        depth_map: array of shape (H,W) representing a depth map. Only values in [0,255] will be visible.
        interp_depth_map: whether the depth map represents a densely interpolated grid. If not, the zero values
            are not meaningful, and we set them to a maximum value, so that matplotlib will ignore them.
        num_dilation_iters: number of iterations to use for dilating with box kernel, to make sparse value
            more visible.
    """
    if not interp_depth_map:
        # fix the zero values?
        depth_map[depth_map == 0] = np.finfo(np.float32).max

    # purely for visualization
    depth_map = cv2.dilate(
        depth_map,
        kernel=np.ones((2, 2), np.uint8),
        iterations=num_dilation_iters,
    )

    # prevent too dark in the foreground, clip far away
    depth_map = np.clip(depth_map, 0, 50)

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.subplot(1, 2, 2)
    plt.imshow((depth_map * 3).astype(np.uint8), cmap="inferno")
