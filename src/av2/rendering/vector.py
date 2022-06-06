# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Utilities for vector graphics rendering."""

from enum import Enum
from typing import Optional, Tuple, Union

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MPath

from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.rendering.color import BLUE_BGR
from av2.rendering.ops.draw import clip_line_frustum
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt, NDArrayNumber


def draw_polygon_mpl(
    ax: plt.Axes,
    polygon: NDArrayFloat,
    color: Union[Tuple[float, float, float], str],
    linewidth: Optional[float] = None,
) -> None:
    """Draw a polygon's boundary.

    The polygon's first and last point must be the same (repeated).

    Args:
        ax: Matplotlib axes instance to draw on
        polygon: Array of shape (N, 2) or (N, 3)
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
        linewidth: Width of the lines.
    """
    if linewidth is None:
        ax.plot(polygon[:, 0], polygon[:, 1], color=color)
    else:
        ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=linewidth)


def plot_polygon_patch_mpl(
    polygon_pts: NDArrayFloat,
    ax: plt.Axes,
    color: Union[Tuple[float, float, float], str] = "y",
    alpha: float = 0.3,
    zorder: int = 1,
) -> None:
    """Plot a lane segment polyon using matplotlib's PathPatch object.

    Reference:
    See Descartes (https://github.com/benjimin/descartes/blob/master/descartes/patch.py)
    and Matplotlib: https://matplotlib.org/stable/gallery/shapes_and_collections/path_patch.html

    Args:
        polygon_pts: Array of shape (N, 2) representing the points of the polygon
        ax: Matplotlib axes.
        color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'.
        alpha: the opacity of the lane segment.
        zorder: Ordering of the plot overlays.
    """
    n, _ = polygon_pts.shape
    codes = np.ones(n, dtype=MPath.code_type) * MPath.LINETO
    codes[0] = MPath.MOVETO

    vertices = polygon_pts[:, :2]
    mpath = MPath(vertices, codes)
    patch = mpatches.PathPatch(mpath, facecolor=color, edgecolor=color, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)


def draw_line_in_img(
    img: NDArrayByte,
    p1: NDArrayNumber,
    p2: NDArrayNumber,
    color: Tuple[int, int, int] = BLUE_BGR,
    thickness: int = 3,
    line_type: Enum = cv2.LINE_AA,
) -> NDArrayByte:
    """Draw a line on an image.

    Args:
        img: (H,W,3) 3-channel image.
        p1: (2,) Starting point of the line segment.
        p2: (2,) Ending point of the line segment.
        color: 3-channel color corresponding to the channel order of the input image.
        thickness: Thickness of the line to be rendered.
        line_type: OpenCV line style.

    Returns:
        The image with a line drawn.
    """
    img_line: NDArrayByte = cv2.line(
        img,
        p1,
        p2,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )
    return img_line


def draw_line_frustum(
    img: NDArrayByte,
    p1: NDArrayNumber,
    p2: NDArrayNumber,
    cam_model: PinholeCamera,
    color: Tuple[int, int, int],
    thickness: int = 3,
    line_type: Enum = cv2.LINE_AA,
) -> NDArrayByte:
    """Draw a line inside of the camera frustum.

    Args:
        img: (H,W,3) 3-channel image.
        p1: (3,) Starting point of the line segment.
        p2: (3,) Ending points of the line segment.
        cam_model: Camera model that contains intrinsics and the size of the pixel coordinate frame.
        color: 3-channel color corresponding to the channel order of the input image.
        thickness: Thickness of the line to be rendered.
        line_type: OpenCV line style.

    Returns:
        The image with a line clipped and drawn inside the frustum (if it's visible).
    """
    clipped_points = clip_line_frustum(p1, p2, cam_model.frustum_planes())
    intersects_with_frustum = not bool(np.isnan(clipped_points).any())
    if intersects_with_frustum:
        # After clipping the line segment to the view frustum in the camera coordinate frame, we obtain
        # the final set of line segment vertices, and then project these into the image.
        uv, _, _ = cam_model.project_cam_to_img(clipped_points)
        uv_int: NDArrayInt = np.round(uv).astype(int)
        p1, p2 = uv_int[0], uv_int[1]
        img = draw_line_in_img(img, p1, p2, color=color, thickness=thickness, line_type=line_type)
    return img
