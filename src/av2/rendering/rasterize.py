# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Raster visualization tools."""
from typing import Dict, Final, Tuple

import numpy as np

from av2.geometry.geometry import crop_points
from av2.rendering.ops.draw import draw_points_kernel
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt, NDArrayNumber

GREEN_RGB: Final[NDArrayByte] = np.array([0, 255, 0], np.uint8)
ELECTRIC_LIME_RGB: Final[NDArrayByte] = np.array([192, 255, 0], np.uint8)

AV2_CATEGORY_CMAP: Final[Dict[str, NDArrayByte]] = {
    "REGULAR_VEHICLE": GREEN_RGB,
    "PEDESTRIAN": ELECTRIC_LIME_RGB,
}


def align_points_center(points: NDArrayNumber) -> NDArrayFloat:
    """Align grid coordinates to the center of the grid cells.

    Reference: https://bartwronski.com/2021/02/15/bilinear-down-upsampling-pixel-grids-and-that-half-pixel-offset/

    Args:
        points: Sequence of points.

    Returns:
        The aligned points.
    """
    aligned_points: NDArrayFloat = np.add(points, 0.5)
    return aligned_points


def xyz_to_bev(
    xyz: NDArrayFloat,
    voxel_resolution: Tuple[float, float, float],
    grid_size_m: Tuple[float, float, float],
    cmap: NDArrayFloat,
) -> NDArrayByte:
    """Convert a set of points in Cartesian space to a bird's-eye-view image.

    Args:
        xyz: (N,3) List of points in R^3.
        voxel_resolution: (3,) Number of voxel bins in the (x,y,z) axes.
        grid_size_m: (3,) Size of the grid in the (x,y,z) axes.
        cmap: RGB colormap.

    Returns:
        (H,W,3) RGB image representing a BEV projection onto the xy plane.
    """
    cmap /= 255.0

    # If only xyz are provided, then assume intensity is 1.0.
    # Otherwise, use the provided intensity.
    intensity: NDArrayByte
    if xyz.shape[-1] == 3:
        intensity = np.ones_like(xyz.shape[0], np.uint8)
    else:
        intensity = xyz[..., -1].copy()

    # Grab the Cartesian coordinates (xyz).
    cart = xyz[..., :-1].copy()

    # Move the origin to the center of the image.
    cart += np.divide(grid_size_m, 2)

    # Scale the Cartesian coordinates by the voxel resolution.
    indices: NDArrayInt = (cart / voxel_resolution).astype(int)

    # Compute the voxel grid size.
    voxel_grid_size_m = (
        int(grid_size_m[0] / voxel_resolution[0]),
        int(grid_size_m[1] / voxel_resolution[1]),
        int(grid_size_m[2] / voxel_resolution[2]),
    )

    # Crop point cloud to the region-of-interest.
    lower_bound_inclusive = (0.0, 0.0, 0.0)
    indices_cropped, grid_boundary_reduction = crop_points(
        indices, lower_bound_inclusive=lower_bound_inclusive, upper_bound_exclusive=grid_size_m
    )

    # Filter the indices and intensity values.
    cmap = cmap[grid_boundary_reduction]
    intensity = intensity[grid_boundary_reduction]

    # Create the raster image.
    im_dims = (voxel_grid_size_m[0] + 1, voxel_grid_size_m[1] + 1, cmap.shape[1])
    img: NDArrayByte = np.zeros(im_dims, dtype=np.uint8)

    # Construct uv coordinates.
    uv = indices_cropped[:, :2]

    npoints = len(indices_cropped)
    for i in range(npoints):
        u = uv[i, 0]
        v = uv[i, 1]

        img[u, v, :3] = cmap[i]
        img[u, v, 3:4] += intensity[i]

    # Normalize the intensity.
    img[..., -1] = img[..., -1] / img[..., -1].max()

    # Gamma correction.
    img[..., -1] = np.power(img[..., -1], 0.05)

    # Scale RGB by intensity.
    img[..., :3] *= img[..., -1:]

    # Map RGB in [0, 1] -> [0, 255].
    img[..., :3] = img[..., :3] * 255.0
    im_rgb: NDArrayByte = img[..., :3]
    return im_rgb


def draw_points_xy_in_img(
    img: NDArrayByte,
    points_xy: NDArrayInt,
    colors: NDArrayByte,
    diameter: int = 1,
    alpha: float = 1.0,
    with_anti_alias: bool = False,
    sigma: float = 1.0,
) -> NDArrayByte:
    """Draw a set of points over an image.

    Args:
        img: (H,W,3) Image canvas.
        points_xy: (N,2) Points (x,y) to be drawn.
        colors: (N,3) BGR colors to be drawn.
        diameter: Diameter of a drawn point.
        alpha: Coefficient for alpha blending.
        with_anti_alias: Boolean flag to enable anti-aliasing.
        sigma: Gaussian width for anti-aliasing.

    Returns:
        The (H,W,3) image with points overlaid.

    Raises:
        ValueError: If `points_xy` is not integer valued.
    """
    if not np.issubdtype(points_xy.dtype, np.integer):
        raise ValueError("Circle centers must be integer-valued coordinates.")

    img = draw_points_kernel(
        img=img,
        points_uv=points_xy,
        colors=colors,
        diameter=diameter,
        alpha=alpha,
        with_anti_alias=with_anti_alias,
        sigma=sigma,
    )
    return img
