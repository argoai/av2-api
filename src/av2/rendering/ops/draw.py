# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Kernels for fast drawing operations."""

from math import exp
from typing import Final, Tuple

import numba as nb
import numpy as np

from av2.utils.constants import NAN
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt, NDArrayNumber

UINT8_MAX: Final[np.uint8] = np.uint8(np.iinfo(np.uint8).max)
UINT8_BITS: Final[np.uint8] = np.log2(UINT8_MAX + 1).astype(np.uint8)


@nb.njit
def integer_linear_interpolation(x: np.uint8, y: np.uint8, alpha: np.uint16, beta: np.uint16) -> np.uint8:
    """Approximate floating point linear interpolation.

    This function approximates the following:
        x * alpha + y * beta,
        s.t. alpha + beta = 1, alpha >=0, beta >= 0.

    Args:
        x: 8-bit unsigned integer value to be weighted by alpha.
        y: 8-bit unsigned integer value to be weighted by beta.
        alpha: 16-bit unsigned integer value in the interval [0, 256].
        beta: 16-bit unsigned integer value in the interval [0, 256].

    NOTE: Alpha and beta are 16 bit integers to properly interpolate
    values at beta=0. Unsigned 16-bit integers can express 2**16 - 1
    values; _HOWEVER_, this function expects alpha and beta to be within
    the interval [0, 256].

    Returns:
        The linear interpolant.
    """
    # Cast to np.uint8 since python bit-shift operator returns a signed integer.
    linear_interpolant = np.uint8((x * alpha + y * beta) >> UINT8_BITS)
    return linear_interpolant


@nb.njit
def alpha_blend_kernel(fg: NDArrayByte, bg: NDArrayByte, alpha: np.uint8) -> Tuple[np.uint8, np.uint8, np.uint8]:
    """Fast integer alpha blending.

    Reference: https://stackoverflow.com/a/12016968

    Args:
        fg: (3,) Foreground RGB or BGR pixel intensities between [0, 255].
        bg: (3,) Background RGB or BGR pixel intensities between [0, 255].
        alpha: Alpha blending coefficient between [0, 255].

    Returns:
        (3,) The blended pixel.
    """
    r1, g1, b1 = fg[0], fg[1], fg[2]
    r2, g2, b2 = bg[0], bg[1], bg[2]

    alpha_uint16 = np.uint16(alpha + 1)
    beta_uint16 = np.uint16(UINT8_MAX + 1 - alpha)

    # Approximate floating point linear interpolation.
    r = integer_linear_interpolation(r1, r2, alpha_uint16, beta_uint16)
    g = integer_linear_interpolation(g1, g2, alpha_uint16, beta_uint16)
    b = integer_linear_interpolation(b1, b2, alpha_uint16, beta_uint16)
    return r, g, b


@nb.njit
def gaussian_kernel(x: float, mu: float, sigma: float) -> float:
    """Return a univariate Gaussian kernel.

    Args:
        x: Value to evaluate the Gaussian.
        mu: Mean parameter of the Gaussian.
        sigma: Width of the Gaussian.

    Returns:
        The Gaussian kernel, N(mu,sigma^2), evaluated at x.
    """
    return exp(-0.5 * (x - mu) ** 2 / (sigma**2))


@nb.njit
def draw_points_kernel(
    img: NDArrayByte,
    points_uv: NDArrayInt,
    colors: NDArrayByte,
    diameter: int = 1,
    alpha: float = 1.0,
    with_anti_alias: bool = False,
    sigma: float = 1.0,
) -> NDArrayByte:
    """Draw points onto an image canvas.

    NOTE: Anti-aliasing will apply a Gaussian kernel at each uv -- parameterized
            as N(uv,sigma^2).

    Args:
        img: (H,W,3) Image canvas.
        points_uv: (N,2) Points (u,v) to be drawn.
        colors: (N,3) BGR colors to be drawn.
        diameter: Diameter of a drawn point.
        alpha: Coefficient for alpha blending.
        with_anti_alias: Boolean flag to enable anti-aliasing.
        sigma: Gaussian width for anti-aliasing.

    Returns:
        (M,N,3) Image with points overlaid.
    """
    H = img.shape[0]
    W = img.shape[1]
    N = len(points_uv)

    left_radius = diameter // 2
    right_radius = diameter - left_radius
    for i in nb.prange(N):
        u = points_uv[i, 0]
        v = points_uv[i, 1]
        for j in nb.prange(-left_radius, right_radius):
            for k in nb.prange(-left_radius, right_radius):
                vj = v + j
                uk = u + k
                if (vj >= 0 and uk >= 0) and (vj < H and uk < W):
                    alpha_vj_uk = alpha
                    if with_anti_alias:
                        alpha_vj_uk *= gaussian_kernel(vj, v, sigma) * gaussian_kernel(uk, u, sigma)
                    alpha_vj_uk *= float(UINT8_MAX)
                    alpha_vj_uk_uint8 = np.uint8(alpha_vj_uk)
                    blend = alpha_blend_kernel(colors[i], img[vj, uk], alpha=alpha_vj_uk_uint8)
                    img[vj, uk, 0] = blend[0]
                    img[vj, uk, 1] = blend[1]
                    img[vj, uk, 2] = blend[2]
    return img


@nb.njit(nogil=True)
def clip_line_frustum(p1: NDArrayFloat, p2: NDArrayFloat, planes: NDArrayFloat) -> NDArrayFloat:
    """Iterate over the frustum planes and intersect them with the segment.

    We exploit the fact that in a camera frustum, all plane normals point inside the frustum volume.
    NOTE: See section "Line-Plane Intersection" for technical details at:
        http://geomalgorithms.com/a05-_intersect-1.html
    NOTE: A similar function, written in C, can be found in the Blender source code at:
        https://fossies.org/dox/blender-2.79b/math__geom_8c_source.html

    Args:
        p1: 3D vector defining a point to constrain a line segment.
        p2: 3D vector defining a point to constrain a line segment.
        planes: Array of shape (5,4), representing the 5 clipping planes for a frustum. Each plane is a 4-tuple
            representing the equation of a plane, e.g. (a, b, c, d) in ax + by + cz = d.

    Returns:
        (2,3) The array of clipped points. Points are NAN if the line segment
            does not lie within the frustum.
    """
    dp = np.subtract(p2, p1)

    p1_fac = 0.0
    p2_fac = 1.0
    pts: NDArrayFloat = np.full((2, 3), fill_value=NAN, dtype=p1.dtype)
    for i, _ in enumerate(planes):
        p = planes[i]
        div = p[:3].dot(dp)

        # check if line vector and plane normal are perpendicular
        # if perpendicular, line and plane are parallel
        if div != 0.0:
            # if not perpendicular, find intersection
            # distance we travel along the ray from p1 to p2
            t = -(p[:3].dot(p1) + p[3])
            if div > 0.0:  # clip p1 lower bounds
                if t >= div:
                    return pts
                if t > 0.0:
                    fac = t / div
                    if fac > p1_fac:
                        p1_fac = fac
                        if p1_fac > p2_fac:
                            # intersection occurs outside of segment
                            return pts
            elif div < 0.0:  # clip p2 upper bounds
                if t > 0.0:
                    return pts
                if t > div:
                    fac = t / div
                    if fac < p2_fac:
                        p2_fac = fac
                        if p1_fac > p2_fac:
                            return pts

    pts[0] = p1 + (dp * p1_fac)
    pts[1] = p1 + (dp * p2_fac)
    return pts
