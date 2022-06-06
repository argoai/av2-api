# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for rasterization related utilities."""

from typing import Any, Callable

import cv2
import numpy as np

import av2.utils.raster as raster_utils
from av2.utils.typing import NDArrayByte, NDArrayFloat


def test_get_mask_from_polygon() -> None:
    """Ensure that a triangle and skinny-column-like rectangle can be correctly rasterized onto a square grid."""
    # fmt: off
    triangle: NDArrayFloat = np.array(
        [
            [1, 1],
            [1, 3],
            [3, 1]
        ]
    )
    rectangle: NDArrayFloat = np.array(
        [
            [5, 1],
            [5, 4],
            [5.5, 4],
            [5.5, 1]
        ]
    )
    # fmt: on
    mask = raster_utils.get_mask_from_polygons(polygons=[triangle, rectangle], img_h=7, img_w=7)

    # fmt: off
    expected_mask: NDArrayByte = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype(np.uint8)
    # fmt: on
    assert np.array_equal(mask, expected_mask)
    assert mask.dtype == expected_mask.dtype


def test_get_mask_from_polygon_repeated_coords() -> None:
    """Verify polygon rasterization works correctly when the first coordinate is repeated (as last coordinate).

    Note: same scenario as above, a square grid with 2 polygons: a triangle and skinny-column-like rectangle.
    """
    # fmt: off
    triangle: NDArrayFloat = np.array(
        [
            [1, 1],
            [1, 3],
            [3, 1],
            [1, 1]
        ]
    )
    rectangle: NDArrayFloat = np.array(
        [
            [5, 1],
            [5, 4],
            [5.5, 4],
            [5.5, 1],
            [5, 1]
        ]
    )
    # fmt: on
    mask = raster_utils.get_mask_from_polygons(polygons=[triangle, rectangle], img_h=7, img_w=7)

    # fmt: off
    expected_mask: NDArrayByte = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype(np.uint8)
    # fmt: on
    assert np.array_equal(mask, expected_mask)
    assert mask.dtype == expected_mask.dtype


def test_get_mask_from_polygon_coords_out_of_bounds() -> None:
    """Test rasterization with polygon coordinates outside of the boundaries."""
    # fmt: off
    rectangle: NDArrayFloat = np.array(
        [
            [-2, 1],
            [8, 1],
            [8, 2],
            [-5, 2]
        ]
    )
    # fmt: on
    mask = raster_utils.get_mask_from_polygons(polygons=[rectangle], img_h=5, img_w=5)

    # fmt: off
    expected_mask: NDArrayByte = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    ).astype(np.uint8)
    # fmt: on
    assert np.array_equal(mask, expected_mask)
    assert mask.dtype == expected_mask.dtype


def test_benchmark_blend_images_cv2(benchmark: Callable[..., Any]) -> None:
    """Benchmark opencv implementation of alpha blending."""

    def blend_images(img0: NDArrayByte, img1: NDArrayByte, alpha: float = 0.7) -> NDArrayByte:
        """Alpha-blend two images together using OpenCV `addWeighted`.

        Args:
            img0: uint8 array of shape (H,W,3).
            img1: uint8 array of shape (H,W,3).
            alpha: alpha blending coefficient.

        Returns:
            uint8 array of shape (H,W,3)
        """
        blended: NDArrayByte = cv2.addWeighted(img0, alpha, img1, (1 - alpha), gamma=0)
        return blended

    size = (2048, 1550, 3)
    img0: NDArrayByte = np.random.randint(0, 255, size=size).astype(np.uint8)
    img1: NDArrayByte = np.random.randint(0, 255, size=size).astype(np.uint8)
    benchmark(blend_images, img0, img1)


def test_benchmark_blend_images_npy(benchmark: Callable[..., Any]) -> None:
    """Benchmark numpy implementation of alpha blending."""

    def blend_images(img0: NDArrayByte, img1: NDArrayByte, alpha: float = 0.7) -> NDArrayByte:
        """Alpha-blend two images together using `numpy`.

        Args:
            img0: uint8 array of shape (H,W,3)
            img1: uint8 array of shape (H,W,3)
            alpha: Alpha blending coefficient.

        Returns:
            uint8 array of shape (H,W,3)
        """
        blended: NDArrayFloat = np.multiply(img0.astype(np.float32), alpha, dtype=float) + np.multiply(
            img1.astype(np.float32),
            (1 - alpha),
            dtype=float,
        )
        blended_uint8: NDArrayByte = blended.astype(np.uint8)
        return blended_uint8

    size = (2048, 1550, 3)
    img0: NDArrayByte = np.random.randint(0, 255, size=size).astype(np.uint8)
    img1: NDArrayByte = np.random.randint(0, 255, size=size).astype(np.uint8)
    benchmark(blend_images, img0, img1)


def test_blend_images() -> None:
    """Test alpha blending two images together."""
    alpha = 0.7
    beta = 1 - alpha
    gamma = 0.0

    H, W = 3, 3
    img_a: NDArrayByte = np.zeros((H, W, 3), dtype=np.uint8)
    img_a[:, :, 0] = np.array([2, 4, 8])  # column 0 has uniform intensity
    img_a[:, :, 1] = np.array([4, 8, 16])
    img_a[:, :, 2] = np.array([8, 16, 32])

    img_b: NDArrayByte = np.zeros((H, W, 3), dtype=np.uint8)
    img_b[:, :, :3] = np.array([2, 4, 8])  # column 0 has uniform intensity

    blended_img_expected: NDArrayByte = np.round(img_a * alpha + img_b * beta + gamma).astype(np.uint8)
    blended_img = raster_utils.blend_images(img_a, img_b, alpha=alpha)
    assert np.array_equal(blended_img, blended_img_expected)
