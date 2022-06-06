# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for PinholeCamera model."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from av2.datasets.sensor.constants import RingCameras
from av2.geometry.camera.pinhole_camera import Intrinsics, PinholeCamera
from av2.geometry.se3 import SE3
from av2.utils.typing import NDArrayFloat, NDArrayInt

_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "test_data"


def _create_pinhole_camera(
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    height_px: int,
    width_px: int,
    cam_name: str,
) -> PinholeCamera:
    """Create a pinhole camera."""
    rotation: NDArrayFloat = np.eye(3)
    translation: NDArrayFloat = np.zeros(3)
    ego_SE3_cam = SE3(rotation=rotation, translation=translation)

    intrinsics = Intrinsics(fx_px=fx_px, fy_px=fy_px, cx_px=cx_px, cy_px=cy_px, width_px=width_px, height_px=height_px)
    pinhole_camera = PinholeCamera(ego_SE3_cam, intrinsics, cam_name)
    return pinhole_camera


def _fit_plane_to_point_cloud(points_xyz: NDArrayFloat) -> Tuple[float, float, float, float]:
    """Use SVD with at least 3 points to fit a plane.

    Args:
        points_xyz: (N,3) array of points.

    Returns:
        (4,) Plane coefficients. Defining ax + by + cz = d for the plane.
    """
    center_xyz: NDArrayFloat = np.mean(points_xyz, axis=0)
    out: Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat] = np.linalg.svd(points_xyz - center_xyz)
    vh = out[2]

    # Get the unitary normal vector
    a, b, c = float(vh[2, 0]), float(vh[2, 1]), float(vh[2, 2])
    d: float = -np.dot([a, b, c], center_xyz)
    return (a, b, c, d)


def test_intrinsics_constructor() -> None:
    """Ensure 3x3 intrinsics matrix is populated correctly."""
    fx_px, fy_px = 1000, 1001

    width_px = 2048
    height_px = 1550

    cx_px, cy_px = 1024, 775

    intrinsics = Intrinsics(fx_px=fx_px, fy_px=fy_px, cx_px=cx_px, cy_px=cy_px, width_px=width_px, height_px=height_px)
    K_expected: NDArrayFloat = np.array(([1000, 0, 1024], [0, 1001, 775], [0, 0, 1]), dtype=np.float64)
    assert np.array_equal(intrinsics.K, K_expected)


def test_right_clipping_plane() -> None:
    """Test form_right_clipping_plane(). Use 4 points to fit the right clipping plane.

    In the camera coordinate frame, y is down the imager, x is across the imager,
    and z is along the optical axis. The focal length is the distance to the center
    of the image plane. We know that a similar triangle is formed as follows:

    (x,y,z)---(x,y,z)
       |         /
       |        / ->outside of frustum
       |       / ->outside of frustum
       | (w/2)/
       o-----o IMAGE PLANE
       |    /
     fx|   /
       |  /
       | /
       O PINHOLE

    Normal must point into the frustum. The plane moves +fx in z-axis for
    every +w/2 in x-axis, so normal will have negative inverse slope components.
    """
    fx_px = 10.0
    width_px = 30
    pinhole_camera = _create_pinhole_camera(
        fx_px=fx_px, fy_px=0, cx_px=0, cy_px=0, height_px=30, width_px=width_px, cam_name="ring_front_center"
    )
    right_plane = pinhole_camera.right_clipping_plane

    Y_OFFSET = 10  # arbitrary extent down the imager

    right: NDArrayFloat = np.array(
        [
            [0, 0, 0],
            [width_px / 2.0, 0, fx_px],
            [0, Y_OFFSET, 0],
            [width_px / 2.0, Y_OFFSET, fx_px],
        ]
    )

    a, b, c, d = _fit_plane_to_point_cloud(right)
    right_plane_expected: NDArrayFloat = np.array([a, b, c, d])

    # enforce that plane normal points into the frustum
    # x-component of normal should point in negative direction.
    if right_plane_expected[0] > 0:
        right_plane_expected *= -1

    assert np.allclose(right_plane, right_plane_expected)


def test_left_clipping_plane() -> None:
    r"""Test left_clipping_plane. Use 4 points to fit the left clipping plane.

                      (x,y,z)-----(x,y,z)
                         \\          |
    outside of frustum <- \\         |
     outside of frustum <- \\        |
                            \\ (-w/2)|
                              o------o IMAGE PLANE
                              \\     |
                               \\    |
                                \\   |fx
                                 \\  |
                                  \\ |
                                     O PINHOLE
    """
    fx_px = 10.0
    width_px = 30

    pinhole_camera = _create_pinhole_camera(
        fx_px=fx_px, fy_px=0, cx_px=0, cy_px=0, height_px=30, width_px=width_px, cam_name="ring_front_center"
    )
    left_plane = pinhole_camera.left_clipping_plane

    Y_OFFSET = 10
    points_xyz: NDArrayFloat = np.array(
        [
            [0, 0, 0],
            [-width_px / 2.0, 0, fx_px],
            [0, Y_OFFSET, 0],
            [-width_px / 2.0, Y_OFFSET, fx_px],
        ]
    )

    a, b, c, d = _fit_plane_to_point_cloud(points_xyz)
    left_plane_expected = -np.array([a, b, c, d])

    # enforce that plane normal points into the frustum
    if left_plane_expected[0] < 0:
        left_plane_expected *= -1

    assert np.allclose(left_plane, left_plane_expected)


def test_top_clipping_plane() -> None:
    r"""Test top_clipping_plane. Use 3 points to fit the TOP clipping plane.

      (x,y,z)               (x,y,z)
          \\=================//
           \\               //
    (-w/h,-h/2,fx)       (w/h,-h/2,fx)
             o-------------o
             |\\         //| IMAGE PLANE
             | \\       // | IMAGE PLANE
             o--\\-----//--o
                 \\   //
                  \\ //
                    O PINHOLE
    """
    fx_px = 10.0
    height_px = 45
    pinhole_camera = _create_pinhole_camera(
        fx_px=fx_px, fy_px=0, cx_px=0, cy_px=0, height_px=height_px, width_px=1000, cam_name="ring_front_center"
    )

    top_plane = pinhole_camera.top_clipping_plane

    width_px = 1000.0
    points_xyz: NDArrayFloat = np.array(
        [
            [0, 0, 0],
            [-width_px / 2, -height_px / 2, fx_px],
            [width_px / 2, -height_px / 2, fx_px],
        ]
    )
    a, b, c, d = _fit_plane_to_point_cloud(points_xyz)
    top_plane_expected: NDArrayFloat = np.array([a, b, c, d])

    # enforce that plane normal points into the frustum
    if top_plane_expected[1] < 0:
        # y-coord of normal should point in pos y-axis dir(down) on top-clipping plane
        top_plane_expected *= -1

    assert top_plane_expected[1] > 0 and top_plane_expected[2] > 0
    assert np.allclose(top_plane, top_plane_expected)


def test_bottom_clipping_plane() -> None:
    r"""Test bottom_clipping_plane. Use 3 points to fit the BOTTOM clipping plane.

           (x,y,z)              (x,y,z)
              \\                   //
               \\ o-------------o //
                \\| IMAGE PLANE |//
                  |             |/
    (-w/h,h/2,fx) o-------------o (w/h,h/2,fx)
                   \\         //
                    \\       //
                     \\     //
                      \\   //
                       \\ //
                         O PINHOLE
    """
    fx_px = 12.0
    height_px = 35
    width_px = 10000

    pinhole_camera = _create_pinhole_camera(
        fx_px=fx_px, fy_px=1, cx_px=0, cy_px=0, height_px=height_px, width_px=width_px, cam_name="ring_front_center"
    )
    bottom_plane = pinhole_camera.bottom_clipping_plane

    low_pts: NDArrayFloat = np.array(
        [
            [0, 0, 0],
            [-width_px / 2, height_px / 2, fx_px],
            [width_px / 2, height_px / 2, fx_px],
        ]
    )
    a, b, c, d = _fit_plane_to_point_cloud(low_pts)
    bottom_plane_expected: NDArrayFloat = np.array([a, b, c, d])

    # enforce that plane normal points into the frustum
    # y-coord of normal should point in neg y-axis dir(up) on low-clipping plane
    # z-coord should point in positive z-axis direction (away from camera)
    if bottom_plane_expected[1] > 0:
        bottom_plane_expected *= -1
    assert bottom_plane_expected[1] < 0 and bottom_plane_expected[2] > 0

    assert np.allclose(bottom_plane, bottom_plane_expected)


def test_form_near_clipping_plane() -> None:
    """Test near_clipping_plane(). Use 4 points to fit the near clipping plane."""
    width_px = 10
    height_px = 15
    near_clip_dist = 30.0

    pinhole_camera = _create_pinhole_camera(
        fx_px=1, fy_px=0, cx_px=0, cy_px=0, height_px=30, width_px=width_px, cam_name="ring_front_center"
    )
    near_plane = pinhole_camera.near_clipping_plane(near_clip_dist)

    points_xyz: NDArrayFloat = np.array(
        [
            [width_px / 2, 0, near_clip_dist],
            [-width_px / 2, 0, near_clip_dist],
            [width_px / 2, -height_px / 2.0, near_clip_dist],
            [width_px / 2, height_px / 2.0, near_clip_dist],
        ]
    )

    a, b, c, d = _fit_plane_to_point_cloud(points_xyz)
    near_plane_expected: NDArrayFloat = np.array([a, b, c, d])

    assert np.allclose(near_plane, near_plane_expected)


def test_frustum_planes_ring_cam() -> None:
    """Test frustum_planes for a ring camera."""
    near_clip_dist = 6.89  # arbitrary value

    # Set "focal_length_x_px_"
    fx_px = 1402.4993697398709

    # Set "focal_length_y_px_"
    fy_px = 1405.1207294310225

    # Set "focal_center_x_px_"
    cx_px = 957.8471720086527

    # Set "focal_center_y_px_"
    cy_px = 600.442948946496

    camera_name = "ring_front_right"
    height_px = 1550
    width_px = 2048

    pinhole_camera = _create_pinhole_camera(
        fx_px=fx_px, fy_px=fy_px, cx_px=cx_px, cy_px=cy_px, height_px=height_px, width_px=width_px, cam_name=camera_name
    )
    left_plane, right_plane, near_plane, bottom_plane, top_plane = pinhole_camera.frustum_planes(near_clip_dist)

    left_plane_expected: NDArrayFloat = np.array([fx_px, 0.0, width_px / 2.0, 0.0])
    right_plane_expected: NDArrayFloat = np.array([-fx_px, 0.0, width_px / 2.0, 0.0])
    near_plane_expected: NDArrayFloat = np.array([0.0, 0.0, 1.0, -near_clip_dist])
    bottom_plane_expected: NDArrayFloat = np.array([0.0, -fx_px, height_px / 2.0, 0.0])
    top_plane_expected: NDArrayFloat = np.array([0.0, fx_px, height_px / 2.0, 0.0])

    assert np.allclose(left_plane, left_plane_expected / np.linalg.norm(left_plane_expected))
    assert np.allclose(right_plane, right_plane_expected / np.linalg.norm(right_plane_expected))
    assert np.allclose(bottom_plane, bottom_plane_expected / np.linalg.norm(bottom_plane_expected))
    assert np.allclose(top_plane, top_plane_expected / np.linalg.norm(top_plane_expected))
    assert np.allclose(near_plane, near_plane_expected)


def test_generate_frustum_planes_stereo() -> None:
    """Test generate_frustum_planes() for a stereo camera."""
    near_clip_dist = 3.56  # arbitrary value

    # Set "focal_length_x_px_"
    fx_px = 3666.534329132812

    # Set "focal_length_y_px_"
    fy_px = 3673.5030423482513

    # Set "focal_center_x_px_"
    cx_px = 1235.0158218941356

    # Set "focal_center_y_px_"
    cy_px = 1008.4536901420888

    camera_name = "stereo_front_left"
    height_px = 1550
    width_px = 2048

    pinhole_camera = _create_pinhole_camera(
        fx_px=fx_px, fy_px=fy_px, cx_px=cx_px, cy_px=cy_px, height_px=height_px, width_px=width_px, cam_name=camera_name
    )
    left_plane, right_plane, near_plane, bottom_plane, top_plane = pinhole_camera.frustum_planes(near_clip_dist)

    left_plane_expected: NDArrayFloat = np.array([fx_px, 0.0, width_px / 2.0, 0.0])
    right_plane_expected: NDArrayFloat = np.array([-fx_px, 0.0, width_px / 2.0, 0.0])
    near_plane_expected: NDArrayFloat = np.array([0.0, 0.0, 1.0, -near_clip_dist])
    bottom_plane_expected: NDArrayFloat = np.array([0.0, -fx_px, height_px / 2.0, 0.0])
    top_plane_expected: NDArrayFloat = np.array([0.0, fx_px, height_px / 2.0, 0.0])

    assert np.allclose(left_plane, left_plane_expected / np.linalg.norm(left_plane_expected))
    assert np.allclose(right_plane, right_plane_expected / np.linalg.norm(right_plane_expected))
    assert np.allclose(bottom_plane, bottom_plane_expected / np.linalg.norm(bottom_plane_expected))
    assert np.allclose(top_plane, top_plane_expected / np.linalg.norm(top_plane_expected))
    assert np.allclose(near_plane, near_plane_expected)


def test_compute_pixel_ray_directions_vectorized_invalid_focal_lengths() -> None:
    """If focal lengths in the x and y directions do not match, we throw an exception.

    Tests vectorized variant (multiple ray directions.)
    """
    uv: NDArrayInt = np.array([[12, 2], [12, 2], [12, 2], [12, 2]])
    fx = 10
    fy = 11

    img_w = 20
    img_h = 10

    pinhole_camera = _create_pinhole_camera(
        fx_px=fx,
        fy_px=fy,
        cx_px=img_w / 2,
        cy_px=img_h / 2,
        height_px=img_h,
        width_px=img_w,
        cam_name="ring_front_center",  # dummy name
    )

    with pytest.raises(ValueError):
        pinhole_camera.compute_pixel_ray_directions(uv)


def test_compute_pixel_ray_direction_invalid_focal_lengths() -> None:
    """If focal lengths in the x and y directions do not match, we throw an exception.

    Tests non-vectorized variant (single ray direction).
    """
    u = 12
    v = 2
    fx = 10
    fy = 11

    img_w = 20
    img_h = 10
    with pytest.raises(ValueError):
        _compute_pixel_ray_direction(u, v, fx, fy, img_w, img_h)


def test_compute_pixel_ray_directions_vectorized() -> None:
    """Ensure that the ray direction (in camera coordinate frame) for each pixel is computed correctly.

    Small scale test, for just four selected positions in a 10 x 20 px image in (height, width).
    """
    fx = 10
    fy = 10

    # dummy 2d coordinates in the image plane.
    uv: NDArrayInt = np.array([[12, 2], [12, 2], [12, 2], [12, 2]])

    # principal point is at (10,5)
    img_w = 20
    img_h = 10

    pinhole_camera = _create_pinhole_camera(
        fx_px=fx,
        fy_px=fy,
        cx_px=img_w / 2,
        cy_px=img_h / 2,
        height_px=img_h,
        width_px=img_w,
        cam_name="ring_front_center",  # dummy name
    )
    ray_dirs = pinhole_camera.compute_pixel_ray_directions(uv)

    gt_ray_dir: NDArrayFloat = np.array([2, -3, 10.0])
    gt_ray_dir /= np.linalg.norm(gt_ray_dir)

    for i in range(4):
        assert np.allclose(gt_ray_dir, ray_dirs[i])


def test_compute_pixel_ray_directions_vectorized_entireimage() -> None:
    """Ensure that the ray direction for each pixel (in camera coordinate frame) is computed correctly.

    Compare all computed rays against non-vectorized variant, for correctness.
    Larger scale test, for every pixel in a 50 x 100 px image in (height, width).
    """
    fx = 10
    fy = 10

    img_w = 100
    img_h = 50

    pinhole_camera = _create_pinhole_camera(
        fx_px=fx,
        fy_px=fy,
        cx_px=img_w / 2,
        cy_px=img_h / 2,
        height_px=img_h,
        width_px=img_w,
        cam_name="ring_front_center",  # dummy name
    )

    uv_list = []
    for u in range(img_w):
        for v in range(img_h):
            uv_list += [(u, v)]

    uv: NDArrayInt = np.array(uv_list)
    assert uv.shape == (img_w * img_h, 2)

    ray_dirs = pinhole_camera.compute_pixel_ray_directions(uv)

    # compare w/ vectorized, should be identical
    for i, ray_dir_vec in enumerate(ray_dirs):
        u, v = uv[i]
        ray_dir_nonvec = _compute_pixel_ray_direction(u, v, fx, fy, img_w, img_h)
        assert np.allclose(ray_dir_vec, ray_dir_nonvec)


def test_compute_pixel_rays() -> None:
    """Ensure that the ray direction (in camera coordinate frame) for a single pixel is computed correctly.

    Small scale test, for just one selected position in a 10 x 20 px image in (height, width).
    For row = 2, column = 12.
    """
    u = 12
    v = 2
    img_w = 20
    img_h = 10
    fx = 10
    fy = 10

    ray_dir = _compute_pixel_ray_direction(u, v, fx, fy, img_w, img_h)

    gt_ray_dir: NDArrayFloat = np.array([2.0, -3.0, 10.0])
    gt_ray_dir /= np.linalg.norm(gt_ray_dir)

    assert np.allclose(gt_ray_dir, ray_dir)


def _compute_pixel_ray_direction(u: float, v: float, fx: float, fy: float, img_w: int, img_h: int) -> NDArrayFloat:
    r"""Generate rays in the camera coordinate frame.

    Note: only used as a test utility.

       Find point P on image plane.

                      (x,y,z)-----(x,y,z)
                         \\          |
    outside of frustum <- \\         |
     outside of frustum <- \\        |
                            \\ (-w/2)|
                              o------o IMAGE PLANE
                              \\     |
                               \\    |
                                \\   |fx
                                 \\  |
                                  \\ |
                                     O PINHOLE

    Args:
        u: pixel's x-coordinate
        v: pixel's y-coordinate
        fx: focal length in x-direction, measured in pixels.
        fy: focal length in y-direction,  measured in pixels.
        img_w: image width (in pixels)
        img_h: image height (in pixels)

    Returns:
        Direction of 3d ray, provided in the camera frame.

    Raises:
        ValueError: If horizontal and vertical focal lengths are not close (within 1e-3).
    """
    if not np.isclose(fx, fy, atol=1e-3):
        raise ValueError(f"Focal lengths in the x and y directions must match: {fx} != {fy}")

    # approximation for principal point
    px = img_w / 2
    py = img_h / 2

    # the camera coordinate frame (where Z is out, x is right, y is down).

    # compute offset from the center
    x_center_offs = u - px
    y_center_offs = v - py

    ray_dir: NDArrayFloat = np.array([x_center_offs, y_center_offs, fx])
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_dir


def test_get_frustum_parameters() -> None:
    r"""Ensure we can compute field of view, and camera's yaw in the egovehicle frame.

      w/2 = 1000
    o----------o IMAGE PLANE
    \\         |       //
      \\       |     //
        \\     |fx = 1000
          \\   |   //
            \\ | //
               O PINHOLE
    """
    fx, fy = 1000, 1000
    img_w = 2000
    img_h = 1000

    pinhole_camera = _create_pinhole_camera(
        fx_px=fx,
        fy_px=fy,
        cx_px=img_w / 2,
        cy_px=img_h / 2,
        height_px=img_h,
        width_px=img_w,
        cam_name="ring_front_center",  # dummy name
    )

    fov_theta_deg = np.rad2deg(pinhole_camera.fov_theta_rad)
    assert np.isclose(fov_theta_deg, 90.0)

    # for identity SE(3), the yaw angle is zero radians.
    cam_yaw_ego = pinhole_camera.egovehicle_yaw_cam_rad
    assert np.isclose(cam_yaw_ego, 0)


def test_get_egovehicle_yaw_cam() -> None:
    """Ensure we can compute the camera's yaw in the egovehicle frame."""
    sample_log_dir = _TEST_DATA_ROOT / "sensor_dataset_logs" / "test_log"

    # clockwise around the top of the car, in degrees.
    expected_ego_yaw_cam_deg_dict = {
        "ring_rear_left": 153.2,
        "ring_side_left": 99.4,
        "ring_front_left": 44.7,
        "ring_front_center": 0.4,
        "ring_front_right": -44.9,
        "ring_side_right": -98.9,
        "ring_rear_right": -152.9,
    }

    for cam_enum in list(RingCameras):
        cam_name = cam_enum.value
        pinhole_camera = PinholeCamera.from_feather(log_dir=sample_log_dir, cam_name=cam_name)

        ego_yaw_cam_deg = np.rad2deg(pinhole_camera.egovehicle_yaw_cam_rad)
        assert np.isclose(ego_yaw_cam_deg, expected_ego_yaw_cam_deg_dict[cam_name], atol=0.1)

        np.rad2deg(pinhole_camera.fov_theta_rad)
