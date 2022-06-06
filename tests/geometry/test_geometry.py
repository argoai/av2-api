# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for geometry utilities."""

from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import av2.geometry.geometry as geometry_utils
from av2.datasets.sensor.constants import AnnotationCategories
from av2.geometry.geometry import mat_to_xyz, xyz_to_mat
from av2.geometry.se3 import SE3
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt


@pytest.mark.parametrize(
    "yaw_1, yaw_2, expected_error_deg",
    [
        (179, -179, 2.0),
        (-179, 179, 2.0),
        (179, 178, 1.0),
        (178, 179, 1.0),
        (3, -3, 6.0),
        (-3, 3, 6.0),
        (-177, -179, 2.0),
        (-179, -177, 2.0),
    ],
    ids=[f"Wrap Angles (Test Case: {idx + 1})" for idx in range(8)],
)
def test_wrap_angles(yaw_1: float, yaw_2: float, expected_error_deg: float) -> None:
    """Test angle mapping (in radians) from domain [-∞, ∞] to [0, π).

    Args:
        yaw_1: Angle 1 in degrees.
        yaw_2: Angle 2 in degrees.
        expected_error_deg: Expected error mapped to the interval [0, π) in degrees.
    """
    yaw1 = np.deg2rad([yaw_1]).astype(np.float64)
    yaw2 = np.deg2rad([yaw_2]).astype(np.float64)

    error_deg = np.rad2deg(geometry_utils.wrap_angles(yaw1 - yaw2))
    assert np.allclose(error_deg, expected_error_deg, atol=1e-2)


def test_cart_to_hom_2d() -> None:
    """Convert 2d cartesian coordinates to homogenous, and back again."""
    cart: NDArrayFloat = np.arange(16 * 2).reshape(16, 2).astype(np.float64)

    hom = geometry_utils.cart_to_hom(cart=cart)
    cart_ = geometry_utils.hom_to_cart(hom=hom)

    assert np.array_equal(cart, cart_)


def test_cart_to_hom_3d() -> None:
    """Convert 3d cartesian coordinates to homogenous, and back again."""
    cart: NDArrayFloat = np.arange(16 * 3).reshape(16, 3).astype(np.float64)

    hom = geometry_utils.cart_to_hom(cart=cart)
    cart_ = geometry_utils.hom_to_cart(hom=hom)

    assert np.array_equal(cart, cart_)


@pytest.mark.parametrize(
    "xy, width, height, expected_uv",
    [
        (
            np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).astype(np.float64),
            1,
            1,
            np.array([[0, 0], [0, -1], [-1, -1], [-1, 0]]).astype(np.float64),
        ),
        (
            np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).astype(np.float64),
            2,
            0,
            np.array([[1, -1], [1, -2], [0, -2], [0, -1]]).astype(np.float64),
        ),
    ],
    ids=[f"Cartesian to texture coordinates conversion (Test Case: {idx + 1})" for idx in range(2)],
)
def test_xy_to_uv(xy: NDArrayFloat, width: int, height: int, expected_uv: NDArrayFloat) -> None:
    """Test conversion of coordinates in R^2 (x,y) to texture coordinates (u,v) in R^2.

    Args:
        xy: (...,2) array of coordinates in R^2 (x,y).
        width: Texture grid width.
        height: Texture grid height.
        expected_uv: Expected (...,2) array of texture / pixel coordinates.
    """
    assert np.array_equal(expected_uv, geometry_utils.xy_to_uv(xy, width, height))


@pytest.mark.parametrize(
    "quat_wxyz",
    [
        (np.array([0, 0, 0, 1]).astype(np.float64)),
        (np.array([[0, 0, 0, 1], [0, 0, 0.70710678, 0.70710678]]).astype(np.float64)),
    ],
    ids=[f"Quaternion to Matrix conversion (Test Case: {idx + 1})" for idx in range(2)],
)
def test_quat_to_mat_3d(quat_wxyz: NDArrayFloat) -> None:
    """Test conversion of quaternion to its matrix representation, and back again.

    Args:
        quat_wxyz: (...,4) array of quaternions in scalar first order.
    """
    # Quaternion to Quaternion round-trip conversion.
    # (Note: For comparison, Quaternion needs to be converted from scalar last to scalar first.)
    quat_to_quat: Callable[[NDArrayFloat], Any] = lambda quat_wxyz: Rotation.as_quat(
        Rotation.from_matrix(geometry_utils.quat_to_mat(quat_wxyz))
    )[..., [3, 0, 1, 2]]

    assert np.allclose(quat_wxyz, quat_to_quat(quat_wxyz))


@pytest.mark.parametrize(
    "cart_xyz, expected_sph_theta_phi_r",
    [
        (np.array([1, 1, 1]).astype(np.float64), np.array([0.78539816, 0.61547971, 1.73205081])),
        (
            np.array([[1, 1, 1], [1, 2, 0]]).astype(np.float64),
            np.array([[0.78539816, 0.61547971, 1.73205081], [1.10714872, 0.0, 2.23606798]]),
        ),
    ],
    ids=[f"Cartesian to Spherical coodinates (Test Case: {idx + 1})" for idx in range(2)],
)
def test_cart_to_sph_3d(cart_xyz: NDArrayFloat, expected_sph_theta_phi_r: NDArrayFloat) -> None:
    """Test conversion of cartesian coordinates to spherical coordinates.

    Args:
        cart_xyz: (...,3) Array of points (x,y,z) in Cartesian space.
        expected_sph_theta_phi_r: (...,3) Array in spherical space. [Order: (azimuth, inclination, radius)].
    """
    assert np.allclose(expected_sph_theta_phi_r, geometry_utils.cart_to_sph(cart_xyz))


@pytest.mark.parametrize(
    "points_xyz, lower_bound_inclusive, upper_bound_exclusive, expected_crop_points, expected_mask",
    [
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]).astype(
                np.int64
            ),
            (0, 0, 0),
            (1.5, 1.5, 1.5),
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]).astype(
                np.int64
            ),
            np.array([True] * 8),
        ),
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]).astype(
                np.int64
            ),
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            np.array([[0, 0, 0]]).astype(np.int64),
            np.array([True] + [False] * 7),
        ),
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]).astype(
                np.int64
            ),
            (0, 0, 0),
            (1.25, 1.25, 1.0),
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).astype(np.int64),
            np.array([True] * 4 + [False] * 4),
        ),
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]).astype(
                np.int64
            ),
            (-1.0, -1.0, -1.0),
            (0.0, 1.0, 1.0),
            np.empty((0, 3)).astype(np.int64),
            np.array([False] * 8),
        ),
    ],
    ids=[
        "All points fall within the cropping region.",
        "Only 1 point falls within the cropping region.",
        "Few points fall within the cropping region.",
        "No points fall within the cropping region.",
    ],
)
def test_crop_points(
    points_xyz: NDArrayInt,
    lower_bound_inclusive: Tuple[float, float, float],
    upper_bound_exclusive: Tuple[float, float, float],
    expected_crop_points: NDArrayInt,
    expected_mask: NDArrayBool,
) -> None:
    """Test cropping cartesian coordinates based on lower and upper boundary.

    Args:
        points_xyz: (...,3) Cartesian coordinates (x,y,z).
        lower_bound_inclusive: (3,) Cartesian coordinates lower bound (inclusive).
        upper_bound_exclusive: (3,) Cartesian coordinates upper bound (exclusive).
        expected_crop_points: (...,3) Expected tuple of cropped Cartesian coordinates.
        expected_mask: (...,) Expected boolean mask.
    """
    cropped_xyz, mask = geometry_utils.crop_points(points_xyz, lower_bound_inclusive, upper_bound_exclusive)

    np.testing.assert_array_equal(expected_crop_points, cropped_xyz)
    np.testing.assert_array_equal(expected_mask, mask)


@pytest.mark.parametrize(
    "points_xyz, expected_is_interior",
    [
        (
            np.array(
                [
                    [0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75],
                ]
            ).astype(np.float64),
            np.array([True] * 3),
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0],
                ]
            ).astype(np.float64),
            np.array([True] * 5),
        ),
        (
            np.array(
                [
                    [-0.5, -0.5, -0.5],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [1.0, 1.0, 1.0],
                    [1.5, 1.5, 1.5],
                ]
            ).astype(np.float64),
            np.array([False] + [True] * 3 + [False]),
        ),
        (
            np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [-0.5, -0.5, -0.5],
                    [1.5, 0.5, 0.5],
                    [2.0, 2.0, 2.0],
                ]
            ).astype(np.float64),
            np.array([False] * 4),
        ),
    ],
    ids=[
        "All points lie inside the bounding box.",
        "All points lie inside the bounding box (with boundary included).",
        "Few points lie inside the bounding box.",
        "No points lie inside the bounding box.",
    ],
)
def test_compute_interior_points_mask(points_xyz: NDArrayFloat, expected_is_interior: NDArrayBool) -> None:
    r"""Test finding the points interior to an axis-aligned cuboid.

    Reference: https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d

            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

    Args:
        points_xyz: (N,3) Array of points.
        expected_is_interior: (N,) Array of booleans whether the points are interior to the cuboid
            defined below.
    """
    rotation: NDArrayFloat = np.eye(3)
    translation: NDArrayFloat = np.array([0.5, 0.5, 0.5])
    egovehicle_SE3_ego = SE3(rotation=rotation, translation=translation)
    cuboid = Cuboid(
        dst_SE3_object=egovehicle_SE3_ego,
        length_m=1.0,
        width_m=1.0,
        height_m=1.0,
        category=AnnotationCategories.REGULAR_VEHICLE,
        timestamp_ns=0,
    )
    is_interior = geometry_utils.compute_interior_points_mask(points_xyz, cuboid.vertices_m)
    assert np.array_equal(is_interior, expected_is_interior)


def test_benchmark_compute_interior_points_mask_optimized(benchmark: Callable[..., Any]) -> None:
    """Benchmark compute_interior_pts on 100000 random points."""
    rotation: NDArrayFloat = np.eye(3)
    translation: NDArrayFloat = np.array([0.5, 0.5, 0.5])
    egovehicle_SE3_ego = SE3(rotation=rotation, translation=translation)
    cuboid = Cuboid(
        dst_SE3_object=egovehicle_SE3_ego,
        length_m=1.0,
        width_m=1.0,
        height_m=1.0,
        category=AnnotationCategories.REGULAR_VEHICLE,
        timestamp_ns=0,
    )

    N = 100000
    points_xyz: NDArrayFloat = 100.0 * np.random.rand(N, 3)
    benchmark(geometry_utils.compute_interior_points_mask, points_xyz, cuboid.vertices_m)


def test_benchmark_compute_interior_points_mask_slow(benchmark: Callable[..., Any]) -> None:
    """Benchmark compute_interior_points_mask on 100000 random points."""
    rotation: NDArrayFloat = np.eye(3)
    translation: NDArrayFloat = np.array([0.5, 0.5, 0.5])
    egovehicle_SE3_ego = SE3(rotation=rotation, translation=translation)
    cuboid = Cuboid(
        dst_SE3_object=egovehicle_SE3_ego,
        length_m=1.0,
        width_m=1.0,
        height_m=1.0,
        category=AnnotationCategories.REGULAR_VEHICLE,
        timestamp_ns=0,
    )

    N = 100000
    points_xyz: NDArrayFloat = 100.0 * np.random.rand(N, 3)

    def compute_interior_points_mask_slow(points_xyz: NDArrayFloat, cuboid_vertices: NDArrayFloat) -> NDArrayBool:
        """Compute the interior points mask with the older slow version.

        Args:
            points_xyz: (N,3) Array representing a point cloud in Cartesian coordinates (x,y,z).
            cuboid_vertices: (8,3) Array representing 3D cuboid vertices, ordered as shown below.

        Returns:
            (N,) An array of boolean flags indicating whether the points are interior to the cuboid.
        """
        u = cuboid_vertices[2] - cuboid_vertices[6]
        v = cuboid_vertices[2] - cuboid_vertices[3]
        w = cuboid_vertices[2] - cuboid_vertices[1]

        # point x lies within the box when the following
        # constraints are respected
        valid_u1 = np.logical_and(
            u.dot(cuboid_vertices[2]) <= points_xyz.dot(u), points_xyz.dot(u) <= u.dot(cuboid_vertices[6])
        )
        valid_v1 = np.logical_and(
            v.dot(cuboid_vertices[2]) <= points_xyz.dot(v), points_xyz.dot(v) <= v.dot(cuboid_vertices[3])
        )
        valid_w1 = np.logical_and(
            w.dot(cuboid_vertices[2]) <= points_xyz.dot(w), points_xyz.dot(w) <= w.dot(cuboid_vertices[1])
        )

        valid_u2 = np.logical_and(
            u.dot(cuboid_vertices[2]) >= points_xyz.dot(u), points_xyz.dot(u) >= u.dot(cuboid_vertices[6])
        )
        valid_v2 = np.logical_and(
            v.dot(cuboid_vertices[2]) >= points_xyz.dot(v), points_xyz.dot(v) >= v.dot(cuboid_vertices[3])
        )
        valid_w2 = np.logical_and(
            w.dot(cuboid_vertices[2]) >= points_xyz.dot(w), points_xyz.dot(w) >= w.dot(cuboid_vertices[1])
        )

        valid_u = np.logical_or(valid_u1, valid_u2)
        valid_v = np.logical_or(valid_v1, valid_v2)
        valid_w = np.logical_or(valid_w1, valid_w2)

        is_interior: NDArrayBool = np.logical_and(np.logical_and(valid_u, valid_v), valid_w)
        return is_interior

    benchmark(compute_interior_points_mask_slow, points_xyz, cuboid.vertices_m)


def test_xyz_to_mat_matrix() -> None:
    """Unit test for converting Tait-Bryan angles to rotation matrix."""
    tait_bryan_angles: NDArrayFloat = np.deg2rad([45.0, 45.0, 45.0])
    rotation_matrix = xyz_to_mat(tait_bryan_angles)
    rotation_matrix_expected = [  # (45, 45, 45) about the xyz axes (roll, pitch, yaw).
        [0.5000000, -0.1464466, 0.8535534],
        [0.5000000, 0.8535534, -0.1464466],
        [-0.7071068, 0.5000000, 0.5000000],
    ]
    np.testing.assert_allclose(rotation_matrix, rotation_matrix_expected)


def test_xyz_to_mat_round_trip() -> None:
    """Unit test for converting Tait-Bryan angles to rotation matrix."""
    tait_bryan_angles: NDArrayFloat = np.deg2rad([45.0, 45.0, 45.0])
    rotation_matrix = xyz_to_mat(tait_bryan_angles)
    tait_bryan_angles_ = mat_to_xyz(rotation_matrix)
    np.testing.assert_allclose(tait_bryan_angles, tait_bryan_angles_)


def test_mat_to_xyz_round_trip() -> None:
    """Unit test for converting Tait-Bryan angles to rotation matrix."""
    rotation_matrix: NDArrayFloat = np.array(
        [  # (45, 0, 90) about the xyz axes (roll, pitch, yaw).
            [0.0000000, -0.7071068, 0.7071068],
            [1.0000000, 0.0000000, -0.0000000],
            [0.0000000, 0.7071068, 0.7071068],
        ]
    )
    tait_bryan_angles = mat_to_xyz(rotation_matrix)
    rotation_matrix_ = xyz_to_mat(tait_bryan_angles)
    np.testing.assert_allclose(rotation_matrix, rotation_matrix_, atol=1e-10)


def test_mat_to_xyz_constrained() -> None:
    """Unit test for constraining Tait-Bryan angles and converting to rotation matrix."""
    rotation_matrix: NDArrayFloat = np.array(
        [  # (45, 0, 90) about the xyz axes (roll, pitch, yaw).
            [0.0000000, -0.7071068, 0.7071068],
            [1.0000000, 0.0000000, -0.0000000],
            [0.0000000, 0.7071068, 0.7071068],
        ]
    )
    xyz = mat_to_xyz(rotation_matrix)
    xyz[0] = 0  # Set roll to zero.
    mat_constrained = xyz_to_mat(xyz)

    xyz_expected = np.deg2rad([0, 0, 90])  # [45, 0, 90] -> constrain roll to zero -> [0, 0, 90].
    mat_constrained_expected = xyz_to_mat(xyz_expected)
    np.testing.assert_allclose(mat_constrained, mat_constrained_expected)


def RzRyRx(x: float, y: float, z: float) -> NDArrayFloat:
    """Convert rotation angles about 3 orthogonal axes to a 3d rotation matrix.

    Computes:
        R = Rz(z) * Ry(y) * Rx(x)

    For a derivation, see page 2 of http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

    Reference: GTSAM
    See: https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Rot3M.cpp#L85

    Args:
        x: roll angle, in radians.
        y: pitch angle, in radians.
        z: yaw angle, in radians.

    Returns:
        The 3d rotation matrix.
    """
    cx = np.cos(x)
    cy = np.cos(y)
    cz = np.cos(z)

    sx = np.sin(x)
    sy = np.sin(y)
    sz = np.sin(z)

    ss_ = sx * sy
    cs_ = cx * sy
    sc_ = sx * cy
    cc_ = cx * cy
    c_s = cx * sz
    s_s = sx * sz
    _cs = cy * sz
    _cc = cy * cz
    s_c = sx * cz
    c_c = cx * cz
    ssc = ss_ * cz
    csc = cs_ * cz
    sss = ss_ * sz
    css = cs_ * sz

    # fmt: off
    R: NDArrayFloat = np.array([
        [_cc, -c_s + ssc,  s_s + csc],
        [_cs,  c_c + sss, -s_c + css],
        [-sy,        sc_,        cc_]
    ])
    # fmt: on
    return R


def test_xyz_to_mat_vs_gtsam() -> None:
    """Compare our implementation (using Scipy) vs. the GTSAM derivation."""
    num_iters = 10000
    for _ in range(num_iters):

        # in [-pi, pi]
        x = 2 * np.pi * (np.random.rand() - 0.5)
        z = 2 * np.pi * (np.random.rand() - 0.5)

        # in [-pi/2, pi/2]
        y = np.pi * (np.random.rand() - 0.5)

        xyz: NDArrayFloat = np.array([x, y, z], dtype=float)
        R = xyz_to_mat(xyz)

        R_gtsam = RzRyRx(x, y, z)
        assert np.allclose(R, R_gtsam)


def test_constrain_cuboid_pose() -> None:
    """Unit test to constrain cuboid pose."""
    path = Path(__file__).parent.resolve() / "data" / "b87683ae-14c5-321f-8af3-623e7bafc3a7" / "annotations.feather"
    cuboid_list = CuboidList.from_feather(path)
    for cuboid in cuboid_list.cuboids:
        pose = cuboid.dst_SE3_object
        roll, pitch, yaw = mat_to_xyz(pose.rotation)
        pitch, roll = 0, 0
        xyz: NDArrayFloat = np.array([roll, pitch, yaw], dtype=float)
        cuboid.dst_SE3_object.rotation = xyz_to_mat(xyz)
        R = xyz_to_mat(xyz)
        R_ = RzRyRx(0, 0, yaw)
        assert np.allclose(R, R_)
