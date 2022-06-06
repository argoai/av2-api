# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Detection evaluation unit tests.

Only the last two unit tests here use map ROI information.
The rest apply no filtering to objects that have their corners located outside of the ROI.
"""

import math
from pathlib import Path
from typing import Final, List

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from av2.evaluation.detection.constants import AffinityType, DistanceType
from av2.evaluation.detection.eval import evaluate
from av2.evaluation.detection.utils import (
    DetectionCfg,
    accumulate,
    assign,
    compute_affinity_matrix,
    compute_evaluated_dts_mask,
    compute_evaluated_gts_mask,
    compute_objects_in_roi_mask,
    distance,
    interpolate_precision,
)
from av2.geometry.geometry import wrap_angles
from av2.geometry.iou import iou_3d_axis_aligned
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import ORDERED_CUBOID_COL_NAMES
from av2.utils.constants import PI
from av2.utils.io import read_city_SE3_ego, read_feather
from av2.utils.typing import NDArrayBool, NDArrayFloat

TEST_DATA_DIR: Final[Path] = Path(__file__).parent.resolve() / "data"

TRANSLATION_COLS: Final[List[str]] = ["tx_m", "ty_m", "tz_m"]
DIMS_COLS: Final[List[str]] = ["length_m", "width_m", "height_m"]
QUAT_COLS: Final[List[str]] = ["qw", "qx", "qy", "qz"]
ANNO_COLS: Final[List[str]] = ["timestamp_ns", "category"] + DIMS_COLS + QUAT_COLS + TRANSLATION_COLS

CUBOID_COLS: Final[List[str]] = ["tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"]


def _get_summary_identity() -> pd.DataFrame:
    """Define an evaluator that compares a set of results to itself."""
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "detections_identity.feather")
    gts: pd.DataFrame = dts.copy()
    gts.loc[:, "num_interior_pts"] = np.array([1, 1, 1, 1, 1, 1])
    _, _, summary = evaluate(dts, gts, detection_cfg)
    return summary


def _get_summary_assignment() -> pd.DataFrame:
    """Define an evaluator that compares a set of results to one with an extra detection to check assignment."""
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "detections_assignment.feather")
    gts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "labels.feather")

    _, _, summary = evaluate(dts, gts, detection_cfg)
    return summary


def _get_summary() -> pd.DataFrame:
    """Get a dummy summary."""
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "detections.feather")
    gts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "labels.feather")
    _, _, summary = evaluate(dts, gts, detection_cfg)
    return summary


def test_affinity_center() -> None:
    """Initialize a detection and a ground truth label.

    Verify that calculated distance matches expected affinity under the specified `AffFnType`.
    """
    dts: NDArrayFloat = np.array([[0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]])
    gts: NDArrayFloat = np.array([[3.0, 4.0, 0.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]])

    expected_result = -5.0
    assert compute_affinity_matrix(dts, gts, AffinityType.CENTER) == expected_result


def test_translation_distance() -> None:
    """Initialize a detection and a ground truth label with only translation parameters.

    Verify that calculated distance matches expected distance under the specified `DistFnType`.
    """
    dts: NDArrayFloat = np.array([[0.0, 0.0, 0.0]])
    gts: NDArrayFloat = np.array([[5.0, 5.0, 5.0]])

    expected_result: float = np.sqrt(25 + 25 + 25)
    assert np.allclose(distance(dts, gts, DistanceType.TRANSLATION), expected_result)


def test_scale_distance() -> None:
    """Initialize a detection and a ground truth label with only shape parameters.

    Verify that calculated scale error matches the expected value.

    NOTE: We only provide shape parameters due to alignment assumption.
    """
    dts: NDArrayFloat = np.array([[5.0, 5.0, 5.0]])
    gts: NDArrayFloat = np.array([[10.0, 10.0, 10.0]])

    expected_result: float = 1 - 0.125
    assert np.allclose(distance(dts, gts, DistanceType.SCALE), expected_result)


def test_orientation_quarter_angles() -> None:
    """Initialize a detection and a ground truth label with only orientation parameters.

    Verify that calculated orientation error matches the expected smallest angle ((2 * PI) / 4)
    between the detection and ground truth label.
    """
    # Check all of the 90 degree angles
    expected_result: float = (2 * PI) / 4
    quarter_angles: List[NDArrayFloat] = [np.array([0, 0, angle]) for angle in np.arange(0, 2 * PI, expected_result)]

    for i in range(len(quarter_angles) - 1):
        quat_xyzw_dts: NDArrayFloat = Rotation.from_rotvec(quarter_angles[i : i + 1]).as_quat()
        quat_xyzw_gts: NDArrayFloat = Rotation.from_rotvec(quarter_angles[i + 1 : i + 2]).as_quat()

        quat_wxyz_dts = quat_xyzw_dts[..., [3, 0, 1, 2]]
        quat_wxyz_gts = quat_xyzw_gts[..., [3, 0, 1, 2]]

        assert np.isclose(distance(quat_wxyz_dts, quat_wxyz_gts, DistanceType.ORIENTATION), expected_result)
        assert np.isclose(distance(quat_wxyz_gts, quat_wxyz_dts, DistanceType.ORIENTATION), expected_result)


def test_orientation_eighth_angles() -> None:
    """Initialize a detection and a ground truth label with only orientation parameters.

    Verify that calculated orientation error matches the expected smallest angle ((2 * PI) / 8)
    between the detection and ground truth label.
    """
    expected_result: float = (2 * PI) / 8
    eigth_angles: List[NDArrayFloat] = [np.array([0, 0, angle]) for angle in np.arange(0, 2 * PI, expected_result)]

    for i in range(len(eigth_angles) - 1):
        quat_xyzw_dts = Rotation.from_rotvec(eigth_angles[i : i + 1]).as_quat()
        quat_xyzw_gts = Rotation.from_rotvec(eigth_angles[i + 1 : i + 2]).as_quat()

        quat_wxyz_dts = quat_xyzw_dts[..., [3, 0, 1, 2]]
        quat_wxyz_gts = quat_xyzw_gts[..., [3, 0, 1, 2]]

        assert np.isclose(distance(quat_wxyz_dts, quat_wxyz_gts, DistanceType.ORIENTATION), expected_result)
        assert np.isclose(distance(quat_wxyz_gts, quat_wxyz_dts, DistanceType.ORIENTATION), expected_result)


def test_wrap_angle() -> None:
    """Test mapping angles to a fixed interval."""
    theta: NDArrayFloat = np.array([-3 * PI / 2])
    expected_result: NDArrayFloat = np.array([PI / 2])
    assert np.isclose(wrap_angles(theta), expected_result)


def test_accumulate() -> None:
    """Verify that the accumulate function matches known output for a self-comparison."""
    cfg = DetectionCfg(eval_only_roi_instances=False)
    gts: pd.DataFrame = pd.read_feather(TEST_DATA_DIR / "labels.feather")

    for _, group in gts.groupby(["log_id", "timestamp_ns"]):
        job = (group.loc[:, CUBOID_COLS].to_numpy(), group.loc[:, CUBOID_COLS + ["num_interior_pts"]].to_numpy(), cfg)
        dts, gts = accumulate(*job)

        # Check that there's a true positive under every threshold.
        assert np.all(dts[:, :4])

        # Check that all error metrics are zero.
        assert (dts[:, 4:7] == 0).all()

        # # Check that there are 2 regular vehicles.
        # assert gts["category"].value_counts()["REGULAR_VEHICLE"] == 2

        # # Check that there are no other labels.
        # assert gts["category"].value_counts().sum() == 2


def test_assign() -> None:
    """Verify that the assign functions as expected by checking ATE of assigned detections against known distance."""
    cfg = DetectionCfg(eval_only_roi_instances=False)

    dts: NDArrayFloat = np.array(
        [
            [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [20.0, 20.0, 20.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    gts: NDArrayFloat = np.array(
        [
            [-10.0, -10.0, -10.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [10.1, 10.0, 10.0, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    dts_assignments, _ = assign(dts, gts, cfg)
    # if these assign correctly, we should get an ATE of 0.1 for the first two
    expected_result: float = 0.1
    ATE_COL_IDX = 4

    assert math.isclose(dts_assignments[0, ATE_COL_IDX], expected_result)  # instance 0
    assert math.isclose(dts_assignments[1, ATE_COL_IDX], expected_result)  # instance 1
    assert math.isclose(dts_assignments[2, ATE_COL_IDX], 2.0)  # instance 32


def test_interp() -> None:
    """Test non-decreasing `interpolation` constraint enforced on precision results."""
    prec: NDArrayFloat = np.array([1.0, 0.5, 0.33, 0.5])
    expected_result: NDArrayFloat = np.array([1.0, 0.5, 0.5, 0.5])
    assert np.isclose(interpolate_precision(prec), expected_result).all()


def test_iou_aligned_3d() -> None:
    """Initialize a detection and a ground truth label with only shape parameters.

    Verify that calculated intersection-over-union matches the expected
    value between the detection and ground truth label.

    NOTE: We only provide shape parameters due to alignment assumption.
    """
    columns = DIMS_COLS
    dt_dims: NDArrayFloat = pd.DataFrame([[4.0, 10.0, 3.0]], columns=columns).to_numpy()
    gt_dims: NDArrayFloat = pd.DataFrame([[9.0, 5.0, 2.0]], columns=columns).to_numpy()

    # Intersection is 40 = 4 * 5 * 2 (min of all dimensions).
    # Union is the sum of the two volumes, minus intersection: 270 = (10 * 3 * 4) + (5 * 2 * 9) - 40.
    expected_result: float = 40 / 270.0
    assert iou_3d_axis_aligned(dt_dims, gt_dims) == expected_result


def test_assignment() -> None:
    """Verify that assignment works as expected; should have one duplicate in the provided results."""
    expected_result: float = 0.999
    assert _get_summary_assignment().loc["AVERAGE_METRICS", "AP"] == expected_result


def test_ap() -> None:
    """Test that AP is 1 for the self-compared results."""
    expected_result: float = 1.0
    assert _get_summary_identity().loc["AVERAGE_METRICS", "AP"] == expected_result


def test_translation_error() -> None:
    """Test that ATE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.017  # 0.1 / 6, one of six dets is off by 0.1
    assert _get_summary_identity().loc["AVERAGE_METRICS", "ATE"] == expected_result_identity
    assert _get_summary().loc["AVERAGE_METRICS", "ATE"] == expected_result_det


def test_scale_error() -> None:
    """Test that ASE is 0 for the self-compared results."""
    expected_result_identity: float = 0.0
    expected_result_det: float = 0.033  # 0.2 / 6, one of six dets is off by 20% in IoU
    assert _get_summary_identity().loc["AVERAGE_METRICS", "ASE"] == expected_result_identity
    assert _get_summary().loc["AVERAGE_METRICS", "ASE"] == expected_result_det


def test_orientation_error() -> None:
    """Test that AOE is 0 for the self-compared results."""
    expected_result_identity = 0.0
    expected_result_det = 0.524  # pi / 6, since one of six dets is off by pi

    assert _get_summary_identity().loc["AVERAGE_METRICS", "AOE"] == expected_result_identity
    assert _get_summary().loc["AVERAGE_METRICS", "AOE"] == expected_result_det


def test_compute_evaluated_dts_mask() -> None:
    """Unit test for computing valid detections cuboids."""
    dts: NDArrayFloat = np.array(
        [
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0],  # In bounds with at least 1 point.
            [175, 175.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0],  # Out of bounds with at least 1 point.
            [-175.0, -175.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0],  # Out of bounds with at least 1 point.
            [1.0, 1.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0],  # In bounds with at least 1 point.
        ],
    )
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    dts_mask = compute_evaluated_dts_mask(dts, detection_cfg)
    dts_mask_: NDArrayBool = np.array([True, False, False, True])
    np.testing.assert_array_equal(dts_mask, dts_mask_)


def test_compute_evaluated_dts_mask_2() -> None:
    """Randomly generate detections and ensure that they never exceed the maximum detection limit."""
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    for i in range(1000):
        dts: NDArrayFloat = np.random.randint(0, 250, size=(detection_cfg.max_num_dts_per_category + i, 10)).astype(
            float
        )
        dts_mask = compute_evaluated_dts_mask(dts, detection_cfg)
        assert dts_mask.sum() <= detection_cfg.max_num_dts_per_category


def test_compute_evaluated_gts_mask() -> None:
    """Unit test for computing valid ground truth cuboids."""
    gts: NDArrayFloat = np.array(
        [
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 5],  # In bounds with at least 1 point.
            [175, 175.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 5],  # Out of bounds with at least 1 point.
            [-175.0, -175.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 5],  # Out of bounds with at least 1 point.
            [1.0, 1.0, 5.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0],  # In bounds with at least 1 point.
        ],
    )
    detection_cfg = DetectionCfg(categories=("REGULAR_VEHICLE",), eval_only_roi_instances=False)
    gts_xyz_ego = gts[..., :3]
    num_interior_pts = gts[..., -1].astype(int)

    gts_mask = compute_evaluated_gts_mask(gts_xyz_ego, num_interior_pts, detection_cfg)
    gts_mask_: NDArrayBool = np.array([True, False, False, False])
    np.testing.assert_array_equal(gts_mask, gts_mask_)


def test_compute_objects_in_roi_mask() -> None:
    """Test filtering ground truth annotations by the ROI.

    Three annotations are tested. The first and third are partially in the ROI --- these
    are both considered VALID since they have at least on of their cuboid vertices within
    the ROI.

    The second annotations is _FULLY_ outside of the ROI, thus it is filtered.
    """
    map_dir = TEST_DATA_DIR / "adcf7d18-0510-35b0-a2fa-b4cea13a6d76" / "map"

    timestamp_ns = 315973157959879000
    track_uuids = [
        "f53639ef-794e-420e-bb2a-d0cde0203b3a",  # Two vertices within ROI.
        "6c198de2-cb7d-4c09-96aa-52547d9bbe37",  # Completely outside of ROI.
        "a7c8f6a2-26b6-4610-9eb3-294799f9846c",  # Two vertices within ROI.
    ]
    avm = ArgoverseStaticMap.from_map_dir(map_dir, build_raster=True)
    annotations = read_feather(TEST_DATA_DIR / "adcf7d18-0510-35b0-a2fa-b4cea13a6d76" / "annotations.feather")
    timestamped_city_SE3_egoposes = read_city_SE3_ego(TEST_DATA_DIR / "adcf7d18-0510-35b0-a2fa-b4cea13a6d76")

    selected_cuboids_mask = np.logical_and(
        annotations.timestamp_ns == timestamp_ns, annotations["track_uuid"].isin(track_uuids)
    )
    sweep_annotations = annotations.loc[selected_cuboids_mask]

    mask = compute_objects_in_roi_mask(
        sweep_annotations.loc[:, ORDERED_CUBOID_COL_NAMES].to_numpy(), timestamped_city_SE3_egoposes[timestamp_ns], avm
    )
    mask_: NDArrayBool = np.array([True, False, True])
    np.testing.assert_array_equal(mask, mask_)
