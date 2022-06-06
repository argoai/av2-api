# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Detection utilities for the Argoverse detection leaderboard.

Accepts detections (in Argoverse ground truth format) and ground truth labels
for computing evaluation metrics for 3d object detection. We have five different,
metrics: mAP, ATE, ASE, AOE, and CDS. A true positive for mAP is defined as the
highest confidence prediction within a specified Euclidean distance threshold
from a bird's-eye view. We prefer these metrics instead of IoU due to the
increased interpretability of the error modes in a set of detections.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

from av2.evaluation.detection.constants import (
    MAX_NORMALIZED_ASE,
    MAX_SCALE_ERROR,
    MAX_YAW_RAD_ERROR,
    MIN_AP,
    MIN_CDS,
    AffinityType,
    CompetitionCategories,
    DistanceType,
    FilterMetricType,
    InterpType,
)
from av2.geometry.geometry import mat_to_xyz, quat_to_mat, wrap_angles
from av2.geometry.iou import iou_3d_axis_aligned
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.constants import EPS
from av2.utils.io import TimestampedCitySE3EgoPoses, read_city_SE3_ego
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionCfg:
    """Instantiates a DetectionCfg object for configuring a evaluation.

    Args:
        affinity_thresholds_m: Affinity thresholds for determining a true positive (in meters).
        affinity_type: Type of affinity function to be used for calculating average precision.
        categories: Detection classes for evaluation.
        eval_only_roi_instances: Only use detections and ground truth annotations that lie
            within region of interest during eval.
        filter_metric: Detection metric to use for filtering of both detections and ground truth annotations.
        max_range_m: Max distance (under a specific metric in meters) for a detection or ground truth cuboid to be
            considered for evaluation.
        num_recall_samples: Number of recall points to sample uniformly in [0, 1].
        tp_threshold_m: Center distance threshold for the true positive metrics (in meters).
    """

    affinity_thresholds_m: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    affinity_type: AffinityType = AffinityType.CENTER
    categories: Tuple[str, ...] = tuple(x.value for x in CompetitionCategories)
    dataset_dir: Optional[Path] = None
    eval_only_roi_instances: bool = True
    filter_metric: FilterMetricType = FilterMetricType.EUCLIDEAN
    max_num_dts_per_category: int = 100
    max_range_m: float = 150.0
    num_recall_samples: int = 100
    tp_threshold_m: float = 2.0

    @property
    def metrics_defaults(self) -> Tuple[float, ...]:
        """Return the evaluation summary default values."""
        return (
            MIN_AP,
            self.tp_threshold_m,
            MAX_NORMALIZED_ASE,
            MAX_YAW_RAD_ERROR,
            MIN_CDS,
        )

    @property
    def tp_normalization_terms(self) -> Tuple[float, ...]:
        """Return the normalization constants for ATE, ASE, and AOE."""
        return (
            self.tp_threshold_m,
            MAX_SCALE_ERROR,
            MAX_YAW_RAD_ERROR,
        )


def accumulate(
    dts: NDArrayFloat,
    gts: NDArrayFloat,
    cfg: DetectionCfg,
    avm: Optional[ArgoverseStaticMap] = None,
    city_SE3_ego: Optional[SE3] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Accumulate the true / false positives (boolean flags) and true positive errors for each class.

    The detections (gts) and ground truth annotations (gts) are expected to be shape (N,11) and (M,11)
    respectively. Their _ordered_ columns are shown below:

    dts: tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz, score.
    gts: tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz, num_interior_pts.

    NOTE: The columns for dts and gts only differ by their _last_ column. Score represents the
        "confidence" of the detection and `num_interior_pts` are the number of points interior
        to the ground truth cuboid at the time of annotation.

    Args:
        dts: (N,11) Detections array.
        gts: (M,11) Ground truth annotations array.
        cfg: 3D object detection configuration.
        avm: Argoverse static map for the log.
        city_SE3_ego: Egovehicle pose in the city reference frame.

    Returns:
        (N,11+T+E+1) Augmented detections.
        (M,11+T+E+1) Augmented ground truth annotations.
        NOTE: The $$T+E+1$$ additional columns consist of the following:
            $$T$$: cfg.affinity_thresholds_m (0.5, 1.0, 2.0, 4.0 by default).
            $$E$$: ATE, ASE, AOE.
            1: `is_evaluated` flag indicating whether the detections or ground truth annotations
                are considered during assignment.
    """
    N, M = len(dts), len(gts)
    T, E = len(cfg.affinity_thresholds_m), 3

    # Sort the detections by score in _descending_ order.
    scores: NDArrayFloat = dts[..., -1]
    permutation: NDArrayInt = np.argsort(-scores).tolist()
    dts = dts[permutation]

    is_evaluated_dts: NDArrayBool = np.ones(N, dtype=bool)
    is_evaluated_gts: NDArrayBool = np.ones(M, dtype=bool)
    if avm is not None and city_SE3_ego is not None:
        is_evaluated_dts &= compute_objects_in_roi_mask(dts, city_SE3_ego, avm)
        is_evaluated_gts &= compute_objects_in_roi_mask(gts, city_SE3_ego, avm)

    is_evaluated_dts &= compute_evaluated_dts_mask(dts[..., :3], cfg)
    is_evaluated_gts &= compute_evaluated_gts_mask(gts[..., :3], gts[..., -1].astype(int), cfg)

    # Initialize results array.
    dts_augmented: NDArrayFloat = np.zeros((N, T + E + 1))
    gts_augmented: NDArrayFloat = np.zeros((M, T + E + 1))

    # `is_evaluated` boolean flag is always the last column of the array.
    dts_augmented[is_evaluated_dts, -1] = True
    gts_augmented[is_evaluated_gts, -1] = True

    if is_evaluated_dts.sum() > 0 and is_evaluated_gts.sum() > 0:
        # Compute true positives by assigning detections and ground truths.
        dts_assignments, gts_assignments = assign(dts[is_evaluated_dts], gts[is_evaluated_gts], cfg)
        dts_augmented[is_evaluated_dts, :-1] = dts_assignments
        gts_augmented[is_evaluated_gts, :-1] = gts_assignments

    # Permute the detections according to the original ordering.
    outputs: Tuple[NDArrayInt, NDArrayInt] = np.unique(permutation, return_index=True)
    _, inverse_permutation = outputs
    dts_augmented = dts_augmented[inverse_permutation]
    return dts_augmented, gts_augmented


def assign(dts: NDArrayFloat, gts: NDArrayFloat, cfg: DetectionCfg) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Attempt assignment of each detection to a ground truth label.

    The detections (gts) and ground truth annotations (gts) are expected to be shape (N,10) and (M,10)
    respectively. Their _ordered_ columns are shown below:

    dts: tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz.
    gts: tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz.

    NOTE: The columns for dts and gts only differ by their _last_ column. Score represents the
        "confidence" of the detection and `num_interior_pts` are the number of points interior
        to the ground truth cuboid at the time of annotation.

    Args:
        dts: (N,10) Detections array.
        gts: (M,10) Ground truth annotations array.
        cfg: 3D object detection configuration.

    Returns:
        (N,T+E) Detections metrics table.
        (M,T+E) Ground truth annotations metrics table.
        NOTE: The $$T+E$$ additional columns consist of the following:
            $$T$$: cfg.affinity_thresholds_m (0.5, 1.0, 2.0, 4.0 by default).
            $$E$$: ATE, ASE, AOE.
    """
    affinity_matrix = compute_affinity_matrix(dts[..., :3], gts[..., :3], cfg.affinity_type)

    # Get the GT label for each max-affinity GT label, detection pair.
    idx_gts: NDArrayInt = affinity_matrix.argmax(axis=1)[None]

    # The affinity matrix is an N by M matrix of the detections and ground truth labels respectively.
    # We want to take the corresponding affinity for each of the initial assignments using `gt_matches`.
    # The following line grabs the max affinity for each detection to a ground truth label.
    affinities: NDArrayFloat = np.take_along_axis(affinity_matrix.transpose(), idx_gts, axis=0)[0]

    # Find the indices of the _first_ detection assigned to each GT.
    assignments: Tuple[NDArrayInt, NDArrayInt] = np.unique(idx_gts, return_index=True)
    idx_gts, idx_dts = assignments

    T, E = len(cfg.affinity_thresholds_m), 3
    dts_metrics: NDArrayFloat = np.zeros((len(dts), T + E))
    dts_metrics[:, 4:] = cfg.metrics_defaults[1:4]
    gts_metrics: NDArrayFloat = np.zeros((len(gts), T + E))
    gts_metrics[:, 4:] = cfg.metrics_defaults[1:4]
    for i, threshold_m in enumerate(cfg.affinity_thresholds_m):
        is_tp: NDArrayBool = affinities[idx_dts] > -threshold_m

        dts_metrics[idx_dts[is_tp], i] = True
        gts_metrics[idx_gts, i] = True

        if threshold_m != cfg.tp_threshold_m:
            continue  # Skip if threshold isn't the true positive threshold.
        if not np.any(is_tp):
            continue  # Skip if no true positives exist.

        idx_tps_dts: NDArrayInt = idx_dts[is_tp]
        idx_tps_gts: NDArrayInt = idx_gts[is_tp]

        tps_dts = dts[idx_tps_dts]
        tps_gts = gts[idx_tps_gts]

        translation_errors = distance(tps_dts[:, :3], tps_gts[:, :3], DistanceType.TRANSLATION)
        scale_errors = distance(tps_dts[:, 3:6], tps_gts[:, 3:6], DistanceType.SCALE)
        orientation_errors = distance(tps_dts[:, 6:10], tps_gts[:, 6:10], DistanceType.ORIENTATION)
        dts_metrics[idx_tps_dts, 4:] = np.stack((translation_errors, scale_errors, orientation_errors), axis=-1)
    return dts_metrics, gts_metrics


def interpolate_precision(precision: NDArrayFloat, interpolation_method: InterpType = InterpType.ALL) -> NDArrayFloat:
    r"""Interpolate the precision at each sampled recall.

    This function smooths the precision-recall curve according to the method introduced in Pascal
    VOC:

    Mathematically written as:
        $$p_{\text{interp}}(r) = \max_{\tilde{r}: \tilde{r} \geq r} p(\tilde{r})$$

    See equation 2 in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
        for more information.

    Args:
        precision: Precision at all recall levels (N,).
        interpolation_method: Accumulation method.

    Returns:
        (N,) The interpolated precision at all sampled recall levels.

    Raises:
        NotImplementedError: If the interpolation method is not implemented.
    """
    precision_interpolated: NDArrayFloat
    if interpolation_method == InterpType.ALL:
        precision_interpolated = np.maximum.accumulate(precision[::-1])[::-1]
    else:
        raise NotImplementedError("This interpolation method is not implemented!")
    return precision_interpolated


def compute_affinity_matrix(dts: NDArrayFloat, gts: NDArrayFloat, metric: AffinityType) -> NDArrayFloat:
    """Calculate the affinity matrix between detections and ground truth annotations.

    Args:
        dts: (N,K) Detections.
        gts: (M,K) Ground truth annotations.
        metric: Affinity metric type.

    Returns:
        The affinity scores between detections and ground truth annotations (N,M).

    Raises:
        NotImplementedError: If the affinity metric is not implemented.
    """
    if metric == AffinityType.CENTER:
        dts_xy_m = dts
        gts_xy_m = gts
        affinities: NDArrayFloat = -cdist(dts_xy_m, gts_xy_m)
    else:
        raise NotImplementedError("This affinity metric is not implemented!")
    return affinities


def compute_average_precision(
    tps: NDArrayBool, recall_interpolated: NDArrayFloat, num_gts: int
) -> Tuple[float, NDArrayFloat]:
    """Compute precision and recall, interpolated over N fixed recall points.

    Args:
        tps: True positive detections (ranked by confidence).
        recall_interpolated: Interpolated recall values.
        num_gts: Number of annotations of this class.

    Returns:
        The average precision and interpolated precision values.
    """
    cum_tps: NDArrayInt = np.cumsum(tps)
    cum_fps: NDArrayInt = np.cumsum(~tps)
    cum_fns: NDArrayInt = num_gts - cum_tps

    # Compute precision.
    precision: NDArrayFloat = cum_tps / (cum_tps + cum_fps + EPS)

    # Compute recall.
    recall: NDArrayFloat = cum_tps / (cum_tps + cum_fns)

    # Interpolate precision -- VOC-style.
    precision = interpolate_precision(precision)

    # Evaluate precision at different recalls.
    precision_interpolated: NDArrayFloat = np.interp(recall_interpolated, recall, precision, right=0)

    average_precision: float = np.mean(precision_interpolated)
    return average_precision, precision_interpolated


def distance(dts: NDArrayFloat, gts: NDArrayFloat, metric: DistanceType) -> NDArrayFloat:
    """Distance functions between detections and ground truth.

    Args:
        dts: (N,D) Detections where D is the number of attributes.
        gts: (N,D) Ground truth labels where D is the number of attributes.
        metric: Distance function type.

    Returns:
        (N,) Distance between the detections and ground truth under the specified metric.

    Raises:
        NotImplementedError: If the distance type is not supported.
    """
    if metric == DistanceType.TRANSLATION:
        translation_errors: NDArrayFloat = np.linalg.norm(dts - gts, axis=1)
        return translation_errors
    elif metric == DistanceType.SCALE:
        scale_errors: NDArrayFloat = 1 - iou_3d_axis_aligned(dts, gts)
        return scale_errors
    elif metric == DistanceType.ORIENTATION:
        yaws_dts: NDArrayFloat = mat_to_xyz(quat_to_mat(dts))[..., 2]
        yaws_gts: NDArrayFloat = mat_to_xyz(quat_to_mat(gts))[..., 2]
        orientation_errors = wrap_angles(yaws_dts - yaws_gts)
        return orientation_errors
    else:
        raise NotImplementedError("This distance metric is not implemented!")


def compute_objects_in_roi_mask(cuboids_ego: NDArrayFloat, city_SE3_ego: SE3, avm: ArgoverseStaticMap) -> NDArrayBool:
    """Compute the evaluated cuboids mask based off whether _any_ of their vertices fall into the ROI.

    Args:
        cuboids_ego: (N,10) Array of cuboid parameters corresponding to `ORDERED_CUBOID_COL_NAMES`.
        city_SE3_ego: Egovehicle pose in the city reference frame.
        avm: Argoverse map object.

    Returns:
        (N,) Boolean mask indicating which cuboids will be evaluated.
    """
    is_within_roi: NDArrayBool
    if len(cuboids_ego) == 0:
        is_within_roi = np.zeros((0,), dtype=bool)
        return is_within_roi
    cuboid_list_ego: CuboidList = CuboidList([Cuboid.from_numpy(params) for params in cuboids_ego])
    cuboid_list_city = cuboid_list_ego.transform(city_SE3_ego)
    cuboid_list_vertices_m_city = cuboid_list_city.vertices_m

    is_within_roi = avm.get_raster_layer_points_boolean(
        cuboid_list_vertices_m_city.reshape(-1, 3)[..., :2], RasterLayerType.ROI
    )
    is_within_roi = is_within_roi.reshape(-1, 8)
    is_within_roi = is_within_roi.any(axis=1)
    return is_within_roi


def compute_evaluated_dts_mask(
    xyz_m_ego: NDArrayFloat,
    cfg: DetectionCfg,
) -> NDArrayBool:
    """Compute the evaluated cuboids mask.

    Valid detections cuboids meet _two_ conditions:
        1. The cuboid's centroid (x,y,z) must lie within the maximum range
            defined in the detection configuration.
        2. The total number of cuboids must not exceed `cfg.max_num_dts_per_category`.

    Args:
        xyz_m_ego: (N,3) Center of the detections in the egovehicle frame.
        cfg: 3D object detection configuration.

    Returns:
        The boolean mask indicating which cuboids will be evaluated.
    """
    is_evaluated: NDArrayBool
    if len(xyz_m_ego) == 0:
        is_evaluated = np.zeros((0,), dtype=bool)
        return is_evaluated
    norm: NDArrayFloat = np.linalg.norm(xyz_m_ego, axis=1)
    is_evaluated = norm < cfg.max_range_m

    cumsum: NDArrayInt = np.cumsum(is_evaluated)
    max_idx_arr: NDArrayInt = np.where(cumsum > cfg.max_num_dts_per_category)[0]
    if len(max_idx_arr) > 0:
        max_idx = max_idx_arr[0]
        is_evaluated[max_idx:] = False
    return is_evaluated


def compute_evaluated_gts_mask(
    xyz_m_ego: NDArrayFloat,
    num_interior_pts: NDArrayInt,
    cfg: DetectionCfg,
) -> NDArrayBool:
    """Compute the ground truth annotations evaluated cuboids mask.

    Valid detections cuboids meet _two_ conditions:
        1. The cuboid's centroid (x,y,z) must lie within the maximum range in the detection configuration.
        2. The cuboid must have at _least_ one point in each cuboid.

    Args:
        xyz_m_ego: (M,3) Center of the ground truth annotations in the egovehicle frame.
        num_interior_pts: (M,) Number of points interior to each cuboid.
        cfg: 3D object detection configuration.

    Returns:
        The boolean mask indicating which cuboids will be evaluated.
    """
    is_evaluated: NDArrayBool
    if len(xyz_m_ego) == 0:
        is_evaluated = np.zeros((0,), dtype=bool)
        return is_evaluated
    norm: NDArrayFloat = np.linalg.norm(xyz_m_ego, axis=1)
    is_evaluated = np.logical_and(norm < cfg.max_range_m, num_interior_pts > 0)
    return is_evaluated


def load_mapped_avm_and_egoposes(
    log_ids: List[str], dataset_dir: Path
) -> Tuple[Dict[str, ArgoverseStaticMap], Dict[str, TimestampedCitySE3EgoPoses]]:
    """Load the maps and egoposes for each log in the dataset directory.

    Args:
        log_ids: List of the log_ids.
        dataset_dir: Directory to the dataset.

    Returns:
        A tuple of mappings from log id to maps and timestamped-egoposes, respectively.

    Raises:
        RuntimeError: If the process for loading maps and timestamped egoposes fails.
    """
    log_id_to_timestamped_poses = {log_id: read_city_SE3_ego(dataset_dir / log_id) for log_id in log_ids}
    avms: Optional[List[ArgoverseStaticMap]] = Parallel(n_jobs=-1, backend="threading", verbose=1)(
        delayed(ArgoverseStaticMap.from_map_dir)(dataset_dir / log_id / "map", build_raster=True) for log_id in log_ids
    )
    if avms is None:
        raise RuntimeError("Map and egopose loading has failed!")
    log_id_to_avm = {log_ids[i]: avm for i, avm in enumerate(avms)}
    return log_id_to_avm, log_id_to_timestamped_poses


def groupby(names: List[str], values: NDArrayFloat) -> Dict[str, NDArrayFloat]:
    """Group a set of values by their corresponding names.

    Args:
        names: String which maps data to a "bin".
        values: Data which will be grouped by their names.

    Returns:
        Dictionary mapping the group name to the corresponding group.
    """
    outputs: Tuple[NDArrayInt, NDArrayInt] = np.unique(names, return_index=True)
    unique_items, unique_items_indices = outputs
    dts_groups: List[NDArrayFloat] = np.split(values, unique_items_indices[1:])
    uuid_to_groups = {unique_items[i]: x for i, x in enumerate(dts_groups)}
    return uuid_to_groups
