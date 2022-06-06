# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Utilities to evaluate motion forecasting predictions and compute metrics."""

import numpy as np

from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayNumber


def compute_ade(forecasted_trajectories: NDArrayNumber, gt_trajectory: NDArrayNumber) -> NDArrayFloat:
    """Compute the average displacement error for a set of K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory.

    Returns:
        (K,) Average displacement error for each of the predicted trajectories.
    """
    displacement_errors = np.linalg.norm(forecasted_trajectories - gt_trajectory, axis=2)
    ade: NDArrayFloat = np.mean(displacement_errors, axis=1)
    return ade


def compute_fde(forecasted_trajectories: NDArrayNumber, gt_trajectory: NDArrayNumber) -> NDArrayFloat:
    """Compute the final displacement error for a set of K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory, FDE will be evaluated against true position at index `N-1`.

    Returns:
        (K,) Final displacement error for each of the predicted trajectories.
    """
    # Compute final displacement error for all K trajectories
    fde_vector = (forecasted_trajectories - gt_trajectory)[:, -1]
    fde: NDArrayFloat = np.linalg.norm(fde_vector, axis=-1)
    return fde


def compute_is_missed_prediction(
    forecasted_trajectories: NDArrayNumber,
    gt_trajectory: NDArrayNumber,
    miss_threshold_m: float = 2.0,
) -> NDArrayBool:
    """Compute whether each of K predicted trajectories (for the same actor) missed by more than a distance threshold.

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory, miss will be evaluated against true position at index `N-1`.
        miss_threshold_m: Minimum distance threshold for final displacement to be considered a miss.

    Returns:
        (K,) Bools indicating whether prediction missed by more than specified threshold.
    """
    fde = compute_fde(forecasted_trajectories, gt_trajectory)
    is_missed_prediction = fde > miss_threshold_m
    return is_missed_prediction


def compute_brier_ade(
    forecasted_trajectories: NDArrayNumber,
    gt_trajectory: NDArrayNumber,
    forecast_probabilities: NDArrayNumber,
    normalize: bool = False,
) -> NDArrayFloat:
    """Compute a probability-weighted (using Brier score) ADE for K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory.
        forecast_probabilities: (K,) probabilities associated with each prediction.
        normalize: Normalizes `forecast_probabilities` to sum to 1 when set to True.

    Returns:
        (K,) Probability-weighted average displacement error for each predicted trajectory.
    """
    # Compute ADE with Brier score component
    brier_score = _compute_brier_score(forecasted_trajectories, forecast_probabilities, normalize)
    ade_vector = compute_ade(forecasted_trajectories, gt_trajectory)
    brier_ade: NDArrayFloat = ade_vector + brier_score
    return brier_ade


def compute_brier_fde(
    forecasted_trajectories: NDArrayNumber,
    gt_trajectory: NDArrayNumber,
    forecast_probabilities: NDArrayNumber,
    normalize: bool = False,
) -> NDArrayFloat:
    """Compute a probability-weighted (using Brier score) FDE for K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory.
        forecast_probabilities: (K,) probabilities associated with each prediction.
        normalize: Normalizes `forecast_probabilities` to sum to 1 when set to True.

    Returns:
        (K,) Probability-weighted final displacement error for each predicted trajectory.
    """
    # Compute FDE with Brier score component
    brier_score = _compute_brier_score(forecasted_trajectories, forecast_probabilities, normalize)
    fde_vector = compute_fde(forecasted_trajectories, gt_trajectory)
    brier_fde: NDArrayFloat = fde_vector + brier_score
    return brier_fde


def _compute_brier_score(
    forecasted_trajectories: NDArrayNumber,
    forecast_probabilities: NDArrayNumber,
    normalize: bool = False,
) -> NDArrayFloat:
    """Compute Brier score for K predicted trajectories.

    Note: This function computes Brier score under the assumption that each trajectory is the true "best" prediction
    (i.e. each predicted trajectory has a ground truth probability of 1.0).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        forecast_probabilities: (K,) probabilities associated with each prediction.
        normalize: Normalizes `forecast_probabilities` to sum to 1 when set to True.

    Raises:
        ValueError: If the number of forecasted trajectories and probabilities don't match.
        ValueError: If normalize=False and `forecast_probabilities` contains values outside of the range [0, 1].

    Returns:
        (K,) Brier score for each predicted trajectory.
    """
    # Validate that # of forecast probabilities matches forecasted trajectories
    if len(forecasted_trajectories) != len(forecast_probabilities):
        raise ValueError()

    # Validate that all forecast probabilities are in the range [0, 1]
    if np.logical_or(forecast_probabilities < 0.0, forecast_probabilities > 1.0).any():
        raise ValueError("At least one forecast probability falls outside the range [0, 1].")

    # If enabled, normalize forecast probabilities to sum to 1
    if normalize:
        forecast_probabilities = forecast_probabilities / np.sum(forecast_probabilities)

    brier_score: NDArrayFloat = np.square((1 - forecast_probabilities))
    return brier_score
