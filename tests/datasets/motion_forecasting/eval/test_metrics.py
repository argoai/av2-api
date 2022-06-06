# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for motion forecasting metrics."""

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Final

import numpy as np
import pytest

import av2.datasets.motion_forecasting.eval.metrics as metrics
from av2.utils.typing import NDArrayFloat, NDArrayNumber

# Build stationary GT trajectory at (0, 0)
test_N: Final[int] = 10
_STATIONARY_GT_TRAJ = np.zeros((test_N, 2))

# Case 1: K=1 forecast stationary at (1, 1)
forecasted_trajectories_stationary_k1 = np.ones((1, test_N, 2))
expected_ade_stationary_k1 = np.full((1,), np.sqrt(2))
expected_fde_stationary_k1 = np.full((1,), np.sqrt(2))

# Case 2: K=6 forecasts stationary at (1, 1)
forecasted_trajectories_stationary_k6 = np.ones((6, test_N, 2))
expected_ade_stationary_k6 = np.full((6,), np.sqrt(2))
expected_fde_stationary_k6 = np.full((6,), np.sqrt(2))

# Case 3: K=1 forecast in straight line on X axis
forecasted_trajectories_straight_k1 = np.stack([np.arange(test_N), np.zeros(test_N)], axis=1)[np.newaxis, ...]  # 1xNx2
expected_ade_straight_k1 = np.full((1,), np.arange(test_N).mean())
expected_fde_straight_k1 = np.full((1,), test_N - 1)

# Case 4: K=6 forecasts in straight line on X axis
forecasted_trajectories_straight_k6: NDArrayFloat = np.concatenate(
    [forecasted_trajectories_straight_k1] * 6, axis=0
)  # 6xNx2
expected_ade_straight_k6 = np.full((6,), np.arange(test_N).mean())
expected_fde_straight_k6 = np.full((6,), test_N - 1)

# Case 5: K=1 forecast in diagonal line
forecasted_trajectories_diagonal_k1 = np.stack([np.arange(test_N), np.arange(test_N)], axis=1)[np.newaxis, ...]  # 1xNx2
expected_ade_diagonal_k1 = np.full((1,), 6.36396103)
expected_fde_diagonal_k1 = np.full((1,), np.hypot(test_N - 1, test_N - 1))


@pytest.mark.parametrize(
    "forecasted_trajectories, expected_ade",
    [
        (forecasted_trajectories_stationary_k1, expected_ade_stationary_k1),
        (forecasted_trajectories_stationary_k6, expected_ade_stationary_k6),
        (forecasted_trajectories_straight_k1, expected_ade_straight_k1),
        (forecasted_trajectories_straight_k6, expected_ade_straight_k6),
        (forecasted_trajectories_diagonal_k1, expected_ade_diagonal_k1),
    ],
    ids=["stationary_k1", "stationary_k6", "straight_k1", "straight_k6", "diagonal_k1"],
)
def test_compute_ade(forecasted_trajectories: NDArrayNumber, expected_ade: NDArrayFloat) -> None:
    """Test that compute_ade returns the correct output with valid inputs.

    Args:
        forecasted_trajectories: Forecasted trajectories for test case.
        expected_ade: Expected average displacement error.
    """
    ade = metrics.compute_ade(forecasted_trajectories, _STATIONARY_GT_TRAJ)
    np.testing.assert_allclose(ade, expected_ade)


@pytest.mark.parametrize(
    "forecasted_trajectories, expected_fde",
    [
        (forecasted_trajectories_stationary_k1, expected_fde_stationary_k1),
        (forecasted_trajectories_stationary_k6, expected_fde_stationary_k6),
        (forecasted_trajectories_straight_k1, expected_fde_straight_k1),
        (forecasted_trajectories_straight_k6, expected_fde_straight_k6),
        (forecasted_trajectories_diagonal_k1, expected_fde_diagonal_k1),
    ],
    ids=["stationary_k1", "stationary_k6", "straight_k1", "straight_k6", "diagonal_k1"],
)
def test_compute_fde(forecasted_trajectories: NDArrayNumber, expected_fde: NDArrayFloat) -> None:
    """Test that compute_fde returns the correct output with valid inputs.

    Args:
        forecasted_trajectories: Forecasted trajectories for test case.
        expected_fde: Expected final displacement error.
    """
    fde = metrics.compute_fde(forecasted_trajectories, _STATIONARY_GT_TRAJ)
    assert np.array_equal(fde, expected_fde)


@pytest.mark.parametrize(
    "forecasted_trajectories, miss_threshold_m, expected_is_missed_label",
    [
        (forecasted_trajectories_stationary_k1, 2.0, False),
        (forecasted_trajectories_stationary_k6, 2.0, False),
        (forecasted_trajectories_straight_k1, expected_fde_straight_k1[0] + 0.01, False),
        (forecasted_trajectories_straight_k1, expected_fde_straight_k1[0] - 0.01, True),
        (forecasted_trajectories_diagonal_k1, 2.0, True),
    ],
    ids=["stationary_k1", "stationary_k6", "straight_below_threshold", "straight_above_threshold", "diagonal"],
)
def test_compute_is_missed_prediction(
    forecasted_trajectories: NDArrayNumber, miss_threshold_m: float, expected_is_missed_label: bool
) -> None:
    """Test that compute_is_missed_prediction returns the correct output with valid inputs.

    Args:
        forecasted_trajectories: Forecasted trajectories for test case.
        miss_threshold_m: Minimum distance threshold for final displacement to be considered a miss.
        expected_is_missed_label: Expected is_missed label for test case.
    """
    is_missed_prediction = metrics.compute_is_missed_prediction(
        forecasted_trajectories, _STATIONARY_GT_TRAJ, miss_threshold_m
    )

    # Check that is_missed labels are of the correct shape and have the correct value
    assert is_missed_prediction.shape == forecasted_trajectories.shape[:1]
    assert np.all(is_missed_prediction == expected_is_missed_label)


uniform_probabilities_k1: NDArrayFloat = np.ones((1,))
uniform_probabilities_k6: NDArrayFloat = np.ones((6,)) / 6
confident_probabilities_k6: NDArrayFloat = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
non_normalized_probabilities_k6: NDArrayFloat = confident_probabilities_k6 / 10
out_of_range_probabilities_k6: NDArrayFloat = confident_probabilities_k6 * 10
wrong_shape_probabilities_k6: NDArrayFloat = np.ones((5,)) / 5

expected_bade_uniform_k1 = expected_ade_stationary_k1
expected_bade_uniform_k6 = expected_ade_straight_k6 + np.square((1 - uniform_probabilities_k6))
expected_bade_confident_k6 = expected_ade_straight_k6 + np.square((1 - confident_probabilities_k6))

expected_bfde_uniform_k1 = expected_fde_stationary_k1
expected_bfde_uniform_k6 = expected_fde_straight_k6 + np.square((1 - uniform_probabilities_k6))
expected_bfde_confident_k6 = expected_fde_straight_k6 + np.square((1 - confident_probabilities_k6))


@pytest.mark.parametrize(
    "forecasted_trajectories, forecast_probabilities, normalize, expected_brier_ade",
    [
        (forecasted_trajectories_stationary_k1, uniform_probabilities_k1, False, expected_bade_uniform_k1),
        (forecasted_trajectories_straight_k6, uniform_probabilities_k6, False, expected_bade_uniform_k6),
        (forecasted_trajectories_straight_k6, confident_probabilities_k6, False, expected_bade_confident_k6),
        (forecasted_trajectories_straight_k6, non_normalized_probabilities_k6, True, expected_bade_confident_k6),
    ],
    ids=[
        "uniform_stationary_k1",
        "uniform_k6",
        "confident_k6",
        "normalize_probabilities_k6",
    ],
)
def test_compute_brier_ade(
    forecasted_trajectories: NDArrayNumber,
    forecast_probabilities: NDArrayNumber,
    normalize: bool,
    expected_brier_ade: NDArrayFloat,
) -> None:
    """Test that test_compute_brier_ade returns the correct output with valid inputs.

    Args:
        forecasted_trajectories: Forecasted trajectories for test case.
        forecast_probabilities: Forecast probabilities for test case.
        normalize: Controls whether forecast probabilities should be normalized for test case.
        expected_brier_ade: Expected probability-weighted ADE for test case.
    """
    brier_ade = metrics.compute_brier_ade(
        forecasted_trajectories, _STATIONARY_GT_TRAJ, forecast_probabilities, normalize
    )
    assert np.allclose(brier_ade, expected_brier_ade)


@pytest.mark.parametrize(
    "forecast_probabilities, normalize, expectation",
    [
        (non_normalized_probabilities_k6, True, does_not_raise()),
        (out_of_range_probabilities_k6, False, pytest.raises(ValueError)),
        (wrong_shape_probabilities_k6, True, pytest.raises(ValueError)),
    ],
    ids=[
        "valid_probabilities",
        "out_of_range_probabilities",
        "wrong_shape_probabilities",
    ],
)
def test_compute_brier_ade_data_validation(
    forecast_probabilities: NDArrayNumber, normalize: bool, expectation: AbstractContextManager
) -> None:
    """Test that test_compute_brier_ade raises the correct errors when inputs are invalid.

    Args:
        forecast_probabilities: Forecast probabilities for test case.
        normalize: Controls whether forecast probabilities should be normalized for test case.
        expectation: Context manager to capture the expected exception for each test case.
    """
    with expectation:
        metrics.compute_brier_ade(
            forecasted_trajectories_straight_k6, _STATIONARY_GT_TRAJ, forecast_probabilities, normalize
        )


@pytest.mark.parametrize(
    "forecasted_trajectories, forecast_probabilities, normalize, expected_brier_fde",
    [
        (forecasted_trajectories_stationary_k1, uniform_probabilities_k1, False, expected_bfde_uniform_k1),
        (forecasted_trajectories_straight_k6, uniform_probabilities_k6, False, expected_bfde_uniform_k6),
        (forecasted_trajectories_straight_k6, confident_probabilities_k6, False, expected_bfde_confident_k6),
        (forecasted_trajectories_straight_k6, non_normalized_probabilities_k6, True, expected_bfde_confident_k6),
    ],
    ids=[
        "uniform_stationary_k1",
        "uniform_k6",
        "confident_k6",
        "normalize_probabilities_k6",
    ],
)
def test_compute_brier_fde(
    forecasted_trajectories: NDArrayNumber,
    forecast_probabilities: NDArrayNumber,
    normalize: bool,
    expected_brier_fde: NDArrayFloat,
) -> None:
    """Test that test_compute_brier_fde returns the correct output with valid inputs.

    Args:
        forecasted_trajectories: Forecasted trajectories for test case.
        forecast_probabilities: Forecast probabilities for test case.
        normalize: Controls whether forecast probabilities should be normalized for test case.
        expected_brier_fde: Expected probability-weighted FDE for test case.
    """
    brier_fde = metrics.compute_brier_fde(
        forecasted_trajectories, _STATIONARY_GT_TRAJ, forecast_probabilities, normalize
    )
    assert np.allclose(brier_fde, expected_brier_fde)


@pytest.mark.parametrize(
    "forecast_probabilities, normalize, expectation",
    [
        (non_normalized_probabilities_k6, True, does_not_raise()),
        (out_of_range_probabilities_k6, False, pytest.raises(ValueError)),
        (wrong_shape_probabilities_k6, True, pytest.raises(ValueError)),
    ],
    ids=[
        "valid_probabilities",
        "out_of_range_probabilities",
        "wrong_shape_probabilities",
    ],
)
def test_compute_brier_fde_data_validation(
    forecast_probabilities: NDArrayNumber, normalize: bool, expectation: AbstractContextManager
) -> None:
    """Test that test_compute_brier_fde raises the correct errors when inputs are invalid.

    Args:
        forecast_probabilities: Forecast probabilities for test case.
        normalize: Controls whether forecast probabilities should be normalized for test case.
        expectation: Context manager to capture the expected exception for each test case.
    """
    with expectation:
        metrics.compute_brier_fde(
            forecasted_trajectories_straight_k6, _STATIONARY_GT_TRAJ, forecast_probabilities, normalize
        )
