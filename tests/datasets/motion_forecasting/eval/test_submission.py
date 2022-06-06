# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for AV2 motion forecasting challenge-related utilities."""

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from av2.datasets.motion_forecasting.constants import AV2_SCENARIO_PRED_TIMESTEPS
from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission, ScenarioPredictions, TrackPredictions

# Build valid submission with predictions for a single track in a single scenario
valid_track_predictions: TrackPredictions = (np.zeros((2, AV2_SCENARIO_PRED_TIMESTEPS, 2)), np.array([0.6, 0.4]))
valid_scenario_predictions: ScenarioPredictions = {"valid_track_id": valid_track_predictions}
valid_submission_predictions = {"valid_scenario_id": valid_scenario_predictions}

# Build invalid track submission with incorrect prediction length
too_short_track_predictions: TrackPredictions = (np.zeros((1, AV2_SCENARIO_PRED_TIMESTEPS - 1, 2)), np.array([1.0]))
too_short_scenario_predictions = {"invalid_track_id": too_short_track_predictions}
too_short_submission_predictions = {"invaild_scenario_id": too_short_scenario_predictions}

# Build invalid track submission with mismatched predicted trajectories and probabilities
mismatched_track_predictions: TrackPredictions = (np.zeros((1, AV2_SCENARIO_PRED_TIMESTEPS, 2)), np.array([0.5, 0.5]))
mismatched_scenario_predictions = {"invalid_track_id": mismatched_track_predictions}
mismatched_submission_predictions = {"invaild_scenario_id": mismatched_scenario_predictions}


@pytest.mark.parametrize(
    "test_submission_dict, expectation",
    [
        (valid_submission_predictions, does_not_raise()),
        (too_short_submission_predictions, pytest.raises(ValueError)),
        (mismatched_submission_predictions, pytest.raises(ValueError)),
    ],
    ids=["valid", "wrong_shape_trajectory", "mismatched_trajectory_probability_shape"],
)
def test_challenge_submission_data_validation(
    test_submission_dict: Dict[str, ScenarioPredictions], expectation: AbstractContextManager
) -> None:
    """Test that validation of submitted trajectories works as expected during challenge submission initialization.

    Args:
        test_submission_dict: Scenario-level predictions used to initialize challenge submission.
        expectation: Context manager to capture the appropriate exception for each test case.
    """
    with expectation:
        ChallengeSubmission(predictions=test_submission_dict)


@pytest.mark.parametrize(
    "test_submission_dict",
    [(valid_submission_predictions)],
    ids=["valid_submission"],
)
def test_challenge_submission_serialization(tmpdir: Path, test_submission_dict: Dict[str, ScenarioPredictions]) -> None:
    """Test that challenge submissions can be serialized/deserialized without changes to internal state.

    Args:
        tmpdir: tmpdir: Temp directory used in the test (provided via built-in fixture).
        test_submission_dict: Scenario-level predictions used to initialize challenge submission.
    """
    # Serialize submission to parquet file and load data back from disk
    submission_file_path = tmpdir / "submission.parquet"
    submission = ChallengeSubmission(predictions=test_submission_dict)
    submission.to_parquet(submission_file_path)
    deserialized_submission = ChallengeSubmission.from_parquet(submission_file_path)

    # Check that deserialized data matches original data exactly
    for scenario_id, scenario_predictions in submission.predictions.items():
        for track_id, (expected_trajectories, expected_probabilities) in scenario_predictions.items():
            deserialized_predictions = deserialized_submission.predictions[scenario_id][track_id]
            (deserialized_trajectories, deserialized_probabilities) = deserialized_predictions
            assert np.array_equal(deserialized_trajectories, expected_trajectories)
            assert np.array_equal(deserialized_probabilities, expected_probabilities)
