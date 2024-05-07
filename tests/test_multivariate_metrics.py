"""Tests for the multivariate_metrics module."""

import numpy as np
import pandas as pd

from paqarin import multivariate_metrics


def generate_testing_sequences() -> tuple[list, int]:
    """Generates synthetic sequences for testing."""
    first_column: str = "a_column"
    second_column: str = "another_column"
    first_sequence: pd.DataFrame = pd.DataFrame(
        {first_column: [0, 1, 2], second_column: [0, 10, 20]}
    )

    second_sequence: pd.DataFrame = pd.DataFrame(
        {first_column: [2, 3, 4], second_column: [20, 30, 40]}
    )

    third_sequence: pd.DataFrame = pd.DataFrame(
        {first_column: [3, 4, 5], second_column: [30, 40, 50]}
    )

    time_series_array: list[pd.DataFrame] = [
        first_sequence,
        second_sequence,
        third_sequence,
    ]

    sequence_length: int = 3

    return time_series_array, sequence_length


def test_do_x_y_split() -> None:
    """Tests splitting data into features and labels."""
    time_series_array, sequence_length = generate_testing_sequences()
    indexes: np.ndarray = np.arange(2)
    features, labels = multivariate_metrics.do_x_y_split(
        np.asarray(time_series_array), indexes, sequence_length
    )

    np.testing.assert_array_equal(
        features,
        np.array(
            [
                [[0, 0], [1, 10]],
                [[2, 20], [3, 30]],
            ]
        ),
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [
                [2, 20],
                [4, 40],
            ]
        ),
    )


def test_split_time_series() -> None:
    """Tests splitting data into training, validation, and testing."""
    time_series_array, sequence_length = generate_testing_sequences()

    (
        x_training,
        y_training,
        x_validation,
        y_validation,
        x_testing,
        y_testing,
    ) = multivariate_metrics.split_time_series(time_series_array, sequence_length, 0.4)

    np.testing.assert_array_equal(x_training, np.array([[[0, 0], [1, 10]]]))
    np.testing.assert_array_equal(y_training, np.array([[2, 20]]))

    np.testing.assert_array_equal(x_validation, np.array([[[2, 20], [3, 30]]]))
    np.testing.assert_array_equal(y_validation, np.array([[4, 40]]))

    np.testing.assert_array_equal(x_testing, np.array([[[3, 30], [4, 40]]]))
    np.testing.assert_array_equal(y_testing, np.array([[5, 50]]))
