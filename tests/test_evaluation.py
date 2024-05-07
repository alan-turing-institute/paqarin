"""Tests for the evaluation module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from paqarin.evaluation import EvaluationPipeline, MetricManager


@patch("paqarin.generators.DoppleGangerGenerator")
@patch("paqarin.generators.TimeGanGenerator")
def test_evaluation_pipeline(
    mock_timegan_generator: MagicMock, mock_doppleganger_generator: MagicMock
) -> None:
    """Tests evaluating multiple time series generators."""
    time_gan_generator: MagicMock = mock_timegan_generator.return_value
    time_gan_generator.generator = None  # To signal the need for training.

    doppleganger_generator: MagicMock = mock_doppleganger_generator.return_value

    evaluation_pipeline: EvaluationPipeline = EvaluationPipeline(
        generator_map={
            "timegan": time_gan_generator,
            "doppleganger": doppleganger_generator,
        }
    )

    training_data: pd.DataFrame = pd.DataFrame()
    evaluation_pipeline.fit(training_data, True, keyword_argument="argument_value")

    time_gan_generator.fit.assert_called_once_with(
        training_data, keyword_argument="argument_value"
    )
    time_gan_generator.save.assert_called_once()

    doppleganger_generator.fit.assert_not_called()
    doppleganger_generator.save.assert_not_called()


def test_metrics_manager() -> None:
    """Tests managing evaluation metrics."""
    metric_manager: MetricManager = MetricManager()

    generator_name: str = "my_generator"
    metric_key: str = "metric_key"
    metric_manager.register_iteration(generator_name, {metric_key: 0.0})
    metric_manager.register_iteration(generator_name, {metric_key: 10.0})
    metric_manager.register_iteration(generator_name, {metric_key: 20.0})

    assert metric_manager.metrics_per_generator[generator_name] == [
        {metric_key: 0.0},
        {metric_key: 10.0},
        {metric_key: 20.0},
    ]

    assert [0.0, 10.0, 20.0] == metric_manager.get_all_values(
        generator_name, metric_key
    )

    with pytest.raises(ValueError) as error_info:
        metric_manager.get_iteration_values("random_generator", 1, metric_key)
    assert (
        str(error_info.value)
        == "There is no generator associated with key: random_generator"
    )

    assert 10.0 == metric_manager.get_iteration_values(generator_name, 1, metric_key)

    assert 10.0 == metric_manager.calculate_average(generator_name, metric_key)
