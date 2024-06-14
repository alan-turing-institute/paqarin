"""Tests for the univariate_metrics module."""

from unittest.mock import MagicMock

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame  # type: ignore

from paqarin import univariate_metrics

ITEM_ID_COLUMN: str = "item_ids"
TIMESTAMP_COLUMN: str = "timestamps"
TARGET_COLUMN: str = "metric"

frequency: str = "D"
covariate_column: str = "is_weekend"


def get_test_transformer() -> univariate_metrics.AutoGluonDataTransformer:
    """Generates an instance of AutoGluonDataTransformer for tests."""
    transformer: univariate_metrics.AutoGluonDataTransformer = (
        univariate_metrics.AutoGluonDataTransformer(
            item_id_column=ITEM_ID_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN,
            target_column=TARGET_COLUMN,
            frequency=frequency,
            covariate_column=covariate_column,
        )
    )

    return transformer


def get_test_dataframe() -> pd.DataFrame:
    """Generates a sample dataframe for tests."""
    data_frame: pd.DataFrame = pd.DataFrame(
        {
            ITEM_ID_COLUMN: [1, 1, 1, 2, 2],
            TIMESTAMP_COLUMN: [
                "2020-01-04",
                "2020-01-04",
                "2020-01-06",
                "2020-01-04",
                "2020-01-06",
            ],
            TARGET_COLUMN: [1.0, 1.0, 2.0, 3.0, None],
        }
    )

    return data_frame


def get_transformed_dataframe() -> pd.DataFrame:
    """Generates the expected transformation of the test dataframe."""
    expected_index: pd.MultiIndex = pd.MultiIndex.from_tuples(
        [
            (1, pd.to_datetime("2020-01-04")),
            (1, pd.to_datetime("2020-01-05")),
            (1, pd.to_datetime("2020-01-06")),
            (2, pd.to_datetime("2020-01-04")),
            (2, pd.to_datetime("2020-01-05")),
            (2, pd.to_datetime("2020-01-06")),
        ],
        names=["item_id", "timestamp"],
    )

    return pd.DataFrame(
        {
            "target": [1.0, 0.0, 2.0, 3.0, 0.0, 0.0],
            covariate_column: [1.0, 1.0, 0.0] * 2,
        },
        index=expected_index,
    )


def test_data_transformer() -> None:
    """Tests the data transformation for AutoGluon."""
    transformer: univariate_metrics.AutoGluonDataTransformer = get_test_transformer()

    data_frame: pd.DataFrame = get_test_dataframe()

    transformed_data_frame: TimeSeriesDataFrame = transformer.transform(data_frame)
    assert len(transformed_data_frame.index) == 6

    pd.testing.assert_frame_equal(
        transformed_data_frame,
        get_transformed_dataframe(),
    )


def test_predictive_scorer() -> None:
    """Tests the calculation of the predictive score."""
    data_transformer: univariate_metrics.AutoGluonDataTransformer = (
        get_test_transformer()
    )
    iterations: int = 1
    forecasting_evaluation_metric: str = "MAPE"
    validation_windows: int = 1
    prediction_length: int = 1
    number_of_sequences: int = 2
    training_time_limit: int = 10
    generation_rounds: int = 2

    predictive_scorer: univariate_metrics.AutoGluonPredictiveScorer = (
        univariate_metrics.AutoGluonPredictiveScorer(
            iterations=iterations,
            forecasting_evaluation_metric=forecasting_evaluation_metric,
            validation_windows=validation_windows,
            prediction_length=prediction_length,
            number_of_sequences=number_of_sequences,
            training_time_limit=training_time_limit,
            transformer=data_transformer,
            generation_rounds=generation_rounds,
        )
    )

    generator_name: str = "tests"
    mock_forecasting_model: MagicMock = MagicMock()
    metric_value: float = 20.0
    mock_forecasting_model.evaluate = MagicMock(
        return_value={forecasting_evaluation_metric: metric_value}
    )

    testing_time_series: pd.DataFrame = TimeSeriesDataFrame.from_data_frame(
        pd.DataFrame(
            {
                univariate_metrics.ITEM_ID_COLUMN: [],
                univariate_metrics.TIMESTAMP_COLUMN: [],
            }
        )
    )
    predicted_time_series: pd.DataFrame = TimeSeriesDataFrame.from_data_frame(
        pd.DataFrame(
            {
                univariate_metrics.ITEM_ID_COLUMN: [],
                univariate_metrics.TIMESTAMP_COLUMN: [],
            }
        )
    )
    iteration: int = 0
    predictive_scorer.testing_time_series = testing_time_series
    predictive_scorer.register_prediction_results(
        iteration,
        generator_name,
        mock_forecasting_model,
        predicted_time_series,
    )
    mock_forecasting_model.evaluate.assert_called_once_with(testing_time_series)
    assert predictive_scorer.metric_manager.get_all_values(
        generator_name, forecasting_evaluation_metric
    ) == [metric_value]

    transformed_data: pd.DataFrame = data_transformer.transform(get_test_dataframe())
    training_data, testing_data = predictive_scorer.split_time_series(transformed_data)
    pd.testing.assert_frame_equal(testing_data, transformed_data)

    training_index: pd.MultiIndex = pd.MultiIndex.from_tuples(
        [
            (1, pd.to_datetime("2020-01-04")),
            (1, pd.to_datetime("2020-01-05")),
            (2, pd.to_datetime("2020-01-04")),
            (2, pd.to_datetime("2020-01-05")),
        ],
        names=["item_id", "timestamp"],
    )
    pd.testing.assert_frame_equal(
        training_data,
        pd.DataFrame(
            {
                "target": [1.0, 0.0, 3.0, 0.0],
                covariate_column: [1.0, 1.0] * 2,
            },
            index=training_index,
        ),
    )

    mock_generator_instance: MagicMock = MagicMock()
    mock_generator_instance.generate.side_effect = [
        [pd.DataFrame({"column": [1, 2]}), pd.DataFrame({"column": [3, 4]})],
        [pd.DataFrame({"column": [5, 6]}), pd.DataFrame({"column": [7, 8]})],
    ]

    synthetic_dataframe: pd.DataFrame = predictive_scorer.generate_synthetic_data(
        generator_name, mock_generator_instance
    )

    assert len(synthetic_dataframe.index) == 8
    assert mock_generator_instance.generate.call_count == generation_rounds
    pd.testing.assert_frame_equal(
        synthetic_dataframe, pd.DataFrame({"column": [1, 2, 3, 4, 5, 6, 7, 8]})
    )
