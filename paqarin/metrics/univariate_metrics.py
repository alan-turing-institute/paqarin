"""This module contains logic for evaluating synthetic time series.

It uses AutoGluon in univariate time series forecasting tasks
"""

import logging
import os.path
import traceback
from typing import Any, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from autogluon.timeseries import TimeSeriesDataFrame  # type: ignore
from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.utils.forecast import (  # type: ignore
    get_forecast_horizon_index_ts_dataframe,
)

from paqarin.evaluation import BasePredictiveScorer
from paqarin.generator import TimeSeriesGenerator
from paqarin.utils import LINE_STYLE, MARKER

from .multivariate_metrics import TRAINING_PREDICTIONS

ITEM_ID_COLUMN: str = "item_id"
TIMESTAMP_COLUMN: str = "timestamp"
TARGET_COLUMN: str = "target"


class AutoGluonDataTransformer:
    """Transformer for AutoGluon forecasting models."""

    def __init__(
        self,
        item_id_column: str,
        timestamp_column: str,
        target_column: str,
        frequency: str,
        covariate_column: str,
    ):
        """Inits the AutoGluon data transformer."""
        self.item_id_column = item_id_column
        self.timestamp_column = timestamp_column
        self.target_column = target_column
        self.frequency = frequency

        self.weekend_indices: list[int] = [5, 6]
        # TODO: If we're always doing weekends, this value shouldn't be a parameter.
        self.covariate_column: str = covariate_column

    def transform(self, data_frame: pd.DataFrame) -> TimeSeriesDataFrame:
        """Transforms the provided data to be suitable for AutoGluon."""
        logging.info("Renaming columns")
        transformed_dataframe: pd.DataFrame = data_frame.rename(
            columns={
                self.item_id_column: ITEM_ID_COLUMN,
                self.timestamp_column: TIMESTAMP_COLUMN,
                self.target_column: TARGET_COLUMN,
            }
        )

        logging.info("From DataFrame to TimeSeriesDataFrame")
        timeseries_dataframe: TimeSeriesDataFrame = (
            TimeSeriesDataFrame.from_data_frame(transformed_dataframe)
        )

        duplicate_rows: np.ndarray = timeseries_dataframe.index.duplicated(
            keep="first"
        )
        logging.info(
            f"Removing {duplicate_rows.sum()} rows with duplicate indexes"
        )
        timeseries_dataframe = timeseries_dataframe[~duplicate_rows]

        logging.info(
            f"Setting up frequency {self.frequency} and filling missing values"
        )
        timeseries_dataframe = timeseries_dataframe.to_regular_index(
            freq=self.frequency
        )

        items_without_frequency: list[str] = [
            item_id
            for item_id in timeseries_dataframe.item_ids
            if AutoGluonDataTransformer.get_item_frequency(
                timeseries_dataframe, item_id
            )
            is None
        ]
        logging.info(
            f"Removing {len(items_without_frequency)} "
            "because of missing frequencies"
        )

        timeseries_dataframe = timeseries_dataframe.drop(
            items_without_frequency, level=ITEM_ID_COLUMN
        )

        logging.info(f"New dataset frequency: {timeseries_dataframe.freq}")
        timeseries_dataframe = timeseries_dataframe.fill_missing_values(
            method="constant"
        )

        logging.info("Adding known covariates")
        self.add_known_covariates(timeseries_dataframe)

        logging.info(f"Final timeseries size: {len(timeseries_dataframe)}")
        return timeseries_dataframe

    @staticmethod
    def get_item_frequency(
        timeseries_dataframe: TimeSeriesDataFrame, item_id: str
    ) -> str:
        """Returns the frequency of the provided timeseries."""
        item_data: TimeSeriesDataFrame = timeseries_dataframe.loc[item_id]
        item_frequency: str = (
            item_data.index.freq or item_data.index.inferred_freq
        )

        return item_frequency

    def add_known_covariates(self, timeseries_dataframe: TimeSeriesDataFrame):
        """Adds a weekend covariate to the provided time series."""
        # TODO: Maybe a more general way of adding covariates?
        timestamps = timeseries_dataframe.index.get_level_values(
            TIMESTAMP_COLUMN
        )
        timeseries_dataframe[self.covariate_column] = timestamps.weekday.isin(
            self.weekend_indices
        ).astype(float)


class AutoGluonPredictiveScorer(BasePredictiveScorer):
    """Calculates the Predictive Score for an univariate forecasting task."""

    def __init__(
        self,
        iterations: int,
        forecasting_evaluation_metric: str,
        validation_windows: int,
        prediction_length: int,
        number_of_sequences: int,
        training_time_limit: int,
        transformer: AutoGluonDataTransformer,
        generation_rounds: int = 10,
        predictor_verbosity: int = 2,
        generation_arguments: Optional[dict[str, Any]] = None,
        reuse_files: bool = False,
    ):
        """Inits the AutoGluon Predictive Scorer."""
        self.forecasting_evaluation_metric: str = forecasting_evaluation_metric
        self.validation_windows: int = validation_windows
        self.prediction_length: int = prediction_length
        self.number_of_sequences: int = number_of_sequences
        self.training_time_limit: int = training_time_limit
        self.generation_rounds: int = generation_rounds
        self.reuse_files: bool = reuse_files

        self.predictor_verbosity: int = predictor_verbosity

        self.transformer: AutoGluonDataTransformer = transformer
        self.generation_arguments: Optional[dict[str, Any]] = (
            generation_arguments
        )

        self.training_time_series: Optional[TimeSeriesDataFrame] = None
        self.testing_time_series: Optional[TimeSeriesDataFrame] = None

        super().__init__(
            iterations=iterations,
            metric_value_key=self.forecasting_evaluation_metric,
        )

    def calculate(
        self,
        generator_instance: TimeSeriesGenerator,
        generator_name: str,
        evaluation_data: pd.DataFrame,
    ):
        """Calculates the predictive score, given the evaluation data provided."""
        logging.info(f"Reusing previous training files: {self.reuse_files}")

        training_file_name: str = (
            f"{generator_name}/{generator_name}_training_data.csv"
        )
        testing_file_name: str = (
            f"{generator_name}/{generator_name}_testing_data.csv"
        )

        if self.reuse_files:
            self.training_time_series = pd.read_csv(training_file_name)
            self.testing_time_series = pd.read_csv(testing_file_name)
            logging.info(
                f"Training data loaded from {training_file_name}. "
                f"Testing data loaded from: {testing_file_name}"
            )
        else:
            (
                self.training_time_series,
                self.testing_time_series,
            ) = self.split_time_series(
                self.transformer.transform(evaluation_data)
            )

            self.training_time_series.to_csv(training_file_name)
            logging.info(
                f"{generator_name} evaluation: Training data "
                f"saved at {training_file_name}"
            )
            self.testing_time_series.to_csv(testing_file_name)
            logging.info(
                f"{generator_name} evaluation: Testing data "
                f"saved at {testing_file_name}"
            )

        train_on_real_key: str = "train_on_real"
        if train_on_real_key not in self.metric_manager.metrics_per_generator:
            for iteration in range(self.iterations):
                logging.info(
                    f"{iteration}: Training forecasting model using real data..."
                )

                self.train_predict_and_register(
                    iteration, train_on_real_key, self.training_time_series
                )

            self.update_summary_metrics(train_on_real_key)

        for iteration in range(self.iterations):
            synthetic_dataframe: pd.DataFrame
            synthetic_data_file: str = (
                f"{generator_name}/{iteration}_{generator_name}_synthetic_data.csv"
            )

            if self.reuse_files and os.path.isfile(synthetic_data_file):
                synthetic_dataframe = pd.read_csv(synthetic_data_file)
                logging.info(
                    f"Synthetic data loaded from: {synthetic_data_file}"
                )
            else:
                synthetic_dataframe = self.generate_synthetic_data(
                    generator_name=generator_name,
                    generator_instance=generator_instance,
                )
                synthetic_dataframe.to_csv(synthetic_data_file)
                logging.info(
                    f"Synthetic data stored at from: {synthetic_data_file}"
                )

            try:
                synthetic_time_series: TimeSeriesDataFrame = (
                    self.transformer.transform(synthetic_dataframe)
                )

                synthetic_training_series, _ = self.split_time_series(
                    synthetic_time_series
                )

                self.train_predict_and_register(
                    iteration, generator_name, synthetic_training_series
                )
            except ValueError:
                logging.warning(
                    f"The synthetic dataset at iteration {iteration} from "
                    f" {generator_name} is not suitable for forecasting"
                )
                logging.error(traceback.format_exc())

        self.update_summary_metrics(generator_name)

        average_metric: float = self.metric_manager.calculate_average(
            generator_name, self.forecasting_evaluation_metric
        )

        if (
            self.best_generator_value is None
            or self.best_generator_value > average_metric
        ):
            logging.info(
                f"{generator_name}"
                f" (Avg. {self.forecasting_evaluation_metric} {average_metric}) "
                f"is better than {self.best_generator_name} "
                f"({self.forecasting_evaluation_metric} {self.best_generator_value})!"
            )
            self.best_generator_name = generator_name
            self.best_generator_value = average_metric

    def train_predict_and_register(
        self, iteration: int, generator_name: str, training_data: pd.DataFrame
    ) -> None:
        """Trains a generator, generates synthetic data, forecast and register results."""
        predictions_csv_file: str = (
            f"{generator_name}/{iteration}_{generator_name}_forecasting.csv"
        )
        model_directory: str = (
            f"{generator_name}/{iteration}_{generator_name}_model"
        )

        trained_model: TimeSeriesPredictor
        model_forecasting: TimeSeriesDataFrame

        if self.reuse_files:
            trained_model = TimeSeriesPredictor.load(model_directory)
            logging.info(
                f"Model for {generator_name} at iteration {iteration} "
                f"loaded from {model_directory}"
            )

            model_forecasting = TimeSeriesDataFrame.from_path(
                predictions_csv_file
            )
            logging.info(
                f"Forecasting for {generator_name} at iteration {iteration} "
                f"loaded from {predictions_csv_file}"
            )

        else:
            (
                trained_model,
                model_forecasting,
            ) = self.predict_after_training(training_data, model_directory)

            trained_model.save()
            logging.info(
                f"Model for {generator_name} at iteration {iteration} "
                f"saved at {model_directory}"
            )

            model_forecasting.to_csv(predictions_csv_file)
            logging.info(
                f"Forecasting for {generator_name} at iteration {iteration} "
                f"saved at {predictions_csv_file}"
            )

        self.register_prediction_results(
            iteration,
            generator_name,
            trained_model,
            training_data,
            model_forecasting,
        )

    def generate_synthetic_data(
        self, generator_name, generator_instance: TimeSeriesGenerator
    ) -> pd.DataFrame:
        """Generates a synthetic dataset, using a generator instance."""
        # TODO: Address the missing datetime column issue. Also, the requirements for
        # AutoGluon training (type, indexes, known covariates)

        logging.info(
            f"Generating {self.number_of_sequences} ({self.generation_rounds} times) "
            f"synthetic sequences using {generator_name}..."
        )

        synthetic_sequences: list[pd.DataFrame] = []

        for _ in range(self.generation_rounds):
            if self.generation_arguments is not None:
                synthetic_sequences += generator_instance.generate(
                    self.number_of_sequences, **self.generation_arguments
                )
            else:
                synthetic_sequences += generator_instance.generate(
                    self.number_of_sequences
                )

        synthetic_dataframe: pd.DataFrame = pd.concat(
            synthetic_sequences, axis=0
        )
        synthetic_dataframe = synthetic_dataframe.reset_index(drop=True)

        return synthetic_dataframe

    def predict_after_training(
        self,
        training_time_series: TimeSeriesDataFrame,
        model_directory: str,
    ) -> tuple[TimeSeriesPredictor, TimeSeriesDataFrame]:
        """Trains a forecasting model, and returns its forecast."""
        forecasting_model: TimeSeriesPredictor = self.create_forecasting_model(
            model_directory
        )
        forecasting_model.fit(
            training_time_series,
            num_val_windows=self.validation_windows,
            time_limit=self.training_time_limit,
            verbosity=self.predictor_verbosity,
        )

        # We are forecasting training on real, to match testing on real.
        forecasting_results: TimeSeriesDataFrame = self.forecast(
            forecasting_model, self.training_time_series
        )

        return forecasting_model, forecasting_results

    def forecast(
        self,
        forecasting_model: TimeSeriesPredictor,
        time_series_dataframe: TimeSeriesDataFrame,
    ) -> TimeSeriesDataFrame:
        """Generates a forecast using an AutoGluon, for the provided dataframe."""
        forecast_index: pd.MultiIndex = (
            get_forecast_horizon_index_ts_dataframe(
                time_series_dataframe, prediction_length=self.prediction_length
            )
        )

        forecast_covariates: TimeSeriesDataFrame = TimeSeriesDataFrame(
            pd.DataFrame(index=forecast_index)
        )

        self.transformer.add_known_covariates(forecast_covariates)

        return forecasting_model.predict(
            time_series_dataframe,
            known_covariates=forecast_covariates,
        )

    def create_forecasting_model(
        self, model_directory: str
    ) -> TimeSeriesPredictor:
        """Creates an AutoGluon predictor instance."""

        known_covariates_names: Optional[list[str]] = None
        if self.transformer.covariate_column:
            known_covariates_names = [self.transformer.covariate_column]

        return TimeSeriesPredictor(
            path=model_directory,
            prediction_length=self.prediction_length,
            known_covariates_names=known_covariates_names,
            eval_metric=self.forecasting_evaluation_metric,
        )

    def split_time_series(
        self,
        evaluation_data: TimeSeriesDataFrame,
    ) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Splits data into training and testing."""

        test_data: TimeSeriesDataFrame = evaluation_data.slice_by_timestep(
            None, None
        )
        train_data: TimeSeriesDataFrame = test_data.slice_by_timestep(
            None, -self.prediction_length
        )

        return train_data, test_data

    def register_prediction_results(
        self,
        iteration: int,
        generator_name: str,
        forecasting_model: TimeSeriesPredictor,
        testing_time_series: TimeSeriesDataFrame,
        predicted_time_series: TimeSeriesDataFrame,
    ):
        """Calculates and stores forecasting metrics."""
        scores_dictionary: Union[int, dict] = forecasting_model.evaluate(
            testing_time_series
        )
        if type(scores_dictionary) is not dict:
            scores_dictionary = {
                self.forecasting_evaluation_metric: scores_dictionary
            }

        metrics_csv_file: str = (
            f"{generator_name}/{iteration}_{generator_name}_metrics.csv"
        )

        pd.DataFrame.from_dict(scores_dictionary, orient="index").to_csv(
            metrics_csv_file
        )
        logging.info(
            f"Evaluation metrics for {generator_name} at iteration {iteration} "
            f"saved at {metrics_csv_file}"
        )

        metric_on_testing: float = scores_dictionary[
            self.forecasting_evaluation_metric
        ]
        logging.info(
            f"Registering forecasting results for {generator_name=}"
            f" {self.forecasting_evaluation_metric}"
            f" {metric_on_testing}"
        )

        iteration_metrics: dict[str, Any] = {
            self.forecasting_evaluation_metric: metric_on_testing,
            TRAINING_PREDICTIONS: predicted_time_series,
        }

        self.metric_manager.register_iteration(
            generator_name, iteration_metrics
        )


def plot_forecast(
    item_ids: list[str],
    target_column: str,
    prediction_length: int,
    training_data: TimeSeriesDataFrame,
    testing_data: TimeSeriesDataFrame,
    forecast_data: TimeSeriesDataFrame,
    font_scale: float,
    figure_size: tuple[int, int] = (20, 3),
) -> None:
    """Plots the quantile forecast."""
    sns.set(font_scale=font_scale)

    figure, axes = plt.subplots(
        nrows=len(item_ids), figsize=figure_size, sharex=True, squeeze=False
    )

    for index, item_id in enumerate(item_ids):
        axis = axes[index][0]
        past_values: np.ndarray = training_data.loc[item_id][target_column]
        axis.set_title(f"Item: {item_id}")

        axis.plot(
            past_values,
            marker=MARKER,
            linestyle=LINE_STYLE,
            label="Past values",
        )

        forecast_mean: np.ndarray = forecast_data.loc[item_id]["mean"]
        axis.plot(
            forecast_mean,
            marker=MARKER,
            linestyle=LINE_STYLE,
            label="Mean forecast",
        )

        real_values: np.ndarray = testing_data.loc[item_id][target_column][
            -prediction_length:
        ]
        axis.plot(
            real_values,
            marker=MARKER,
            linestyle=LINE_STYLE,
            label="Real values",
        )

        axis.fill_between(
            forecast_data.loc[item_id].index,
            forecast_data.loc[item_id]["0.1"],
            forecast_data.loc[item_id]["0.9"],
            alpha=0.1,
            label="Confidence Interval",
            color="red",
        )

    plt.legend(loc="upper left")
    plt.show()

    sns.set()
