"""Abstractions for gathering evaluation metrics of synthetic time series."""

import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

from paqarin.generator import (
    DummyTransformer,
    GeneratorParameters,
    GeneratorTransformer,
    TimeSeriesGenerator,
)
from paqarin.generators import (
    DoppleGangerGenerator,
    DoppleGanGerParameters,
    TimeGanGenerator,
    TimeGanParameters,
)

TRAINING_DATA_KEY: str = "training_data"
SYNTHETIC_DATA_KEY: str = "synthetic_data"

AVERAGE_PREFIX: str = "avg_"
GENERATOR_NAME_KEY: str = "generator_name"
METRIC_VALUE_KEY: str = "mean_absolute_error"
STANDARD_DEVIATION_PREFIX: str = "std_"


class TrainingMetadata:
    """Stores information needed for triggering training."""

    def __init__(self, provider: str, generator_parameters: GeneratorParameters):
        """Inits training metadata."""
        self.provider: str = provider
        self.generator_parameters: GeneratorParameters = generator_parameters

    def create_generator(self) -> TimeSeriesGenerator:
        """Gets a generator instance from metadata."""
        if isinstance(self.generator_parameters, DoppleGanGerParameters):
            return DoppleGangerGenerator(
                provider=self.provider, generator_parameters=self.generator_parameters
            )
        elif isinstance(self.generator_parameters, TimeGanParameters):
            return TimeGanGenerator(
                provider=self.provider, generator_parameters=self.generator_parameters
            )
        else:
            raise ValueError("Cannot create a generator with the provided information")


class MetricManager:
    """Manages metrics obtained over multiple iterations."""

    def __init__(self):
        """Inits the Metric Manager."""
        self.metrics_per_generator: dict[str, list[dict]] = {}  # type: ignore

    def register_iteration(
        self, generator_name: str, iteration_metrics: dict[str, Any]
    ):
        """For a given generator, adds the metrics from an iteration."""
        if generator_name not in self.metrics_per_generator:
            self.metrics_per_generator[generator_name] = []
        self.metrics_per_generator[generator_name].append(iteration_metrics)

    def get_all_values(self, generator_name: str, metric_key: str) -> list[float]:
        """Return the values of a metric over all iterations."""
        all_values: list[float] = [
            iteration_metrics[metric_key]
            for iteration_metrics in self.metrics_per_generator[generator_name]
        ]

        return all_values

    def get_iteration_values(
        self, generator_name: str, iteration_index: int, metric_key: str
    ):
        """Get metric values for an specific iteration."""
        if generator_name not in self.metrics_per_generator:
            raise ValueError(
                f"There is no generator associated with key: {generator_name}"
            )

        return self.metrics_per_generator[generator_name][iteration_index][metric_key]

    def calculate_average(self, generator_name: str, metric_key: str) -> float:
        """Calculate the average of a metric over all iterations."""
        return np.mean(self.get_all_values(generator_name, metric_key)).astype(float)

    def calculate_standard_deviation(
        self, generator_name: str, metric_key: str
    ) -> float:
        """Calculate the standard deviation of a metric over all iterations."""
        return np.std(self.get_all_values(generator_name, metric_key)).astype(float)


class BasePredictiveScorer(ABC):
    """Base class for Predictive Scorers.

    Predictive scorers evaluate synthetic data on forecasting tasks.
    """

    def __init__(self, iterations: int = 10, metric_value_key: str = METRIC_VALUE_KEY):
        """Inits the Predictive Scorer."""
        self.iterations: int = iterations
        self.metric_value_key: str = metric_value_key

        # TODO: We might need a better way for exposing these data.
        self.metric_manager: MetricManager = MetricManager()
        self.summary_metrics: list[dict[str, Any]] = []

        self.best_generator_name: Optional[str] = None
        self.best_generator_value: Optional[float] = None

    def update_summary_metrics(self, generator_name: str) -> None:
        """Registers consolidated metrics after multiple iteration runs."""
        self.summary_metrics.append(
            {
                GENERATOR_NAME_KEY: generator_name,
                AVERAGE_PREFIX
                + self.metric_value_key: self.metric_manager.calculate_average(
                    generator_name, self.metric_value_key
                ),
                STANDARD_DEVIATION_PREFIX
                + self.metric_value_key: self.metric_manager.calculate_standard_deviation(
                    generator_name, self.metric_value_key
                ),
            }
        )

    @abstractmethod
    def calculate(
        self,
        generator_instance: TimeSeriesGenerator,
        generator_name: str,
        training_data: pd.DataFrame,
    ):
        """Calculates a metric.

        It uses synthetic data from a generator, trained over real data.
        """


class EvaluationPipeline:
    """Evaluates the performance of multiple synthetic time-series generators."""

    def __init__(
        self,
        generator_map: dict[str, TimeSeriesGenerator],
        scoring: Optional[BasePredictiveScorer] = None,
    ):
        """Inits the evaluation pipeline."""
        self.generator_map: dict[str, TimeSeriesGenerator] = generator_map
        self.training_results: list[dict[str, Any]] = []
        self.scoring_object: Optional[BasePredictiveScorer] = scoring
        self.best_generator: Optional[str] = None

    def fit(
        self,
        training_data: pd.DataFrame,
        save_after_fitting: bool = False,
        **training_arguments: Any,
    ) -> None:
        """Trains the algorithms in the map, and computes performance metrics."""
        # TODO: We should add a progress bar here...
        for generator_name, generator_instance in self.generator_map.items():
            self.fit_generator(
                generator_instance,
                generator_name,
                training_data,
                save_after_fitting,
                **training_arguments,
            )

            self.calculate_metrics(training_data, generator_instance, generator_name)

        if self.scoring_object is not None:
            self.training_results = (
                self.training_results + self.scoring_object.summary_metrics
            )
            self.best_generator = self.scoring_object.best_generator_name

    def calculate_metrics(
        self,
        training_data: pd.DataFrame,
        generator_instance: TimeSeriesGenerator,
        generator_name: str,
    ) -> None:
        """Computes performance metrics and stores them."""
        if self.scoring_object is not None:
            self.scoring_object.calculate(
                generator_instance, generator_name, training_data
            )
        else:
            logging.info(
                "No scoring object provided. No evaluation performed "
                f"after training {generator_name}"
            )

    def fit_generator(
        self,
        generator_instance: TimeSeriesGenerator,
        generator_name: str,
        training_data: pd.DataFrame,
        save_after_fitting: bool,
        **training_arguments: Any,
    ) -> None:
        """Trains a synthetic data generator using the provided data."""
        # TODO: We need to decouple this logic from the pipeline, to trigger training
        # without evaluation.
        if generator_instance.transformer is None:
            raise ValueError("Missing transformer instance")

        generator_transformer: GeneratorTransformer = generator_instance.transformer

        if (
            type(generator_transformer) is not DummyTransformer
            and not generator_transformer.is_fitted()
        ):
            print(f"Fitting transformer for {generator_name}")
            generator_transformer.fit(training_data)

        if generator_instance.generator is None:
            print(f"Fitting generator {generator_name}")
            generator_instance.fit(training_data, **training_arguments)

            if save_after_fitting:
                generator_instance.save()

        else:
            print(f"There's already a trained instance of {generator_name}")

    def save_training_metadata(self, generator_name: str, file_name: str):
        """Saves training configuration to a file."""
        generator_instance: TimeSeriesGenerator = self.generator_map[generator_name]
        training_metadata: TrainingMetadata = TrainingMetadata(
            provider=generator_instance.provider,
            generator_parameters=generator_instance.parameters,
        )

        with open(file_name, "wb") as output:
            pickle.dump(training_metadata, output)

        print(f"Training metadata for generator {generator_name} save to {file_name}")
