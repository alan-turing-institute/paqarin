"""Code for assessing the quality of the generated synthetic data.

It focusses on the assessment of multivariate forecasting tasks.
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping  # type: ignore
from keras.layers import LSTM, Dense  # type: ignore
from keras.losses import MeanAbsoluteError  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.optimizers import Adam  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore

from paqarin import ydata_adapter
from paqarin.evaluation import BasePredictiveScorer  # type: ignore
from paqarin.generator import TimeSeriesGenerator

EVALUATION_FEATURES: str = "evaluation_features"
EVALUATION_LABELS: str = "evaluation_labels"
TRAINING_PREDICTIONS: str = "training_predictions"


def do_x_y_split(
    time_series_array: np.ndarray, indexes: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """From an array and indexes, this splits into features and labels."""
    x_data: np.ndarray = time_series_array[indexes, : sequence_length - 1, :]
    y_data: np.ndarray = time_series_array[indexes, -1, :]

    return x_data, y_data


def split_time_series(
    time_series_data: list[np.ndarray],
    sequence_length: int,
    training_size: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Given a list of sequences, it splits into training and testing dataset."""
    time_series_array: np.ndarray = np.asarray(time_series_data)
    number_of_sequences: int = len(time_series_array)

    all_indexes: np.ndarray = np.arange(number_of_sequences)
    training_threshold: int = int(training_size * number_of_sequences)
    validation_threshold: int = training_threshold + int(
        (number_of_sequences - training_threshold) / 2
    )

    training_indexes: np.ndarray = all_indexes[:training_threshold]
    validation_indexes: np.ndarray = all_indexes[
        training_threshold:validation_threshold
    ]
    testing_indexes: np.ndarray = all_indexes[validation_threshold:]

    x_training, y_training = do_x_y_split(
        time_series_array, training_indexes, sequence_length
    )
    logging.info(
        f"Training X shape: {x_training.shape} " f"Training y shape: {y_training.shape}"
    )

    x_validation, y_validation = do_x_y_split(
        time_series_array, validation_indexes, sequence_length
    )
    logging.info(
        f"Validation X shape: {x_validation.shape}"
        f" Validation y shape: {y_validation.shape}"
    )

    x_testing, y_testing = do_x_y_split(
        time_series_array, testing_indexes, sequence_length
    )
    logging.info(
        f"Testing X shape: {x_testing.shape} testing y shape: {y_testing.shape}"
    )

    return x_training, y_training, x_validation, y_validation, x_testing, y_testing


class PredictiveScorer(BasePredictiveScorer):
    """Trains an LSTM on real/synthetic data, and evaluates on real using MAE.

    Attributes:
        iterations: Number of times the metric is computer.
            In the TimeGAN repo, this value defaults to 10.

    """

    # TODO: Maybe a "plot" method like Pandas Dataframes?

    def __init__(
        self,
        lstm_units: int,
        scorer_epochs: int,
        scorer_batch_size: int,
        number_of_features: int,
        sequence_length: int,
        # TODO: Maybe support for categorical ones?
        numerical_columns: list[str],
        metric_value_key: str,
        iterations: int = 10,
        patience: int = 2,
        training_size: float = 0.7,
    ):
        """Inits the Predictive Scorer."""
        self.lstm_units: int = lstm_units
        self.epochs: int = scorer_epochs
        self.batch_size: int = scorer_batch_size
        self.number_of_features: int = number_of_features
        self.patience: int = patience
        self.training_size: float = training_size

        self.sequence_length: int = sequence_length
        self.timegan_transformer: ydata_adapter.TimeGanTransformer = (
            ydata_adapter.TimeGanTransformer(
                sequence_length=sequence_length,
                numerical_columns=numerical_columns,
            )
        )

        self.training_features: Optional[np.ndarray] = None
        self.training_labels: Optional[np.ndarray] = None

        self.validation_features: Optional[np.ndarray] = None
        self.validation_labels: Optional[np.ndarray] = None

        self.testing_features: Optional[np.ndarray] = None
        self.testing_labels: Optional[np.ndarray] = None

        self.best_generator_value: Optional[float] = None

        super().__init__(iterations=iterations, metric_value_key=metric_value_key)

    def calculate(
        self,
        generator_instance: TimeSeriesGenerator,
        generator_name: str,
        training_data: pd.DataFrame,
    ):
        """Calculates the predictive score metric."""
        # TODO: We might need a refactor to avoid doing this every time.
        preprocessed_real_data: list[np.ndarray] = self.timegan_transformer.transform(
            training_data
        )

        (
            self.training_features,
            self.training_labels,
            self.validation_features,
            self.validation_labels,
            self.testing_features,
            self.testing_labels,
        ) = split_time_series(
            preprocessed_real_data,
            self.sequence_length,
            training_size=self.training_size,
        )

        train_on_real_key: str = "train_on_real"
        if train_on_real_key not in self.metric_manager.metrics_per_generator:
            for iteration in range(self.iterations):
                logging.info(f"{iteration}: Training an RNN model using real data...")
                train_in_real_predictions: np.ndarray = self.predict_after_training(
                    self.training_features,
                    self.training_labels,
                    self.validation_features,
                    self.validation_labels,
                    self.testing_features,
                )

                self.register_prediction_results(
                    generator_name=train_on_real_key,
                    labels=self.testing_labels,
                    predictions=train_in_real_predictions,
                )
            self.update_summary_metrics(train_on_real_key)

        for iteration in range(self.iterations):
            x_synth_training, y_synth_training = self.generate_synthetic_data(
                training_data, generator_name, generator_instance
            )

            logging.info(
                f"{iteration}: Training an RNN model using synthetic data by "
                + f"{generator_name}..."
            )
            train_in_synth_predictions: np.ndarray = self.predict_after_training(
                x_synth_training,
                y_synth_training,
                self.validation_features,
                self.validation_labels,
                self.testing_features,
            )

            self.register_prediction_results(
                generator_name=generator_name,
                labels=self.testing_labels,
                predictions=train_in_synth_predictions,
            )
        self.update_summary_metrics(generator_name)

        average_mae: float = self.metric_manager.calculate_average(
            generator_name, self.metric_value_key
        )
        if self.best_generator_value is None or self.best_generator_value > average_mae:
            logging.info(
                f"{generator_name} (Avg. MAE {average_mae}) "
                f"is better than {self.best_generator_name} "
                f"(MAE {self.best_generator_value})!"
            )
            self.best_generator_name = generator_name
            self.best_generator_value = average_mae

    def register_prediction_results(
        self, generator_name: str, labels: np.ndarray, predictions: np.ndarray
    ) -> None:
        """Stores the metric values and predictions for later reporting."""
        # TODO: We can have a proper type here, instead of a Dict.
        generator_mae: float = float(mean_absolute_error(labels, predictions))
        iteration_metrics: dict[str, Any] = {
            self.metric_value_key: generator_mae,
            TRAINING_PREDICTIONS: predictions,
        }

        self.metric_manager.register_iteration(generator_name, iteration_metrics)

    def generate_synthetic_data(
        self,
        real_dataframe: pd.DataFrame,
        generator_name: str,
        generator_instance: TimeSeriesGenerator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates features and labels from generated synthetic data."""
        number_of_sequences: int = int(len(real_dataframe.index) / self.sequence_length)
        logging.info(
            f"Generating {number_of_sequences} "
            f"synthetic sequences using {generator_name}..."
        )
        synthetic_sequences: list = generator_instance.generate(number_of_sequences)
        synthetic_dataframe: pd.DataFrame = pd.concat(synthetic_sequences, axis=0)
        if len(synthetic_dataframe.index) != len(real_dataframe.index):
            logging.warn(
                f"We have {len(real_dataframe.index)} real records and"
                + f" {len(synthetic_dataframe.index)} synthetic ones. They should match"
            )

        preprocessed_synth_data: list[np.ndarray] = self.timegan_transformer.transform(
            synthetic_dataframe
        )

        x_synth_training, y_synth_training, _, _, _, _ = split_time_series(
            preprocessed_synth_data, self.sequence_length
        )

        return x_synth_training, y_synth_training

    def predict_after_training(
        self,
        x_training: np.ndarray,
        y_training: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        x_test: np.ndarray,
    ) -> np.ndarray:
        """Returns predictions over sequences, using a model trained with data."""
        rnn_model, early_stop_callback = self.create_rnn_model()

        _ = rnn_model.fit(
            x=x_training,
            y=y_training,
            epochs=self.epochs,
            validation_data=(x_validation, y_validation),
            callbacks=[early_stop_callback],
            batch_size=self.batch_size,
        )

        logging.info("Obtaining predictions...")
        predictions: np.ndarray = rnn_model.predict(x_test)
        return predictions

    def create_rnn_model(self) -> Tuple[Sequential, EarlyStopping]:
        """Returns a multi-out RNN for single-step forecasting."""
        # TODO: We are hardcoding the model here, but this should be configurable.
        optimizer: Adam = Adam()

        # TODO: They're using MAE as loss. According to DLP Book,
        # this should be MSE (it's smoother)
        loss: MeanAbsoluteError = MeanAbsoluteError()

        model: Sequential = Sequential(
            [LSTM(units=self.lstm_units), Dense(units=self.number_of_features)]
        )

        model.compile(loss=loss, optimizer=optimizer)

        early_stopping: EarlyStopping = EarlyStopping(
            monitor="val_loss", patience=self.patience
        )

        return model, early_stopping
