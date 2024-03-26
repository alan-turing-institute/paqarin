"""Adapter for ydata-synthetic generator implementations."""
import logging
import pickle
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore
from sklearn.exceptions import NotFittedError  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from tensorflow import keras  # type: ignore
from ydata_synthetic.preprocessing.timeseries.utils import (  # type: ignore
    real_data_loading,
)
from ydata_synthetic.synthesizers import ModelParameters  # type: ignore
from ydata_synthetic.synthesizers import TrainParameters
from ydata_synthetic.synthesizers.timeseries import (  # type: ignore
    TimeSeriesSynthesizer,
)

from paqarin.adapter import Provider, TimeSeriesGeneratorAdapter
from paqarin.generator import GeneratorTransformer
from paqarin.generators import (
    DoppleGangerGenerator,
    DoppleGanGerParameters,
    TimeGanGenerator,
    TimeGanParameters,
)

YDATA_PROVIDER: str = "ydata"
TRANSFORMER_SUFFIX: str = ".transformer"


class DoppleGanGerTransformer(GeneratorTransformer):
    """Transformer for the DoppleGANger algorithm.

    It only applies a max-min scaling (0 to 1) of numerical features, as this seems to be
    expected from the original DoppleGANger implementation.
    """

    def __init__(self, numerical_columns: Optional[list[str]]):
        """Inits the DoppleGANger transformer."""
        self.min_max_scaler: MinMaxScaler = MinMaxScaler()
        self.numerical_columns: Optional[list[str]] = numerical_columns

    def fit(self, training_data: pd.DataFrame) -> None:
        """Fits the transformer using the provided training data."""
        if self.numerical_columns is None:
            raise ValueError("Please provide the list of numerical columns")

        features_to_scale: pd.DataFrame = training_data[self.numerical_columns]
        self.min_max_scaler.fit(features_to_scale)

    def is_fitted(self) -> bool:
        """True if the transformer was trained, false otherwise."""
        try:
            check_is_fitted(self.min_max_scaler)
            return True
        except NotFittedError:
            return False

    def transform(self, unscaled_data: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data to the format required by the generator."""
        logging.info(f"Scaling {len(unscaled_data)} rows...")

        after_transform_data: pd.DataFrame = unscaled_data.copy()
        after_transform_data[self.numerical_columns] = self.min_max_scaler.transform(
            after_transform_data[self.numerical_columns]
        )

        return after_transform_data

    def inverse_transform(
        self, processed_data: list[pd.DataFrame]
    ) -> list[pd.DataFrame]:
        """Reverses the transformation applied."""
        logging.info(f"De-scaling {len(processed_data)} sequences...")

        descaled_data: list[pd.DataFrame] = []
        for scaled_data in processed_data:
            inverse_transformed_data: pd.DataFrame = scaled_data.copy()

            inverse_transformed_data[
                self.numerical_columns
            ] = self.min_max_scaler.inverse_transform(
                inverse_transformed_data[self.numerical_columns]
            )

            descaled_data.append(inverse_transformed_data)

        return descaled_data

    def save(self, file_name: str) -> None:
        """Saves the transformer to disk. Not yet implemented!"""
        # TODO: Implement this later.
        raise NotImplementedError()


# IMPORTANT!! ydata's implementation disables eager execution:
# https://www.tensorflow.org/api_docs/python/tf/compat/v1/disable_eager_execution
class DoppleGanGerGeneratorAdapter(TimeSeriesGeneratorAdapter):
    """Adapter for YData-synthetic's implementation of the DoppleGANger algorithm."""

    def __init__(
        self,
        doppleganger_parameters: DoppleGanGerParameters,
        transformer: DoppleGanGerTransformer,
    ):
        """Inits the DoppleGANger generator adapter."""
        if transformer is not None:
            self._transformer = transformer
        else:
            self._transformer = DoppleGanGerTransformer(
                numerical_columns=doppleganger_parameters.numerical_columns
            )

    @property
    def transformer(self) -> GeneratorTransformer:
        """Returns the transformer instance."""
        return self._transformer

    def train_generator(
        self,
        generator_parameters: DoppleGanGerParameters,
        training_data: pd.DataFrame,
        **training_arguments: Any,
    ) -> TimeSeriesSynthesizer:
        """Trains the generator according to the provided parameters and data."""
        keras.backend.clear_session()
        scaled_training_data: pd.DataFrame = self._transformer.transform(training_data)
        return train_doppleganger(
            generator_parameters, training_data=scaled_training_data
        )

    def save_generator(self, generator: TimeSeriesSynthesizer, file_name: str) -> None:
        """Saves the generator to disk."""
        save(generator, file_name)

    def generate_sequences(
        self, generator: TimeSeriesSynthesizer, number_of_sequences: int
    ) -> list:
        """Generates synthetic time series sequences."""
        return generate(generator, number_of_sequences)

    @staticmethod
    def load_generator(
        generator_parameters: DoppleGanGerParameters,
    ) -> DoppleGangerGenerator:
        """Loads the generator from disk."""
        return load_doppleganger_generator(generator_parameters)


class TimeGanGeneratorAdapter(TimeSeriesGeneratorAdapter):
    """Adapter for YData-Synthetic implementation of the TimeGAN algorithm."""

    def __init__(
        self,
        time_gan_parameters: TimeGanParameters,
        transformer: Optional[GeneratorTransformer],
    ):
        """Inits the TimeGAN generator adapter."""
        if transformer is not None:
            self._transformer = transformer
        else:
            self._transformer = TimeGanTransformer(
                sequence_length=time_gan_parameters.sequence_length,
                numerical_columns=time_gan_parameters.numerical_columns,
            )

    @property
    def transformer(self) -> GeneratorTransformer:
        """Returns the transformer attribute."""
        return self._transformer

    def train_generator(
        self,
        generator_parameters: TimeGanParameters,
        training_data: pd.DataFrame,
        **training_arguments: Any,
    ) -> TimeSeriesSynthesizer:
        """Trains the generator using the given parameters and training data."""
        return train_timegan(generator_parameters, training_data)

    def save_generator(self, generator: TimeSeriesSynthesizer, file_name: str) -> None:
        """Saves the generator to disk."""
        save(generator, file_name)

    def generate_sequences(
        self, generator: TimeSeriesSynthesizer, number_of_sequences: int
    ) -> list:
        """Samples synthetic sequences from the generator."""
        return generate(generator, number_of_sequences)

    @staticmethod
    def load_generator(generator_parameters: TimeGanParameters) -> TimeGanGenerator:
        """Loads a generator from disk."""
        return load_timegan_generator(generator_parameters)


class TimeGanTransformer(GeneratorTransformer):
    """Handles pre-processing logic for the TimeGAN approach."""

    def __init__(
        self,
        sequence_length: Optional[int],
        numerical_columns: Optional[list[str]],
    ) -> None:
        """Inits the TimeGAN transformer."""
        if sequence_length is None or numerical_columns is None:
            raise ValueError(
                "The TimeGAN transformer needs values for sequence_length "
                + "and numerical_columns"
            )

        self.min_max_scaler: MinMaxScaler = MinMaxScaler()
        self.sequence_length: int = sequence_length
        self.numerical_columns: list[str] = numerical_columns

    def fit(self, training_data: pd.DataFrame) -> None:
        """Configures prep-processing logic according to the provided data."""
        self.min_max_scaler.fit(training_data)

    def is_fitted(self) -> bool:
        """True if the transformer was trained, false otherwise."""
        try:
            check_is_fitted(self.min_max_scaler)
            return True
        except NotFittedError:
            return False

    def transform(self, unprocessed_data: pd.DataFrame) -> list[np.ndarray]:
        """Applies the pre-processing logic to the provided data."""
        from paqarin import ydata_adapter

        return ydata_adapter.preprocess_timegan(
            unprocessed_data, self.numerical_columns, self.sequence_length
        )

    def inverse_transform(
        self, processed_data: Union[List[pd.DataFrame], List[np.ndarray]]
    ) -> list[pd.DataFrame]:
        """Reverse the pre-processing logic to the provided data."""
        result: list[pd.DataFrame] = [
            pd.DataFrame.from_records(
                self.min_max_scaler.inverse_transform(sequence),
                columns=self.numerical_columns,
            )
            for sequence in processed_data
        ]

        return result

    def save(self, generator_file: str) -> None:
        """Saves the transformer to disk."""
        if self.is_fitted():
            transformer_file: str = f"{generator_file}{TRANSFORMER_SUFFIX}"
            with open(transformer_file, "wb") as output:
                pickle.dump(self, output)

            logging.info("TimeGAN transformer was saved at %s", transformer_file)
        else:
            logging.info("Transformer not fitted, so we are not saving it.")


def preprocess_timegan(
    training_data: pd.DataFrame, data_columns: List[str], sequence_length: int
) -> List[np.ndarray]:
    """Processes data in the format required for training.

    The ydata-synthetic implementation of the algorithm does the following:

        1. Reverses the dataset.
        2. Applies MinMaxScalar to numerical_columns.
        3. Splits the transformed dataset into sequences of length sequence_length.
        4. Then it randomizes the output.
    """
    reversed_sequences: list[np.ndarray] = real_data_loading(
        training_data[data_columns].values, seq_len=sequence_length
    )
    # We're undoing the reversal since it's not necessary:
    # https://github.com/jsyoon0823/TimeGAN/issues/30
    # They're also skipping the first element. This might be a bug.
    preprocessed_data: list[np.ndarray] = [
        np.flip(sequence, axis=0) for sequence in reversed(reversed_sequences)
    ]
    return preprocessed_data


def train_timegan(
    time_gan_parameters: TimeGanParameters, training_data: pd.DataFrame
) -> TimeSeriesSynthesizer:
    """Creates an trains a TimeGAN generator provided by ydata."""
    # TODO: Extract this logic to an external function to make this testable.
    model_parameters: ModelParameters = ModelParameters(
        batch_size=time_gan_parameters.batch_size,
        lr=time_gan_parameters.learning_rate,
        noise_dim=time_gan_parameters.noise_dimension,
        layers_dim=time_gan_parameters.layers_dimension,
        latent_dim=time_gan_parameters.latent_dimension,
        gamma=time_gan_parameters.gamma,
    )

    train_parameters: TrainParameters = TrainParameters(
        epochs=time_gan_parameters.epochs,
        sequence_length=time_gan_parameters.sequence_length,
        number_sequences=time_gan_parameters.number_of_sequences,
    )

    synthesizer: TimeSeriesSynthesizer = TimeSeriesSynthesizer(
        modelname="timegan", model_parameters=model_parameters
    )

    print("Training YData-Synthetic TimeGAN implementation...")
    synthesizer.fit(
        training_data,
        train_parameters,  # type: ignore
        num_cols=time_gan_parameters.numerical_columns,
    )

    return synthesizer


def train_doppleganger(
    doppleganger_parameters: DoppleGanGerParameters, training_data: pd.DataFrame
) -> TimeSeriesSynthesizer:
    """Creates and trains a DoppleGANger generator provided by ydata."""
    model_parameters: ModelParameters = ModelParameters(
        batch_size=doppleganger_parameters.batch_size,
        lr=doppleganger_parameters.learning_rate,
        betas=doppleganger_parameters.exponential_decay_rates,
        latent_dim=doppleganger_parameters.latent_dimension,
        gp_lambda=doppleganger_parameters.wgan_weight,
        pac=doppleganger_parameters.packing_degree,
    )

    train_parameters: TrainParameters = TrainParameters(
        epochs=doppleganger_parameters.epochs,
        sequence_length=doppleganger_parameters.sequence_length,
        sample_length=doppleganger_parameters.sample_length,
        rounds=doppleganger_parameters.steps_per_batch,
        measurement_cols=doppleganger_parameters.measurement_columns,
    )

    synthesizer: TimeSeriesSynthesizer = TimeSeriesSynthesizer(
        modelname="doppelganger", model_parameters=model_parameters
    )

    logging.info("Training YData-Synthetic DoppleGANger implementation...")
    synthesizer.fit(
        training_data,
        train_parameters,  # type: ignore
        num_cols=doppleganger_parameters.numerical_columns,
        cat_cols=doppleganger_parameters.categorical_columns,
    )

    return synthesizer


def save(synthesizer: TimeSeriesSynthesizer, filename: str) -> None:
    """Store the TimeGAN model in a file."""
    synthesizer.save(filename)  # type: ignore


def load(filename: str) -> TimeSeriesSynthesizer:
    """Loads a YData generator instance from a file or directory."""
    return TimeSeriesSynthesizer.load(filename)  # type: ignore


def generate(generator: TimeSeriesSynthesizer, number_of_sequences: int) -> list:
    """Generates synthetic sequences using a TimeGAN generator."""
    return generator.sample(number_of_sequences)  # type: ignore


# TODO: Load from cloud bucket
def load_timegan_generator(time_gan_parameters: TimeGanParameters) -> TimeGanGenerator:
    """Loads a TimeGAN generator from a file."""
    if time_gan_parameters.filename is None:
        raise ValueError("Provide a valid filename value")

    file_name: str = time_gan_parameters.filename

    time_gan_implementation: Optional[Any] = None
    time_gan_transformer: Optional[GeneratorTransformer] = None

    logging.info("Loading TimeGAN implementation from ydata stored at %s", file_name)
    time_gan_implementation = load(file_name)

    transformer_file: str = f"{time_gan_parameters.filename}{TRANSFORMER_SUFFIX}"
    logging.info("Loading TimeGAN transformer from: %s", transformer_file)
    with open(transformer_file, "rb") as output:
        time_gan_transformer = pickle.load(output)

    time_gan_generator: TimeGanGenerator = TimeGanGenerator(
        provider=YDATA_PROVIDER,
        generator_parameters=time_gan_parameters,
        generator=time_gan_implementation,
        transformer=time_gan_transformer,
    )

    return time_gan_generator


def load_doppleganger_generator(
    doppleganger_parameters: DoppleGanGerParameters,
) -> DoppleGangerGenerator:
    """Loads a DoppleGanger generator from a directory."""
    if doppleganger_parameters.filename is None:
        raise ValueError("Provide a valid filename value")

    file_name: str = doppleganger_parameters.filename
    doppleganger_implementation: Optional[Any] = None

    logging.info(
        f"Loading DoppleGANger implementation from YData stored at {file_name}"
    )
    with tf.Graph().as_default() as _:
        doppleganger_implementation = load(file_name)

        doppleganger_generator: DoppleGangerGenerator = DoppleGangerGenerator(
            provider=Provider.YDATA.value,
            generator_parameters=doppleganger_parameters,
            generator=doppleganger_implementation,
        )

        return doppleganger_generator
