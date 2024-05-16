"""Abstractions related to the TimeGAN approach to synthetic time generation."""

import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from tensorflow import keras  # type: ignore

from paqarin.adapter import (
    Method,
    Provider,
    TimeSeriesGeneratorAdapter,
    get_generator_adapter,
)
from paqarin.generator import (
    DummyTransformer,
    GeneratorParameters,
    GeneratorTransformer,
    TimeSeriesGenerator,
)


class TimeGanParameters(GeneratorParameters):
    """The parameters of the model and the training process.

    Attributes:
        item_id_column: Required by Synthcity. It's the entity identifier.
        timestamp_column: Required by Synthcity. The column containing the timestamp.
        batch_size: Samples per batch. In synthcity, default value is 64.
        latent_dimension: Hidden state dimension. In Synthcity, we map this value to
            generator_n_units_hidden (default 150)  and discriminator_n_units_hidden
            (default 300)
        gamma:  Latent representation penalty. A discriminator loss parameter.
        epochs: Training iterations.
        sequence_length: Sequence length.
        number_of_sequences:  A parameter of the recovery network.
    """

    def __init__(
        self,
        numerical_columns: Optional[List[str]],
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        latent_dimension: Optional[int] = None,
        gamma: Optional[int] = None,
        epochs: Optional[int] = None,
        sequence_length: Optional[int] = None,
        number_of_sequences: Optional[int] = None,
        item_id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        noise_dimension: Optional[int] = None,
        layers_dimension: Optional[int] = None,
        filename: Optional[str] = None,
    ):
        """Inits TimeGAN Parameters."""
        self.batch_size = batch_size
        # TODO: LR Seems to be very influential in model performance!
        # Let's explore how to fine-tune it.
        self.learning_rate = learning_rate
        self.noise_dimension = noise_dimension
        self.layers_dimension = layers_dimension
        self.latent_dimension = latent_dimension
        self.gamma = gamma
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.number_of_sequences = number_of_sequences
        self.numerical_columns = numerical_columns
        self.item_id_column = item_id_column
        self.timestamp_column = timestamp_column
        self._filename = filename

    @property
    def filename(self) -> Optional[str]:
        """Returns the name of the file where to store the model."""
        return self._filename

    @filename.setter
    def filename(self, value) -> None:
        self._filename = value


class TimeGanGenerator(TimeSeriesGenerator):
    """Wrapper class for implementations of the TimeGAN approach."""

    def __init__(
        self,
        provider: str,
        generator_parameters: TimeGanParameters,
        generator: Optional[Any] = None,
        transformer: Optional[GeneratorTransformer] = None,
    ):
        """Inits the TimeGAN generator."""
        self._generator: Optional[Any] = generator
        self._provider: str = provider

        self._parameters: TimeGanParameters = generator_parameters

        self.generator_adapter: TimeSeriesGeneratorAdapter = get_generator_adapter(
            provider=Provider(provider),
            method=Method.TIME_GAN,
            generator_parameters=generator_parameters,
            transformer=transformer,
        )

    @property
    def generator(self) -> Any:
        """Returns the generator instance."""
        return self._generator

    @property
    def provider(self) -> str:
        """Returns the provider value."""
        return self._provider

    @property
    def transformer(self) -> Optional[GeneratorTransformer]:
        """Returns the transformer instance."""
        return self.generator_adapter.transformer

    @property
    def parameters(self) -> TimeGanParameters:
        """Returns the parameter values."""
        return self._parameters

    def fit(self, training_data: pd.DataFrame, **training_arguments: Any) -> None:
        """Trains a TimeGAN generator for synthetic time series."""
        keras.backend.clear_session()

        # YData does not need preprocessing for training
        self._generator = self.generator_adapter.train_generator(
            self._parameters, training_data, **training_arguments
        )

    def save(self):
        """Saves the trained generator to a file."""
        self.save_model()
        self.save_transformer()

    def save_model(self) -> None:
        """Saves the generator to a file. It does not include the transformer."""
        if self._parameters.filename is None:
            raise ValueError("Provide a valid filename value")

        file_name: str = self._parameters.filename
        if self._generator is not None:
            self.generator_adapter.save_generator(self._generator, file_name)
            logging.info("TimeGAN from %s was saved at %s", self._provider, file_name)
        else:
            logging.info("The generator wasn't trained.")

    def save_transformer(self) -> None:
        """Saves the transformer to a file."""
        transformer_to_save: Optional[GeneratorTransformer] = (
            self.generator_adapter.transformer
        )
        if (
            (transformer_to_save is not None)
            and (not isinstance(transformer_to_save, DummyTransformer))
            and (self._parameters.filename is not None)
        ):
            transformer_to_save.save(self._parameters.filename)
        else:
            logging.info(
                f"Cannot save transformer {transformer_to_save} to"
                f" file {self._parameters.filename}"
            )

    def generate(
        self, number_of_sequences: int, **generation_arguments: Any
    ) -> list[np.ndarray]:
        """Produces synthetic sequences using this generator."""
        preprocessed_sequences: list[
            np.ndarray
        ] = self.generator_adapter.generate_sequences(
            self._generator, number_of_sequences
        )  # type: ignore

        current_transformer: Optional[GeneratorTransformer] = (
            self.generator_adapter.transformer
        )
        # TODO: We can refactor this logic.
        if (current_transformer is not None) and (
            not isinstance(current_transformer, DummyTransformer)
        ):
            return current_transformer.inverse_transform(preprocessed_sequences)

        logging.info("Returning sequences as-is, since there's no transformer")
        return preprocessed_sequences
