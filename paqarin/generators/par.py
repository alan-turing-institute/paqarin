"""Abstractions for generating time series using the PAR algorithm."""

import logging
from typing import Any, Optional

import pandas as pd

from paqarin.adapter import (
    Method,
    Provider,
    TimeSeriesGeneratorAdapter,
    get_generator_adapter,
)
from paqarin.generator import DummyTransformer, GeneratorParameters, TimeSeriesGenerator


class ParParameters(GeneratorParameters):
    """The parameters of the model and the training process.

    Attributes:
        item_id_column: Also called sequence key in PAR.
        epochs: By default, PAR uses 128 epochs.
        sequence_length: In PAR, we only use this for sampling. Provide None for using
            the length learned.
    """

    def __init__(
        self,
        epochs: Optional[int],
        sequence_length: Optional[int],
        item_id_columns: tuple[str],
        timestamp_column: str,
        filename: Optional[str],
    ):
        """Inits the PAR parameters."""
        self._filename: Optional[str] = filename
        self.sequence_length: Optional[int] = sequence_length
        self.epochs: Optional[int] = epochs

        self.item_id_columns: tuple[str] = item_id_columns
        self.timestamp_column: str = timestamp_column

    @property
    def filename(self) -> Optional[str]:
        """Returns the name of the file where to store the model."""
        return self._filename

    @filename.setter
    def filename(self, value) -> None:
        self._filename = value


class ParGenerator(TimeSeriesGenerator):
    """Wrapper class for implementations of the PAR approach."""

    def __init__(
        self,
        provider: str,
        generator_parameters: ParParameters,
        generator: Optional[Any] = None,
    ):
        """Inits the PAR generator."""
        self._generator: Optional[Any] = generator
        self._provider: str = provider
        self._parameters: ParParameters = generator_parameters

        # TODO: Workaround
        self._transformer = DummyTransformer()

        self.generator_adapter: TimeSeriesGeneratorAdapter = get_generator_adapter(
            provider=Provider(provider),
            method=Method.PAR,
            generator_parameters=generator_parameters,
            transformer=self._transformer,
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
    def transformer(self) -> Any:
        """Returns the transformer instance."""
        return self._transformer

    @property
    def parameters(self) -> ParParameters:
        """Returns the parameter values."""
        return self._parameters

    def fit(self, training_data: pd.DataFrame, **training_arguments: Any) -> None:
        """Trains a PAR generator for synthetic time series."""
        self._generator = self.generator_adapter.train_generator(
            self._parameters, training_data
        )

    def save(self):
        """Saves the trained generator to disk."""
        self.generator_adapter.save_generator(
            self._generator, self._parameters.filename
        )

        logging.info(
            "PAR from %s was saved at %s", self._provider, self._parameters.filename
        )

    def generate(
        self, number_of_sequences: int = 0, **generation_arguments
    ) -> list[pd.DataFrame]:
        """Produces synthetic sequences using this generator."""
        return self.generator_adapter.generate_sequences(
            self._generator, number_of_sequences, **generation_arguments
        )


def load_generator(provider: str, par_parameters: ParParameters) -> ParGenerator:
    """Loads a generator from disk. It's not implemented yet!"""
    raise NotImplementedError()
