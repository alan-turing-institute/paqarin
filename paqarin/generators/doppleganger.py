"""Abstractions related to the DoppleGANger approach to synthetic time generation."""

import logging
from typing import Any, List, Optional, Tuple

import pandas as pd

from paqarin.adapter import (
    Method,
    Provider,
    TimeSeriesGeneratorAdapter,
    get_generator_adapter,
)
from paqarin.generator import (
    GeneratorParameters,
    GeneratorTransformer,
    TimeSeriesGenerator,
)

DATE_INDEX_KEY: str = "date_index"


class DoppleGanGerParameters(GeneratorParameters):
    """The parameters of the model and the training process.

    Attributes:
        batch_size: Samples per batch.
        epochs: Training iterations.
        latent_dimension: The dimension of noise for generating attributes.
        exponential_decay_rates: Parameters of the Adam Optimiser.
        sequence_length: Must be a multiple of sample length. Also, training
            samples must be a multiple of sequence length. While nor the original
            implementation enforces a fixed sequence length, they do it in ydata's
            version. The recommended value in the paper is 5.
        packing_degree: Packing degree. Used in PacGAN for addressing mode collapse.
        sample_length: The time series batch size, i.e. the number of time
            steps generated at each RNN rollout.
        steps_per_batch: Number of steps per batch.
        measurement_columns: In DoppleGANger, samples contain in metadata
            (or attributes, a list of values) and measurements (or features, a time
            series of measurements).
    """

    # TODO: There's overlap between TimeGAN and DoppleGANger. Maybe we can have
    # another abstraction.
    def __init__(
        self,
        batch_size: Optional[int],
        learning_rate: Optional[float],
        latent_dimension: Optional[int],
        exponential_decay_rates: Optional[Tuple[float, float]],
        wgan_weight: Optional[float],
        packing_degree: Optional[int],
        epochs: Optional[int],
        sequence_length: Optional[int],
        sample_length: Optional[int],
        steps_per_batch: Optional[int],
        numerical_columns: Optional[List[str]],
        categorical_columns: Optional[List[str]],
        measurement_columns: Optional[List[str]],
        filename: Optional[str],
    ):
        """Inits DoppleGANger parameters."""
        self.batch_size: Optional[int] = batch_size
        self.learning_rate: Optional[float] = learning_rate
        self.latent_dimension: Optional[int] = latent_dimension
        self.exponential_decay_rates: Optional[
            Tuple[float, float]
        ] = exponential_decay_rates
        self.wgan_weight: Optional[float] = wgan_weight
        self.packing_degree: Optional[int] = packing_degree
        self.epochs: Optional[int] = epochs
        # In ydata's implementation, sequence length must be a multiple of
        # sequence length.
        self.sequence_length: Optional[int] = sequence_length
        self.sample_length: Optional[int] = sample_length
        self.steps_per_batch: Optional[int] = steps_per_batch
        self.numerical_columns: Optional[List[str]] = numerical_columns
        self.categorical_columns: Optional[List[str]] = categorical_columns
        self.measurement_columns: Optional[List[str]] = measurement_columns

        self._filename: Optional[str] = filename

    @property
    def filename(self) -> Optional[str]:
        """The name of the file where to save the trained model."""
        return self._filename

    @filename.setter
    def filename(self, value) -> None:
        self._filename = value


class DoppleGangerGenerator(TimeSeriesGenerator):
    """Wrapper class for implementations of the DoppleGANger approach."""

    def __init__(
        self,
        provider: str,
        generator_parameters: DoppleGanGerParameters,
        generator: Optional[Any] = None,
        transformer: Optional[GeneratorTransformer] = None,
    ):
        """Inits the DoppleGANger generator."""
        self._generator: Optional[Any] = generator
        self._provider: str = provider

        self._parameters: DoppleGanGerParameters = generator_parameters

        self.generator_adapter: TimeSeriesGeneratorAdapter = get_generator_adapter(
            provider=Provider(provider),
            method=Method.DOPPLEGANGER,
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
    def transformer(self) -> Any:
        """Returns the transformer instance."""
        return self.generator_adapter.transformer

    @property
    def parameters(self) -> DoppleGanGerParameters:
        """Returns the parameter values."""
        return self._parameters

    def fit(self, training_data: pd.DataFrame, **training_arguments: Any) -> None:
        """Trains a DoppleGANger generator for synthetic time series."""
        self._generator = self.generator_adapter.train_generator(
            self._parameters, training_data, **training_arguments
        )

    def save(self) -> None:
        """Saves the trained DoppleGANger generator to a file."""
        # TODO: Also save the transformer here.

        if self._parameters.filename is None:
            raise ValueError("Provide a valid filename value")

        file_name: str = self._parameters.filename
        if self._generator is not None:
            self.generator_adapter.save_generator(self._generator, file_name)
            logging.info(f"DoppleGANger from {self._provider} was saved at {file_name}")
        else:
            logging.info("The generator wasn't trained.")

    def generate(self, number_of_sequences: int, **generation_arguments) -> list:
        """Produces DoppleGANger synthetic sequences using this generator."""
        generated_sequences: list[
            pd.DataFrame
        ] = self.generator_adapter.generate_sequences(
            self._generator, number_of_sequences
        )  # type: ignore

        current_transformer: Optional[
            GeneratorTransformer
        ] = self.generator_adapter.transformer

        if current_transformer is not None:
            generated_sequences = current_transformer.inverse_transform(
                generated_sequences
            )
        else:
            logging.info("Using sequences as-is, since there's no transformer")

        if DATE_INDEX_KEY in generation_arguments:
            date_index: pd.DatetimeIndex = generation_arguments[DATE_INDEX_KEY]
            for dataframe in generated_sequences:
                dataframe[date_index.name] = date_index
        else:
            logging.info("No date index provided to be included in generate sequence.")

        return generated_sequences
