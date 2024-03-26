"""Abstractions for handling multiple libraries for synthetic time series generation."""

import typing
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Type

import pandas as pd

from paqarin.generator import (
    GeneratorParameters,
    GeneratorTransformer,
    TimeSeriesGenerator,
)


class Provider(Enum):
    """The library provider of synthetic time series generation algorithms."""

    YDATA: str = "ydata"
    SYNTHCITY: str = "synthcity"
    SDV: str = "sdv"


class Method(Enum):
    """The algorithm for synthetic time series generation."""

    TIME_GAN: str = "timegan"
    DOPPLEGANGER: str = "doppleganger"
    PAR: str = "par"


class TimeSeriesGeneratorAdapter(ABC):
    """Adapter for a specific algorithm and implementation."""

    @property
    @abstractmethod
    def transformer(self) -> Optional[GeneratorTransformer]:
        """Returns the transformer object required by the adapter."""

    @abstractmethod
    def train_generator(
        self,
        generator_parameters: Any,
        training_data: pd.DataFrame,
        **training_arguments: Any,
    ) -> Any:
        """Return a trained generator according to a specific library."""

    @abstractmethod
    def save_generator(self, generator: Any, file_name: str) -> None:
        """Stores the trained generator in the library' required format."""

    @abstractmethod
    def generate_sequences(self, generator: Any, number_of_sequences: int) -> list:
        """Generates synthetic samples based using a third-party generator."""

    @staticmethod
    @abstractmethod
    def load_generator(generator_parameters: Any) -> TimeSeriesGenerator:
        """Loads a trained generator saved in disk."""


@typing.no_type_check
def get_generator_adapter(
    provider: Provider,
    method: Method,
    generator_parameters: Type[GeneratorParameters],
    transformer: Optional[GeneratorTransformer] = None,
) -> TimeSeriesGeneratorAdapter:
    """Returns an adapter instance, for given method, provider, and parameters."""
    # TODO: We should have defaults for methods we only have one provider.
    if provider.value == Provider.YDATA.value:
        from paqarin import ydata_adapter

        if method.value == Method.TIME_GAN.value:
            return ydata_adapter.TimeGanGeneratorAdapter(
                generator_parameters, transformer
            )
        elif method.value == Method.DOPPLEGANGER.value:
            return ydata_adapter.DoppleGanGerGeneratorAdapter(
                generator_parameters, transformer
            )
    elif provider.value == Provider.SYNTHCITY.value:
        from paqarin import synthcity_adapter

        if method.value == Method.TIME_GAN.value:
            return synthcity_adapter.TimeGanGeneratorAdapter(generator_parameters)
    elif provider.value == Provider.SDV.value:
        from paqarin import sdv_adapter

        if method.value == Method.PAR.value:
            return sdv_adapter.ParGeneratorAdapter(generator_parameters, transformer)

    raise ValueError(
        f"No adapter registered for provider {provider} and method {method}"
    )


def load_generator(
    provider: Provider,
    method: Method,
    generator_parameters: Type[GeneratorParameters],
    transformer: Optional[GeneratorTransformer] = None,
) -> TimeSeriesGenerator:
    """Loads a generator for disk."""
    generator_adapter: TimeSeriesGeneratorAdapter = get_generator_adapter(
        provider, method, generator_parameters, transformer
    )

    return generator_adapter.load_generator(generator_parameters)
