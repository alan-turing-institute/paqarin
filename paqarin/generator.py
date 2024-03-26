"""Module for types shared by all synthetic time series generators."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

SURROGATE_ITEM_ID: str = "surrogate_item_id"


class GeneratorTransformer(ABC):
    """Performs pre-processing and its reversal on data fed to the generator."""

    @abstractmethod
    def is_fitted(self) -> bool:
        """True if the transformer has been fitted with data, false otherwise."""

    @abstractmethod
    def fit(self, training_data: pd.DataFrame) -> None:
        """Fits the transformer using the data received."""

    @abstractmethod
    def inverse_transform(self, processed_data: Any) -> list:
        """Reverse the transformation applied to processed_data."""

    @abstractmethod
    def save(self, file_name: str) -> None:
        """Stores the transformer object in disk."""


class DummyTransformer(GeneratorTransformer):
    """This is a dummy transformer. It's meant to be used just a placeholder."""

    def fit(self, training_data: pd.DataFrame) -> None:
        """As a dummy transformer, this method is not implemented."""
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        """As a dummy transformer, this method is not implemented."""
        raise NotImplementedError()

    def inverse_transform(self, processed_data: list) -> list:
        """As a dummy transformer, this method is not implemented."""
        raise NotImplementedError()

    def save(self, file_name: str) -> None:
        """As a dummy transformer, this method is not implemented."""
        raise NotImplementedError()


# TODO: We need a way of showing training parameters to the user.
class GeneratorParameters(ABC):
    """Stores parameters for the generation of synthetic time series."""

    @property
    @abstractmethod
    def filename(self) -> Optional[str]:
        """Return the file where to store/load a generator's model."""

    @filename.setter
    def filename(self, value):
        pass


class TimeSeriesGenerator(ABC):
    """Abstract base class for synthetic time series generators."""

    @abstractmethod
    def save(self):
        """Stores the generator in disk."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Returns the name of the library used by paqarin for generation."""

    @property
    @abstractmethod
    def transformer(self) -> Optional[GeneratorTransformer]:
        """Returns the transformer object required by the generator."""

    @property
    @abstractmethod
    def generator(self) -> Any:
        """Returns the concrete generator implementation object."""

    @property
    @abstractmethod
    def parameters(self) -> GeneratorParameters:
        """Returns the object containing the generator's parameters."""

    @abstractmethod
    def fit(self, training_data: pd.DataFrame, **training_arguments: Any) -> None:
        """Fits the synthetic time series generator, using the provided data."""

    @abstractmethod
    def generate(self, number_of_sequences: int, **generation_arguments: Any) -> list:
        """Generates synthetic sequences."""
