"""Adapter code for running synthetic time series generation algorithms from Synthcity."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from synthcity.plugins import Plugins  # type: ignore
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader  # type: ignore
from synthcity.plugins.time_series.plugin_timegan import TimeGANPlugin  # type: ignore
from synthcity.utils.serialization import load_from_file  # type: ignore
from synthcity.utils.serialization import save_to_file

from paqarin.adapter import Provider, TimeSeriesGeneratorAdapter
from paqarin.generator import SURROGATE_ITEM_ID, GeneratorTransformer
from paqarin.generators import TimeGanGenerator, TimeGanParameters
from paqarin.utils import add_surrogate_key, normalise_sequences

OUTCOME_COLUMN: str = "outcome"
SEQUENCE_ID_COLUMN: str = "seq_id"


class TimeGanTransformer(GeneratorTransformer):
    """Transformer for Synthcity's TimeGAN implementation."""

    def __init__(
        self,
        item_id_column: Optional[str],
        timestamp_column: Optional[str],
        numerical_columns: Optional[list[str]],
    ) -> None:
        """Init for the TimeGAN transformer."""
        if (
            item_id_column is None
            or timestamp_column is None
            or numerical_columns is None
        ):
            raise ValueError(
                "Transformer requires values for "
                "item_id_column, timestamp_column, numerical_columns"
            )
        self.item_id_column: str = item_id_column
        self.timestamp_column: str = timestamp_column
        self.numerical_columns: list[str] = numerical_columns

        self.static_prefix: str = "seq_static_"
        self.outcome_prefix: str = "seq_out_"
        self.temporal_prefix: str = "seq_temporal_"

    def is_fitted(self) -> bool:
        """Returns true, if the transformer is fitted."""
        logging.info("This transformer does not need training")
        return True

    def fit(self, training_data: pd.DataFrame):
        """Fits the transformer. It's not implemented!"""
        raise NotImplementedError()

    def save(self, file_name: str) -> None:
        """Saves the transformer to disk. It's not implemented!"""
        logging.info(f"This transformer does not support saving to file {file_name}.")

    def inverse_transform(self, all_sequences: list[pd.DataFrame]) -> list:
        """Reverts the data transformations done by Synthcity."""
        result: list = []
        for current_sequence in all_sequences:
            current_sequence = current_sequence.drop(
                columns=[
                    SEQUENCE_ID_COLUMN,
                    f"{self.outcome_prefix}{OUTCOME_COLUMN}",
                ]
            )

            current_sequence = current_sequence.rename(
                columns={
                    "seq_time_id": self.timestamp_column,
                    f"{self.static_prefix}{self.item_id_column}": self.item_id_column,
                }
            )

            current_sequence = current_sequence.rename(
                columns={
                    f"{self.temporal_prefix}{column_name}": column_name
                    for column_name in self.numerical_columns
                }
            )

            result.append(current_sequence.reset_index(drop=True))

        return result


class TimeGanGeneratorAdapter(TimeSeriesGeneratorAdapter):
    """Adapter for Synthcity's TimeGAN implementation."""

    def __init__(self, time_gan_parameters: TimeGanParameters):
        """Inits the TimeGAN generator adapter."""
        self._transformer = TimeGanTransformer(
            item_id_column=time_gan_parameters.item_id_column,
            timestamp_column=time_gan_parameters.timestamp_column,
            numerical_columns=time_gan_parameters.numerical_columns,
        )

    @property
    def transformer(self) -> GeneratorTransformer:
        """Returns the transformer instance."""
        return self._transformer

    def train_generator(
        self,
        generator_parameters: TimeGanParameters,
        training_data: pd.DataFrame,
        **training_arguments: Any,
    ) -> TimeGANPlugin:
        """Trains and returns a TimeGAN generator from Synthcity."""
        if (
            generator_parameters.item_id_column is None
            or generator_parameters.timestamp_column is None
            or generator_parameters.numerical_columns is None
        ):
            raise ValueError(
                "Training TimeGAN requires "
                "item_id_column, timestamp_column, and numerical_columns  "
            )

        logging.info(f"Preprocessing {len(training_data)} training records ")
        temporal_data, observation_data, static_data, outcome = preprocess_data(
            generator_parameters.item_id_column,
            generator_parameters.timestamp_column,
            generator_parameters.numerical_columns,
            training_data,
            **training_arguments,
        )

        data_loader: TimeSeriesDataLoader = TimeSeriesDataLoader(
            temporal_data=temporal_data,
            observation_times=observation_data,
            static_data=static_data,
            outcome=outcome,
        )

        time_gan_plugin: TimeGANPlugin = Plugins().get(
            "timegan",
            n_iter=generator_parameters.epochs,
            batch_size=generator_parameters.batch_size,
            discriminator_lr=generator_parameters.learning_rate,
            generator_lr=generator_parameters.learning_rate,
            gamma_penalty=generator_parameters.gamma,
            generator_n_units_hidden=generator_parameters.latent_dimension,
            discriminator_n_units_hidden=generator_parameters.latent_dimension,
        )

        time_gan_plugin.fit(data_loader)

        return time_gan_plugin

    def save_generator(self, generator: TimeGANPlugin, file_name: str) -> None:
        """Saves the generator to file."""
        save_to_file(file_name, generator)

    def generate_sequences(
        self, generator: TimeGANPlugin, number_of_sequences: int
    ) -> list:
        """Generates synthetic time series using the TimeGAN algorithm."""
        all_sequences: pd.DataFrame = generator.generate(
            count=number_of_sequences
        ).dataframe()
        return [
            all_sequences.query(f"{SEQUENCE_ID_COLUMN} == {value}")
            for value in np.nditer(all_sequences[SEQUENCE_ID_COLUMN].unique())
        ]

    @staticmethod
    def load_generator(generator_parameters: TimeGanParameters) -> TimeGanGenerator:
        """WARNING: Memory hungry method. It can take 2GB of your RAM."""
        time_gan_plugin: TimeGANPlugin = load_from_file(generator_parameters.filename)
        logging.info(
            f"TimeGAN implementation loaded from {generator_parameters.filename}"
        )

        return TimeGanGenerator(
            provider=Provider.SYNTHCITY.value,
            generator_parameters=generator_parameters,
            generator=time_gan_plugin,
        )


class TimeVaeTransformer(GeneratorTransformer):
    """Transformer for Synthcity's TimeVAE implementation."""

    def __init__(
        self,
        item_id_column: Optional[str],
        timestamp_column: Optional[str],
        numerical_columns: Optional[list[str]],
    ) -> None:
        """Init for the TimeVAE transformer."""
        if (
            item_id_column is None
            or timestamp_column is None
            or numerical_columns is None
        ):
            raise ValueError(
                "Transformer requires values for "
                "item_id_column, timestamp_column, numerical_columns"
            )
        self.item_id_column: str = item_id_column
        self.timestamp_column: str = timestamp_column
        self.numerical_columns: list[str] = numerical_columns

        self.static_prefix: str = "seq_static_"
        self.outcome_prefix: str = "seq_out_"
        self.temporal_prefix: str = "seq_temporal_"

    def is_fitted(self) -> bool:
        """Returns true, if the transformer is fitted."""
        logging.info("This transformer does not need training")
        return True

    def fit(self, training_data: pd.DataFrame):
        """Fits the transformer. It's not implemented!"""
        raise NotImplementedError()

    def save(self, file_name: str) -> None:
        """Saves the transformer to disk. It's not implemented!"""
        logging.info(f"This transformer does not support saving to file {file_name}.")

    def inverse_transform(self, all_sequences: list[pd.DataFrame]) -> list:
        """Reverts the data transformations done by Synthcity."""
        result: list = []
        for current_sequence in all_sequences:
            current_sequence = current_sequence.drop(
                columns=[
                    SEQUENCE_ID_COLUMN,
                    f"{self.outcome_prefix}{OUTCOME_COLUMN}",
                ]
            )

            current_sequence = current_sequence.rename(
                columns={
                    "seq_time_id": self.timestamp_column,
                    f"{self.static_prefix}{self.item_id_column}": self.item_id_column,
                }
            )

            current_sequence = current_sequence.rename(
                columns={
                    f"{self.temporal_prefix}{column_name}": column_name
                    for column_name in self.numerical_columns
                }
            )

            result.append(current_sequence.reset_index(drop=True))

        return result


class TimeVaeGeneratorAdapter(TimeSeriesGeneratorAdapter):
    """Adapter for Synthcity's TimeVAE implementation."""

    def __init__(self, time_vae_parameters: TimeVaeParameters):
        """Inits the TimeVAE generator adapter."""
        self._transformer = TimeVaeTransformer(
            item_id_column=time_vae_parameters.item_id_column,
            timestamp_column=time_vae_parameters.timestamp_column,
            numerical_columns=time_vae_parameters.numerical_columns,
        )

    @property
    def transformer(self) -> GeneratorTransformer:
        """Returns the transformer instance."""
        return self._transformer

    def train_generator(
        self,
        generator_parameters: TimeVaeParameters,
        training_data: pd.DataFrame,
        **training_arguments: Any,
    ) -> TimeVAEPlugin:
        """Trains and returns a TimeVAE generator from Synthcity."""
        if (
            generator_parameters.item_id_column is None
            or generator_parameters.timestamp_column is None
            or generator_parameters.numerical_columns is None
        ):
            raise ValueError(
                "Training TimeVAE requires "
                "item_id_column, timestamp_column, and numerical_columns  "
            )

        logging.info(f"Preprocessing {len(training_data)} training records ")
        temporal_data, observation_data, static_data, outcome = preprocess_data(
            generator_parameters.item_id_column,
            generator_parameters.timestamp_column,
            generator_parameters.numerical_columns,
            training_data,
            **training_arguments,
        )

        data_loader: TimeSeriesDataLoader = TimeSeriesDataLoader(
            temporal_data=temporal_data,
            observation_times=observation_data,
            static_data=static_data,
            outcome=outcome,
        )

        time_vae_plugin: TimeVAEPlugin = Plugins().get(
            "timevae",
            n_iter=generator_parameters.epochs,
            batch_size=generator_parameters.batch_size,
            discriminator_lr=generator_parameters.learning_rate,
            generator_lr=generator_parameters.learning_rate,
            gamma_penalty=generator_parameters.gamma,
            generator_n_units_hidden=generator_parameters.latent_dimension,
            discriminator_n_units_hidden=generator_parameters.latent_dimension,
        )

        time_vae_plugin.fit(data_loader)

        return time_vae_plugin

    def save_generator(self, generator: TimeVAEPlugin, file_name: str) -> None:
        """Saves the generator to file."""
        save_to_file(file_name, generator)

    def generate_sequences(
        self, generator: TimeVAEPlugin, number_of_sequences: int
    ) -> list:
        """Generates synthetic time series using the TimeVAE algorithm."""
        all_sequences: pd.DataFrame = generator.generate(
            count=number_of_sequences
        ).dataframe()
        return [
            all_sequences.query(f"{SEQUENCE_ID_COLUMN} == {value}")
            for value in np.nditer(all_sequences[SEQUENCE_ID_COLUMN].unique())
        ]

    @staticmethod
    def load_generator(generator_parameters: TimeVaeParameters) -> TimeVaeGenerator:
        """WARNING: Memory hungry method. It can take 2GB of your RAM."""
        time_vae_plugin: TimeVAEPlugin = load_from_file(generator_parameters.filename)
        logging.info(
            f"TimeVAE implementation loaded from {generator_parameters.filename}"
        )

        return TimeVaeGenerator(
            provider=Provider.SYNTHCITY.value,
            generator_parameters=generator_parameters,
            generator=time_vae_plugin,
        )



def preprocess_data(
    item_id_column: str,
    timestamp_column: str,
    numerical_columns: list[str],
    raw_data: pd.DataFrame,
    nan_value: Any = 0.0,
    date_format: str = "%d/%m/%Y",
) -> tuple[list[pd.DataFrame], list[list], pd.DataFrame, pd.DataFrame]:
    """Pre-process time series data according to Synthcity's requirements."""
    # TODO: Experimental!!! make sequences the same size:
    normalized_data, _, _ = normalise_sequences(
        item_id_column, timestamp_column, "D", raw_data, date_format=date_format
    )
    # Synthcity fails when using Timestamp values.
    normalized_data[timestamp_column] = normalized_data[timestamp_column].dt.strftime(
        date_format
    )

    extended_data: pd.DataFrame = add_surrogate_key((item_id_column,), normalized_data)
    extended_data = extended_data.fillna(nan_value)

    observation_data: list[list] = []
    temporal_data: list[pd.DataFrame] = []
    for item_id in extended_data[SURROGATE_ITEM_ID].unique():
        item_temporal_data: pd.DataFrame = extended_data[
            extended_data[SURROGATE_ITEM_ID] == item_id
        ]

        item_observations: list = item_temporal_data[timestamp_column].to_list()
        observation_data.append(item_observations)

        item_temporal_data.set_index(timestamp_column, inplace=True)
        temporal_data.append(item_temporal_data[numerical_columns])

    # TODO: Support multiple attributes for the entity.
    static_data: pd.DataFrame = extended_data[[item_id_column]].drop_duplicates()
    static_data.reset_index(inplace=True, drop=True)

    # Only to comply with synthcity's API
    # Otherwise, training fails.
    outcome: pd.DataFrame = pd.DataFrame(
        {
            OUTCOME_COLUMN: [0 for _ in range(len(static_data))],
        }
    )

    return temporal_data, observation_data, static_data, outcome
