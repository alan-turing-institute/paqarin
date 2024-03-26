"""Adapter code for running generation algorithms from the SDV library."""

import logging
from typing import Any, Optional

import pandas as pd
from sdv.metadata import SingleTableMetadata  # type: ignore
from sdv.sequential import PARSynthesizer  # type: ignore

from paqarin.adapter import Provider, TimeSeriesGeneratorAdapter
from paqarin.generator import GeneratorTransformer
from paqarin.generators import ParGenerator, ParParameters
from paqarin.utils import add_surrogate_key  # type: ignore

SURROGATE_ITEM_ID: str = "surrogate_item_id"
CONTEXT_DATAFRAME_KEY: str = "context"


class ParGeneratorAdapter(TimeSeriesGeneratorAdapter):
    """Adapter for SDV's PAR implementation."""

    def __init__(
        self, par_parameters: ParParameters, transformer: GeneratorTransformer
    ):
        """Inits the PAR generator adapter."""
        self.par_parameters: ParParameters = par_parameters
        self._transformer = transformer  # No transformer needed for PAR-SDV

    def train_generator(
        self,
        generator_parameters: ParParameters,
        training_data: pd.DataFrame,
        **training_arguments: Any,
    ) -> PARSynthesizer:
        """Trains and returns a PAR generator from SDV."""
        return train_par(generator_parameters, training_data)

    def generate_sequences(
        self,
        generator: PARSynthesizer,
        number_of_sequences: int,
        **generation_arguments,
    ) -> list:
        """Generates synthetic time series using the PAR algorithm."""
        if CONTEXT_DATAFRAME_KEY in generation_arguments:
            context_dataframe: pd.DataFrame = generation_arguments[
                CONTEXT_DATAFRAME_KEY
            ]

            logging.info(
                f"Size of the context dataframe: {len(context_dataframe)} "
                f"Sequence length: {self.par_parameters.sequence_length}"
            )

            all_sequences: pd.DataFrame = generate(
                generator, context_dataframe, self.par_parameters.sequence_length
            )

            list_of_sequences: list[pd.DataFrame] = [
                sequence
                for _, sequence in all_sequences.groupby(
                    list(self.par_parameters.item_id_columns)
                )
            ]
            logging.info(f"Number of sequences: {len(list_of_sequences)}")
            return list_of_sequences

        raise ValueError("Missing context dataframe")

    def save_generator(self, generator: PARSynthesizer, file_name: str) -> None:
        """Saves the generator to disk."""
        generator.save(file_name)

    @staticmethod
    def load_generator(generator_parameters: ParParameters) -> ParGenerator:
        """Loads the generator from disk."""
        par_synthesizer: PARSynthesizer = PARSynthesizer.load(
            generator_parameters.filename
        )
        logging.info(f"PAR implementation loaded from {generator_parameters.filename}")

        return ParGenerator(
            provider=Provider.SDV.value,
            generator_parameters=generator_parameters,
            generator=par_synthesizer,
        )

    @property
    def transformer(self) -> GeneratorTransformer:
        """Returns the transformer instance."""
        return self._transformer


def train_par(
    par_parameters: ParParameters, training_data: pd.DataFrame
) -> PARSynthesizer:
    """Trains and returns a PAR generator from SDV."""
    metadata: SingleTableMetadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=training_data)

    # TODO: Add support for datetime format.
    # TODO: There are plenty more options. Including epochs. And context columns!
    metadata.update_column(
        column_name=par_parameters.timestamp_column, sdtype="datetime"
    )

    metadata.update_column(column_name=SURROGATE_ITEM_ID, sdtype="id")
    metadata.set_sequence_key(column_name=SURROGATE_ITEM_ID)
    metadata.set_sequence_index(column_name=par_parameters.timestamp_column)

    synthesizer: PARSynthesizer = PARSynthesizer(
        metadata=metadata,
        epochs=par_parameters.epochs,
        verbose=True,
        context_columns=list(par_parameters.item_id_columns),
    )

    synthesizer.fit(training_data)
    return synthesizer


def generate(
    generator: PARSynthesizer,
    context_dataframe: pd.DataFrame,
    sequence_length: Optional[int] = None,
) -> pd.DataFrame:
    """Generates synthetic time series using the PAR algorithm."""
    return generator.sample_sequential_columns(
        context_columns=context_dataframe, sequence_length=sequence_length
    )


def preprocess_data(
    item_id_columns: tuple[str], raw_data: pd.DataFrame
) -> pd.DataFrame:
    """Pre-process time series data according to SDV's requirements."""
    result: pd.DataFrame = raw_data.fillna(0)
    return add_surrogate_key(item_id_columns, result)
