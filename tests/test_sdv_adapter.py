"""Tests for the sdv_adapter module."""

from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
from sdv.sequential import PARSynthesizer  # type: ignore

from paqarin import sdv_adapter
from paqarin.adapter import Provider
from paqarin.generator import DummyTransformer
from paqarin.generators import ParParameters

ITEM_ID_COLUMNS: tuple[str] = ("item_id",)
TIMESTAMP_COLUMN: str = "timestamp"
EPOCHS: int = 128
SEQUENCE_LENGTH: Optional[int] = 180


def get_test_parameters() -> ParParameters:
    """Generates sample parameter values for testing."""
    PAR_FILENAME: str = "par_payments"

    return ParParameters(
        filename=PAR_FILENAME,
        item_id_columns=ITEM_ID_COLUMNS,
        timestamp_column=TIMESTAMP_COLUMN,
        epochs=EPOCHS,
        sequence_length=SEQUENCE_LENGTH,
    )


@patch("sdv.sequential.par.PARSynthesizer.fit")
@patch("sdv.sequential.par.PARSynthesizer.sample_sequential_columns")
def test_train_par_and_generate(
    mock_sample_sequential: MagicMock, mock_fit: MagicMock
) -> None:
    """Tests training a PAR model and using it for generating data."""
    par_parameters: ParParameters = get_test_parameters()
    surrogate_item_id_column: str = "surrogate_item_id"
    training_data: pd.DataFrame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: [datetime.now()],
            ITEM_ID_COLUMNS[0]: "item_1",
            surrogate_item_id_column: ["1"],
        }
    )

    synthesizer: PARSynthesizer = sdv_adapter.train_par(
        par_parameters=par_parameters, training_data=training_data
    )

    metadata: dict = synthesizer.get_metadata().to_dict()
    column_type_key: str = "sdtype"
    metadata_columns: dict = metadata["columns"]

    assert "datetime" == metadata_columns[TIMESTAMP_COLUMN][column_type_key]
    assert "id" == metadata_columns[surrogate_item_id_column][column_type_key]
    assert TIMESTAMP_COLUMN == metadata["sequence_index"]
    assert surrogate_item_id_column == metadata["sequence_key"]

    parameters: dict = synthesizer.get_parameters()

    assert EPOCHS == parameters["epochs"]
    assert list(ITEM_ID_COLUMNS) == parameters["context_columns"]
    mock_fit.assert_called_once_with(training_data)

    synthetic_sequence: pd.DataFrame = pd.DataFrame()
    mock_sample_sequential.return_value = synthetic_sequence

    generated_sequence: pd.DataFrame = sdv_adapter.generate(
        generator=synthesizer,
        context_dataframe=pd.DataFrame({"item_id": ["item_1"]}),
        sequence_length=SEQUENCE_LENGTH,
    )

    assert synthetic_sequence is generated_sequence


def test_save_model() -> None:
    """Tests saving the generator to disk."""
    adapter: sdv_adapter.ParGeneratorAdapter = sdv_adapter.ParGeneratorAdapter(
        par_parameters=get_test_parameters(), transformer=DummyTransformer()
    )
    file_name: str = "a_file.pkl"

    mock_generator: MagicMock = MagicMock()

    adapter.save_generator(mock_generator, file_name)
    mock_generator.save.assert_called_once_with(file_name)


@patch("paqarin.sdv_adapter.PARSynthesizer")
def test_load_generator(synthesizer_mock: MagicMock) -> None:
    """Tests loading a PAR generator from disk."""
    par_parameters: ParParameters = get_test_parameters()
    synthesizer_dummy: dict = {}
    synthesizer_mock.load.return_value = synthesizer_dummy

    generator = sdv_adapter.ParGeneratorAdapter.load_generator(par_parameters)

    synthesizer_mock.load.assert_called_once_with(par_parameters.filename)
    assert generator.provider == Provider.SDV.value
    assert generator.generator is synthesizer_dummy
