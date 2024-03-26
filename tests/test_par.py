"""Tests for the PAR module."""
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from paqarin import sdv_adapter
from paqarin.generator import DummyTransformer
from paqarin.generators import ParGenerator, ParParameters


@patch("paqarin.sdv_adapter.generate")
def test_create_generator(mock_generate: MagicMock) -> None:
    """Tests creating a PAR generator."""
    par_filename: str = "par_payments"
    provider: str = "sdv"

    epochs: int = 128

    sequence_length: Optional[int] = 180
    timestamp_column: str = "timestamp"
    item_id_columns: tuple[str] = ("item_id",)

    generator_parameters: ParParameters = ParParameters(
        filename=par_filename,
        item_id_columns=item_id_columns,
        timestamp_column=timestamp_column,
        epochs=epochs,
        sequence_length=sequence_length,
    )

    generator: ParGenerator = ParGenerator(
        provider=provider,
        generator_parameters=generator_parameters,
    )

    assert isinstance(generator.transformer, DummyTransformer)
    assert hasattr(generator, "generator_adapter")

    with pytest.raises(ValueError, match="Missing context dataframe"):
        generator.generate(number_of_sequences=1)

    first_entity: pd.DataFrame = pd.DataFrame(
        {
            item_id_columns[0]: ["item_1", "item_1"],
            timestamp_column: ["01-01-2024", "01-02-2024"],
        }
    )
    second_entity: pd.DataFrame = pd.DataFrame(
        {
            item_id_columns[0]: ["item_2", "item_2"],
            timestamp_column: ["01-01-2024", "01-02-2024"],
        }
    )

    mock_generate.return_value = pd.concat([first_entity, second_entity])
    generated_sequences: list[pd.DataFrame] = generator.generate(context=pd.DataFrame())
    assert 2 == len(generated_sequences)
    assert_frame_equal(generated_sequences[0], first_entity)
    assert_frame_equal(generated_sequences[1], second_entity)

    with pytest.raises(ValueError, match=" is not a valid Provider"):
        _: ParGenerator = ParGenerator(
            provider="random",
            generator_parameters=generator_parameters,
        )


def test_preprocess_data() -> None:
    """Tests pre-processing data for the SDV implementation."""
    # TODO: Move this test to the test_sdv_adapter module.
    item_id_columns: tuple[str] = ("item_id",)
    timestamp_column: str = "timestamp"

    raw_data: pd.DataFrame = pd.DataFrame(
        {
            item_id_columns[0]: ["item_1", "item_1", "item_2", "item_2"],
            timestamp_column: ["01-01-2024", "01-02-2024", "01-01-2024", "01-02-2024"],
            "transact_num_domestic": [None, 1, 2, 3],
        }
    )

    preprocessed_data: pd.DataFrame = sdv_adapter.preprocess_data(
        item_id_columns, raw_data
    )

    assert_frame_equal(
        preprocessed_data,
        pd.DataFrame(
            {
                item_id_columns[0]: ["item_1", "item_1", "item_2", "item_2"],
                timestamp_column: [
                    "01-01-2024",
                    "01-02-2024",
                    "01-01-2024",
                    "01-02-2024",
                ],
                "transact_num_domestic": [0.0, 1.0, 2.0, 3.0],
                "surrogate_item_id": [0, 0, 1, 1],
            }
        ),
    )
