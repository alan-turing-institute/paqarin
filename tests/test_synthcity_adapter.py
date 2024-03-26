"""Tests for the synthcity_adapter module."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from synthcity.plugins.time_series.plugin_timegan import TimeGANPlugin  # type: ignore

from paqarin import synthcity_adapter
from paqarin.generators import TimeGanGenerator, TimeGanParameters

ITEM_ID_COLUMN: str = "id"
TIMESTAMP_COLUMN: str = "time_point"
NUMERICAL_COLUMN: str = "temp_b"
TIMESTAMPS: list[str] = ["01/02/2024", "02/02/2024", "03/02/2024", "04/02/2024"]

TEST_GENERATED_SEQUENCES: list[pd.DataFrame] = [
    pd.DataFrame(
        {
            "seq_id": [0, 0],
            "seq_time_id": ["01/02/2024", "02/02/2024"],
            "seq_static_id": ["id_1", "id_1"],
            "seq_temporal_temp_b": [0.0, 0.1],
            "seq_out_outcome": [True, True],
        }
    ),
    pd.DataFrame(
        {
            "seq_id": [1, 1],
            "seq_time_id": ["01/02/2024", "02/02/2024"],
            "seq_static_id": ["id_2", "id_2"],
            "seq_temporal_temp_b": [0.2, 0.3],
            "seq_out_outcome": [False, False],
        }
    ),
]


def get_test_training_data() -> pd.DataFrame:
    """Get sample training data for testing."""
    raw_data: pd.DataFrame = pd.DataFrame(
        {
            ITEM_ID_COLUMN: [
                "B7C3B9",
                "B7C3B9",
                "B7C3B9",
                "B7C3B9",
                "C02981",
                "C02981",
                "C02981",
                "C02981",
            ],
            TIMESTAMP_COLUMN: TIMESTAMPS * 2,
            NUMERICAL_COLUMN: [
                4.520580,
                3.345429,
                4.223980,
                None,
                4.774060,
                5.311364,
                4.360277,
                0.0,
            ],
        }
    )

    return raw_data


def test_timegan_transformer() -> None:
    """Tests preprocessing data for synthcity's TimeGAN implementation."""
    timegan_transformer: synthcity_adapter.TimeGanTransformer = (
        synthcity_adapter.TimeGanTransformer(
            item_id_column=ITEM_ID_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN,
            numerical_columns=[NUMERICAL_COLUMN],
        )
    )

    assert timegan_transformer.is_fitted()

    all_sequences: list[pd.DataFrame] = TEST_GENERATED_SEQUENCES

    result: list[pd.DataFrame] = timegan_transformer.inverse_transform(all_sequences)

    pd.testing.assert_frame_equal(
        result[0],
        pd.DataFrame(
            {
                TIMESTAMP_COLUMN: ["01/02/2024", "02/02/2024"],
                ITEM_ID_COLUMN: ["id_1", "id_1"],
                NUMERICAL_COLUMN: [0.0, 0.1],
            }
        ),
    )

    pd.testing.assert_frame_equal(
        result[1],
        pd.DataFrame(
            {
                TIMESTAMP_COLUMN: ["01/02/2024", "02/02/2024"],
                ITEM_ID_COLUMN: ["id_2", "id_2"],
                NUMERICAL_COLUMN: [0.2, 0.3],
            }
        ),
    )


@patch("synthcity.plugins.time_series.plugin_timegan.TimeGANPlugin.generate")
def test_generator_adapter(mock_plugin_generate: MagicMock) -> None:
    """Tests the generator adapter functionality for Synthcity's TimeGAN."""
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 0.1
    gamma: int = 1
    latent_dimension: int = 20

    generator_parameters: TimeGanParameters = TimeGanParameters(
        item_id_column=ITEM_ID_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        numerical_columns=[NUMERICAL_COLUMN],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        latent_dimension=latent_dimension,
    )
    generator_adapter: synthcity_adapter.TimeGanGeneratorAdapter = (
        synthcity_adapter.TimeGanGeneratorAdapter(generator_parameters)
    )
    assert type(generator_adapter.transformer) is synthcity_adapter.TimeGanTransformer

    training_data: pd.DataFrame = get_test_training_data()

    wrong_format: str = "%Y-%m-%d"
    right_format: str = "%d/%m/%Y"

    with pytest.raises(ValueError):
        trained_plugin: TimeGANPlugin = generator_adapter.train_generator(
            generator_parameters, training_data, date_format=wrong_format
        )

    trained_plugin = generator_adapter.train_generator(
        generator_parameters, training_data, date_format=right_format
    )
    assert isinstance(trained_plugin, TimeGANPlugin)

    assert epochs == trained_plugin.n_iter
    assert batch_size == trained_plugin.batch_size

    assert learning_rate == trained_plugin.generator_lr
    assert learning_rate == trained_plugin.discriminator_lr

    assert latent_dimension == trained_plugin.generator_n_units_hidden
    assert latent_dimension == trained_plugin.discriminator_n_units_hidden

    assert gamma == trained_plugin.gamma_penalty

    generate_return_value: MagicMock = mock_plugin_generate.return_value
    generate_return_value.dataframe.return_value = pd.DataFrame(
        {
            "seq_id": [0, 0, 1, 1],
            "seq_time_id": [8, 8, 8, 8],  # I saw this values in an example.
            "seq_static_id": [
                "ITEM_1",
                "ITEM_1",
                "ITEM_2",
                "ITEM_2",
            ],
            "seq_temporal_variable": [1, 2, 3, 4],
        },
        index=pd.Index(
            [
                "01-01-2024",
                "02-01-2024",
                "01-01-2024",
                "02-01-2024",
            ],
            name=TIMESTAMP_COLUMN,
        ),
    )

    number_of_sequences: int = 3
    generated_sequences: list = generator_adapter.generate_sequences(
        trained_plugin, number_of_sequences
    )

    mock_plugin_generate.assert_called_once_with(count=number_of_sequences)
    assert len(generated_sequences) == 2

    pd.testing.assert_frame_equal(
        generated_sequences[0],
        pd.DataFrame(
            {
                "seq_id": [0, 0],
                "seq_time_id": [8, 8],
                "seq_static_id": ["ITEM_1", "ITEM_1"],
                "seq_temporal_variable": [1, 2],
            },
            index=pd.Index(
                ["01-01-2024", "02-01-2024"],
                name=TIMESTAMP_COLUMN,
            ),
        ),
    )

    pd.testing.assert_frame_equal(
        generated_sequences[1],
        pd.DataFrame(
            {
                "seq_id": [1, 1],
                "seq_time_id": [8, 8],
                "seq_static_id": ["ITEM_2", "ITEM_2"],
                "seq_temporal_variable": [3, 4],
            },
            index=pd.Index(
                ["01-01-2024", "02-01-2024"],
                name=TIMESTAMP_COLUMN,
            ),
        ),
    )


@patch("paqarin.synthcity_adapter.load_from_file")
def test_load_generator_from_file(load_from_file: MagicMock) -> None:
    """Tests loading a Synthcity generator from disk."""
    dummy_timegan_plugin: dict = {}
    load_from_file.return_value = dummy_timegan_plugin

    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 0.1
    gamma: int = 1
    latent_dimension: int = 20
    filename: str = "model.file"

    generator_parameters: TimeGanParameters = TimeGanParameters(
        item_id_column=ITEM_ID_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        numerical_columns=[NUMERICAL_COLUMN],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        latent_dimension=latent_dimension,
        filename=filename,
    )

    generator_from_file: TimeGanGenerator = (
        synthcity_adapter.TimeGanGeneratorAdapter.load_generator(generator_parameters)
    )

    load_from_file.assert_called_once_with(filename)
    assert generator_from_file.generator is dummy_timegan_plugin


def test_preprocess_data() -> None:
    """Tests the data pre-processing logic."""
    raw_data: pd.DataFrame = get_test_training_data()

    (
        temporal_data,
        observation_data,
        static_data,
        outcome,
    ) = synthcity_adapter.preprocess_data(
        ITEM_ID_COLUMN, TIMESTAMP_COLUMN, [NUMERICAL_COLUMN], raw_data
    )

    assert 2 == len(temporal_data)
    pd.testing.assert_frame_equal(
        temporal_data[0],
        pd.DataFrame(
            {NUMERICAL_COLUMN: [4.520580, 3.345429, 4.223980, 0.0]},
            index=pd.Index(TIMESTAMPS, name=TIMESTAMP_COLUMN),
        ),
    )
    pd.testing.assert_frame_equal(
        temporal_data[1],
        pd.DataFrame(
            {NUMERICAL_COLUMN: [4.774060, 5.311364, 4.360277, 0.0]},
            index=pd.Index(TIMESTAMPS, name=TIMESTAMP_COLUMN),
        ),
    )

    assert 2 == len(observation_data)
    assert TIMESTAMPS == observation_data[0]
    assert TIMESTAMPS == observation_data[1]

    pd.testing.assert_frame_equal(
        static_data,
        pd.DataFrame(
            {ITEM_ID_COLUMN: ["B7C3B9", "C02981"]},
            index=[0, 1],
        ),
    )

    pd.testing.assert_frame_equal(
        outcome,
        pd.DataFrame(
            {"outcome": [0, 0]},
            index=[0, 1],
        ),
    )
