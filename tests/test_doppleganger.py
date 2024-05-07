"""Tests for the doppleganger module."""

from unittest.mock import MagicMock, patch

import pandas as pd

from paqarin import ydata_adapter
from paqarin.generators import DoppleGangerGenerator, DoppleGanGerParameters
from paqarin.utils import normalise_sequences

NUMERICAL_COLUMNS: list[str] = ["numerical_column"]


def create_test_transformer() -> (
    tuple[ydata_adapter.DoppleGanGerTransformer, pd.DataFrame]
):
    """Creates a sample transformer for testing."""
    transformer: ydata_adapter.DoppleGanGerTransformer = (
        ydata_adapter.DoppleGanGerTransformer(
            numerical_columns=["numerical_column", "another_numerical_column"]
        )
    )
    training_data: pd.DataFrame = pd.DataFrame(
        {
            "categorical_column": ["a", "b", "c", "d"],
            "numerical_column": [-1.0, -0.5, 0.0, 1.0],
            "another_numerical_column": [2.0, 6.0, 10.0, 18.0],
        }
    )

    transformer.fit(training_data)

    return transformer, training_data


def test_transformer() -> None:
    """Tests the transformer logic for DoppleGANger."""
    transformer, training_data = create_test_transformer()

    assert transformer.is_fitted()

    scaled_data: pd.DataFrame = transformer.transform(training_data)
    pd.testing.assert_frame_equal(
        scaled_data,
        pd.DataFrame(
            {
                "categorical_column": ["a", "b", "c", "d"],
                "numerical_column": [0.0, 0.25, 0.5, 1.0],
                "another_numerical_column": [0.0, 0.25, 0.5, 1.0],
            }
        ),
    )

    pd.testing.assert_frame_equal(
        training_data, transformer.inverse_transform([scaled_data])[0]
    )


def get_test_parameters() -> DoppleGanGerParameters:
    """Generates sample DoppleGANger parameters for testing."""
    sequence_length: int = 2
    DOPPLEGANGER_FOLDER: str = "doppleganger_payments"
    epochs: int = 400

    categorical_columns: list[str] = ["categorical_column"]
    measurement_columns: list[str] = NUMERICAL_COLUMNS

    batch_size: int = 100
    learning_rate: float = 0.001
    latent_dimension: int = 100
    wgan_weight: float = 2
    packing_degree: int = 1
    sample_length: int = 6
    steps_per_batch: int = 1

    exponential_decay_rates: tuple[float, float] = (0.2, 0.9)
    doppleganger_parameters: DoppleGanGerParameters = DoppleGanGerParameters(
        batch_size=batch_size,
        learning_rate=learning_rate,
        latent_dimension=latent_dimension,
        exponential_decay_rates=exponential_decay_rates,
        wgan_weight=wgan_weight,
        packing_degree=packing_degree,
        epochs=epochs,
        sequence_length=sequence_length,
        sample_length=sample_length,
        steps_per_batch=steps_per_batch,
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=categorical_columns,
        measurement_columns=measurement_columns,
        filename=DOPPLEGANGER_FOLDER,
    )

    return doppleganger_parameters


@patch("paqarin.ydata_adapter.generate")
def test_generator(mock_generate: MagicMock) -> None:
    """Tests a generator for the DoppleGANger algorithm."""
    provider: str = "ydata"

    doppleganger_parameters: DoppleGanGerParameters = get_test_parameters()
    ydata_generator: DoppleGangerGenerator = DoppleGangerGenerator(
        provider, doppleganger_parameters
    )

    assert ydata_generator.transformer is not None
    assert isinstance(
        ydata_generator.transformer, ydata_adapter.DoppleGanGerTransformer
    )
    assert ydata_generator.transformer.numerical_columns == NUMERICAL_COLUMNS

    assert hasattr(ydata_generator, "generator_adapter")

    trained_transformer, training_data = create_test_transformer()
    generator_with_transformer: DoppleGangerGenerator = DoppleGangerGenerator(
        provider, doppleganger_parameters, transformer=trained_transformer
    )
    assert generator_with_transformer.transformer is trained_transformer

    mock_generate.return_value = [
        pd.DataFrame(
            {
                "categorical_column": ["a", "b", "c", "d"],
                "numerical_column": [0.0, 0.25, 0.5, 1.0],
                "another_numerical_column": [0.0, 0.25, 0.5, 1.0],
            }
        )
    ]

    sequences_without_date: list = generator_with_transformer.generate(1)
    assert len(sequences_without_date) == 1
    pd.testing.assert_frame_equal(training_data, sequences_without_date[0])

    index_name: str = "timestamp"
    index_values: list = [
        pd.to_datetime(day, format="%d/%m/%Y")
        for day in ["1/1/2024", "2/1/2024", "3/1/2024", "4/1/2024"]
    ]
    date_index: pd.DatetimeIndex = pd.DatetimeIndex(index_values, name=index_name)
    sequences_with_date: list = generator_with_transformer.generate(
        1, date_index=date_index
    )

    expected_sequence: pd.DataFrame = training_data.copy()
    expected_sequence[index_name] = index_values
    pd.testing.assert_frame_equal(sequences_with_date[0], expected_sequence)


def test_normalise_sequences() -> None:
    """Tests normalising sequences, for them all to have the same size."""
    item_id_column: str = "item_id"
    timestamp_column: str = "timestamp"
    numeric_column: str = "numeric_column"
    frequency: str = "1D"
    date_format: str = "%Y-%m-%d %H:%M:%S.%f"

    raw_data: pd.DataFrame = pd.DataFrame(
        {
            item_id_column: ["item_1", "item_1", "item_2", "item_2"],
            numeric_column: [10.0, 20.0, 30.0, 40.0],
            timestamp_column: [
                "2018-10-26 12:00:00.0000000011",
                "2018-10-26 13:00:00.0000000011",
                "2018-10-27 12:00:00.0000000011",
                "2018-10-27 13:00:00.0000000011",
            ],
        }
    )

    pre_processed_data, number_of_samples, sequence_length = normalise_sequences(
        item_id_column,
        timestamp_column,
        frequency,
        raw_data,
        inclusive_range="both",
        date_format=date_format,
    )

    assert 2 == number_of_samples
    assert 2 == sequence_length

    pd.testing.assert_frame_equal(
        pre_processed_data,
        pd.DataFrame(
            {
                timestamp_column: [
                    pd.to_datetime(date, format="%Y-%m-%d")
                    for date in [
                        "2018-10-26",
                        "2018-10-27",
                        "2018-10-26",
                        "2018-10-27",
                    ]
                ],
                numeric_column: [30.0, 0.0, 0.0, 70.0],
                item_id_column: ["item_1", "item_1", "item_2", "item_2"],
            }
        ),
    )
