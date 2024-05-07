"""Tests for the timegan module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from paqarin import synthcity_adapter, ydata_adapter
from paqarin.generators import TimeGanGenerator, TimeGanParameters
from tests.test_synthcity_adapter import TEST_GENERATED_SEQUENCES


def get_data_for_transformer() -> pd.DataFrame:
    """Generates sample data for testing."""
    # Inspired by Scikit-Learn documentation:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    training_data: pd.DataFrame = pd.DataFrame(
        {"feature_1": [-1, -0.5, 0, 1], "feature_2": [2, 6, 10, 18]}
    )

    return training_data


def test_transformer() -> None:
    """Tests data preprocessing for TimeGAN."""
    sequence_length: int = 2
    transformer: ydata_adapter.TimeGanTransformer = ydata_adapter.TimeGanTransformer(
        sequence_length=sequence_length,
        numerical_columns=["feature_1", "feature_2"],
    )

    assert not transformer.is_fitted()

    training_data: pd.DataFrame = get_data_for_transformer()
    transformer.fit(training_data)
    assert transformer.is_fitted()

    original_sequence: np.ndarray = np.array([[-0.5, 6.0], [0.0, 10.0]])
    transformed_sequence: np.ndarray = np.array([[0.25, 0.25], [0.5, 0.5]])

    another_sequence: np.ndarray = np.array([[0, 10], [1, 18]])
    another_transformed: np.ndarray = np.array([[0.5, 0.5], [1.0, 1.0]])

    transformed_data: list[np.ndarray] = transformer.transform(training_data)
    assert any(
        np.array_equal(transformed_sequence, sequence) for sequence in transformed_data
    )
    assert any(
        np.array_equal(another_transformed, sequence) for sequence in transformed_data
    )

    recovered_sequences: list[pd.DataFrame] = transformer.inverse_transform(
        [transformed_sequence, another_transformed]
    )
    np.testing.assert_array_equal(original_sequence, recovered_sequences[0])
    np.testing.assert_array_equal(another_sequence, recovered_sequences[1])


@patch("paqarin.ydata_adapter.train_timegan")
def test_create_generator(mock_train_timegan: MagicMock) -> None:
    """Tests creating a TimeGAN generator instance."""
    provider: str = "ydata"
    sequence_length: int = 8
    numerical_columns: list[str] = ["column"]
    batch_size: int = 32
    learning_rate: float = 5e-4
    latent_dimension: int = 24
    gamma: int = 1
    epochs: int = 10
    number_of_sequences: int = 6

    generator_parameters: TimeGanParameters = TimeGanParameters(
        sequence_length=sequence_length,
        numerical_columns=numerical_columns,
        batch_size=batch_size,
        learning_rate=learning_rate,
        latent_dimension=latent_dimension,
        gamma=gamma,
        epochs=epochs,
        number_of_sequences=number_of_sequences,
    )

    ydata_no_transformer_generator: TimeGanGenerator = TimeGanGenerator(
        provider=provider,
        generator_parameters=generator_parameters,
    )

    transformer: ydata_adapter.TimeGanTransformer = (
        ydata_no_transformer_generator.transformer  # type: ignore
    )
    assert transformer is not None
    assert sequence_length == transformer.sequence_length
    assert numerical_columns == transformer.numerical_columns

    assert isinstance(
        ydata_no_transformer_generator.generator_adapter,
        ydata_adapter.TimeGanGeneratorAdapter,
    )

    training_data: pd.DataFrame = pd.DataFrame()
    ydata_no_transformer_generator.fit(
        training_data, date_format="%Y-%m-%d %H:%M:%S.%f"
    )
    mock_train_timegan.assert_called_once_with(generator_parameters, training_data)


@patch("pickle.dump")
def test_create_generator_with_fitted_transformer(mock_dump: MagicMock) -> None:
    """Tests creating a TimeGAN generator when providing a trained transformer."""
    provider: str = "ydata"
    sequence_length: int = 8
    numerical_columns: list[str] = ["column"]
    batch_size: int = 32
    learning_rate: float = 5e-4
    latent_dimension: int = 24
    gamma: int = 1
    epochs: int = 10
    number_of_sequences: int = 6
    file_name: str = "my_file"

    transformer: ydata_adapter.TimeGanTransformer = ydata_adapter.TimeGanTransformer(
        sequence_length=sequence_length,
        numerical_columns=numerical_columns,
    )

    generator_parameters: TimeGanParameters = TimeGanParameters(
        sequence_length=sequence_length,
        numerical_columns=numerical_columns,
        batch_size=batch_size,
        learning_rate=learning_rate,
        latent_dimension=latent_dimension,
        gamma=gamma,
        epochs=epochs,
        number_of_sequences=number_of_sequences,
        filename=file_name,
    )

    generator: TimeGanGenerator = TimeGanGenerator(
        provider=provider,
        generator_parameters=generator_parameters,
        transformer=transformer,
    )

    assert generator.transformer is transformer

    generator.save_transformer()
    mock_dump.assert_not_called()

    transformer.fit(get_data_for_transformer())
    generator.save_transformer()
    mock_dump.assert_called_once()

    with pytest.raises(ValueError):
        _ = TimeGanGenerator(
            provider="random_provider",
            generator_parameters=generator_parameters,
            transformer=transformer,
        )


@patch.object(synthcity_adapter.TimeGanGeneratorAdapter, "generate_sequences")
@patch("paqarin.generator.DummyTransformer.save")
def test_synthcity_generator(
    mock_dummy_save: MagicMock, mock_generate_sequences: MagicMock
) -> None:
    """Tests creating a TimeGAN generator using the Synthcity implementation."""
    item_id_column: str = "item_id"
    timestamp_column: str = "timestamp"
    numerical_columns: list[str] = ["numerical_column"]

    TIME_GAN_FILENAME: str = "time_gan_payments.pkl"

    provider: str = "synthcity"
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 5e-4
    latent_dimension: int = 24
    gamma: int = 1

    time_gan_parameters: TimeGanParameters = TimeGanParameters(
        batch_size=batch_size,
        learning_rate=learning_rate,
        latent_dimension=latent_dimension,
        gamma=gamma,
        epochs=epochs,
        item_id_column=item_id_column,
        timestamp_column=timestamp_column,
        numerical_columns=numerical_columns,
        filename=TIME_GAN_FILENAME,
    )

    generator: TimeGanGenerator = TimeGanGenerator(
        provider=provider,
        generator_parameters=time_gan_parameters,
    )

    assert isinstance(generator.transformer, synthcity_adapter.TimeGanTransformer)
    assert isinstance(
        generator.generator_adapter,
        synthcity_adapter.TimeSeriesGeneratorAdapter,
    )

    number_of_sequences: int = 2
    generated_sequences: list[pd.DataFrame] = TEST_GENERATED_SEQUENCES
    mock_generate_sequences.return_value = generated_sequences
    generator.generator_adapter = synthcity_adapter.TimeGanGeneratorAdapter(
        time_gan_parameters
    )

    synthetic_sequences: list = generator.generate(number_of_sequences)
    assert len(synthetic_sequences) == number_of_sequences

    generator.save_transformer()
    mock_dummy_save.assert_not_called()
