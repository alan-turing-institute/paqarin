"""Test for the ydata_adapter module."""

from unittest.mock import MagicMock, patch

from ydata_synthetic.synthesizers import ModelParameters  # type: ignore
from ydata_synthetic.synthesizers.timeseries import (  # type: ignore
    TimeSeriesSynthesizer,
)

from paqarin import ydata_adapter


@patch("ydata_synthetic.synthesizers.timeseries.timegan.model.TimeGAN.sample")
def test_generate(mock_sample: MagicMock) -> None:
    """Test the generation of synthetic sequences."""
    mock_generator: TimeSeriesSynthesizer = TimeSeriesSynthesizer(
        modelname="timegan", model_parameters=ModelParameters()
    )

    number_of_sequences: int = 10
    ydata_adapter.generate(mock_generator, number_of_sequences)

    mock_sample.assert_called_with(number_of_sequences)


@patch("ydata_synthetic.synthesizers.timeseries.TimeSeriesSynthesizer.load")
def test_load(mock_load: MagicMock) -> None:
    """Tests loading a generator from disk."""
    filename: str = "model_file"
    ydata_adapter.load(filename)

    mock_load.assert_called_with(filename)


@patch("ydata_synthetic.synthesizers.timeseries.timegan.model.TimeGAN.save")
def test_save(mock_save: MagicMock) -> None:
    """Tests saving the generator to disk."""
    mock_generator: TimeSeriesSynthesizer = TimeSeriesSynthesizer(
        modelname="timegan", model_parameters=ModelParameters()
    )

    filename: str = "model_file"
    ydata_adapter.save(mock_generator, filename)

    mock_save.assert_called_with(filename)
