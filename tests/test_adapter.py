"""Tests for the adapter module."""

from paqarin import synthcity_adapter, ydata_adapter
from paqarin.adapter import Method, Provider, get_generator_adapter
from paqarin.generators import ParParameters, TimeGanParameters
from tests import test_doppleganger


def test_get_generator_adapter() -> None:
    """Tests getting the right instance for a method an implementation."""
    time_gan_parameters: TimeGanParameters = TimeGanParameters(
        item_id_column="item_id",
        timestamp_column="time",
        numerical_columns=["numerical_column"],
        sequence_length=1,
    )
    assert isinstance(
        get_generator_adapter(
            Provider.YDATA,
            Method.TIME_GAN,
            time_gan_parameters,
        ),
        ydata_adapter.TimeGanGeneratorAdapter,
    )

    assert isinstance(
        get_generator_adapter(
            Provider.YDATA,
            Method.DOPPLEGANGER,
            test_doppleganger.get_test_parameters(),
        ),
        ydata_adapter.DoppleGanGerGeneratorAdapter,
    )

    assert isinstance(
        get_generator_adapter(
            Provider.SYNTHCITY,
            Method.TIME_GAN,
            time_gan_parameters,
        ),
        synthcity_adapter.TimeGanGeneratorAdapter,
    )

    par_parameters: ParParameters = ParParameters(
        filename="file.pkl",
        item_id_columns=("item_id",),
        timestamp_column="timestamp",
        epochs=1,
        sequence_length=2,
    )

    assert isinstance(
        get_generator_adapter(
            Provider.SDV,
            Method.PAR,
            par_parameters,
        ),
        synthcity_adapter.TimeSeriesGeneratorAdapter,
    )
