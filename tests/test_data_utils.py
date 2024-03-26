"""Tests for the data_utils module."""
import pandas as pd

from paqarin.utils import add_surrogate_key


def test_add_surrogate_key() -> None:
    """Tests adding a surrogate key to a time series dataframe."""
    item_id_columns: tuple = ("item_id_1", "item_id_2")
    raw_data: pd.DataFrame = pd.DataFrame(
        {
            item_id_columns[0]: ["a", "a", "a", "b", "b", "b"],
            item_id_columns[1]: [0, 1, 0, 1, 0, 1],
        },
        index=[0, 1, 2, 3, 4, 5],
    )

    data_with_surrogate: pd.DataFrame = add_surrogate_key(item_id_columns, raw_data)

    pd.testing.assert_frame_equal(
        data_with_surrogate,
        pd.DataFrame(
            {
                item_id_columns[0]: ["a", "a", "a", "b", "b", "b"],
                item_id_columns[1]: [0, 1, 0, 1, 0, 1],
                "surrogate_item_id": [0, 1, 0, 2, 3, 2],
            },
            index=[0, 1, 2, 3, 4, 5],
        ),
    )
