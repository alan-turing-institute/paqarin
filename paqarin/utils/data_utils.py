"""Data processing functionality that's common to many data generation algorithms."""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

from paqarin.generator import SURROGATE_ITEM_ID


def add_surrogate_key(
    item_id_columns: tuple[str], raw_data: pd.DataFrame
) -> pd.DataFrame:
    """Adds a surrogate key, based on the provided item_id columns."""
    result: pd.DataFrame = raw_data.copy()
    all_entities: pd.DataFrame = result[list(item_id_columns)].drop_duplicates()
    all_entities = all_entities.reset_index(drop=True)

    item_to_index: dict[Any, Any] = {
        tuple(row.values): index for index, row in all_entities.iterrows()
    }

    result[SURROGATE_ITEM_ID] = result.apply(
        lambda row: item_to_index[tuple(row[list(item_id_columns)].values)], axis=1
    )

    return result


def normalise_sequences(
    item_id_column: str,
    timestamp_column: str,
    frequency: str,
    raw_data: pd.DataFrame,
    date_format: str = "%d/%m/%Y",
    inclusive_range: Literal["both", "neither", "left", "right"] = "both",
) -> tuple[pd.DataFrame, int, int]:
    """Makes all sequences in the time series dataset to have the same length."""
    items_ids: np.ndarray = raw_data[item_id_column].unique()

    start_date = pd.to_datetime(
        raw_data[timestamp_column].min(), format=date_format
    ).floor("D")
    end_date = pd.to_datetime(
        raw_data[timestamp_column].max(), format=date_format
    ).floor("D")
    new_index: pd.DatetimeIndex = pd.DatetimeIndex(
        pd.date_range(
            start=start_date, end=end_date, freq=frequency, inclusive=inclusive_range
        ),
        name=timestamp_column,
    )

    logging.info(f"Start date {new_index.min()}, End data: {new_index.max()}")

    sample_dataframes: list[pd.DataFrame] = []
    for item_id in items_ids:
        sample_rows: pd.DataFrame = raw_data.query(
            f"{item_id_column} == '{item_id}'"
        ).copy()

        sample_rows[timestamp_column] = pd.to_datetime(
            sample_rows[timestamp_column], format=date_format
        )
        sample_rows = sample_rows.set_index(timestamp_column)

        sample_rows = (
            sample_rows.resample(frequency)
            .sum(numeric_only=True)
            .reindex(new_index)
            .fillna(0)
        )
        sample_rows[item_id_column] = item_id

        sample_dataframes.append(sample_rows.reset_index())

    return (
        pd.concat(sample_dataframes, ignore_index=True),
        len(items_ids),
        len(new_index),
    )
