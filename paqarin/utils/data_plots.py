"""Module for plotting multivariate time series and its synthetic data generation."""
import itertools
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type:ignore

from paqarin.evaluation import MetricManager

MARKER: str = "*"
LINE_STYLE: str = "dashed"


def plot_multivariate_time_series(
    time_series_dataframe: pd.DataFrame,
    figure_size: tuple[int, int],
    layout: tuple[int, int],
    font_size: int,
) -> None:
    """Plots a multivariate time series using subplots."""
    sns.set(font_scale=font_size)

    time_series_dataframe.plot(
        subplots=True,
        figsize=figure_size,
        marker=MARKER,
        linestyle=LINE_STYLE,
        layout=layout,
    )

    sns.set()


def plot_and_compare(
    real_sequence: np.ndarray, synthetic_sequence: np.ndarray, columns: List[str]
) -> None:
    """Plots a real multivariate sequence against a synthetic one, one plot per column."""
    for column_index, column_name in enumerate(columns):
        plot_column_comparison(
            real_sequence=real_sequence[:, column_index],
            synthetic_sequence=synthetic_sequence[:, column_index],
            column_name=column_name,
            figure_size=(10, 6),
        )


def plot_column_comparison(
    real_sequence: np.ndarray,
    synthetic_sequence: np.ndarray,
    column_name: str,
    figure_size: tuple[int, int],
):
    """Plots a real univariate sequence next to another."""
    data_frame: pd.DataFrame = pd.DataFrame(
        {"Real": real_sequence, "Synthetic": synthetic_sequence}
    )

    data_frame.plot(
        title=column_name, marker=MARKER, linestyle=LINE_STYLE, figsize=figure_size
    )


def plot_predictions(
    labels: np.ndarray,
    predictions: list[tuple[str, np.ndarray]],
    column_name: str,
    figure_size: tuple[int, int],
):
    """Plots labels against predictions for time series."""
    plt.figure(figsize=figure_size)
    plt.plot(labels, marker=MARKER, linestyle=LINE_STYLE, label="Label")

    colors = itertools.cycle(["r", "g", "b"])
    for prediction_label, data in predictions:
        plt.scatter(
            range(labels.shape[0]),
            data,
            marker="o",
            color=next(colors),
            label=prediction_label,
        )  # type: ignore
    plt.ylabel(column_name)
    plt.legend()
    plt.show()


def plot_metrics(metric_manager: MetricManager, metric: str):
    """Simple box plot with values."""
    data_for_plot: pd.DataFrame = pd.DataFrame(
        {
            generator_name: metric_manager.get_all_values(generator_name, metric)
            for generator_name in metric_manager.metrics_per_generator.keys()
        }
    )

    plt.boxplot(data_for_plot)
    plt.xticks(range(1, len(data_for_plot.columns) + 1), data_for_plot.columns)
