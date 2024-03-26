"""Example of synthetic data generation using DoppleGANger."""

from paqarin.generators import DoppleGangerGenerator, DoppleGanGerParameters
from paqarin.evaluation import EvaluationPipeline
from paqarin import multivariate_metrics
import pandas as pd

print("Defining Generator")
doppleganger_generator: DoppleGangerGenerator = DoppleGangerGenerator(
    provider="ydata",
    generator_parameters=DoppleGanGerParameters(
        batch_size=512,
        learning_rate=0.001,
        latent_dimension=20,
        exponential_decay_rates=(0.2, 0.9),
        wgan_weight=2,
        packing_degree=1,
        epochs=100,
        sequence_length=56,
        sample_length=8,
        steps_per_batch=1,
        numerical_columns=["traffic_byte_counter", "ping_loss_rate"],
        measurement_columns=["traffic_byte_counter", "ping_loss_rate"],
        categorical_columns=["isp", "technology", "state"],
        filename="doppleganger_generator",
    )
)

print("Defining Pipeline")
evaluation_pipeline: EvaluationPipeline = EvaluationPipeline(
    generator_map={"doppleganger": doppleganger_generator},
    scoring=multivariate_metrics.PredictiveScorer(
        lstm_units=12,
        iterations=3,
        scorer_epochs=100,
        scorer_batch_size=128,
        number_of_features=2,
        numerical_columns=["traffic_byte_counter", "ping_loss_rate"],
        sequence_length=56,
        metric_value_key="mean_absolute_error")
)

print("Starting Training and Evaluation")
evaluation_pipeline.fit(pd.read_csv("fcc_mba.csv"))

print("Evaluation results")
if evaluation_pipeline.scoring_object is not None:
    print(evaluation_pipeline.scoring_object.summary_metrics)

print("Data Generation")
synthetic_data_blocks: list[pd.DataFrame] = doppleganger_generator.generate(
    number_of_sequences=10
)

print(synthetic_data_blocks[:2])