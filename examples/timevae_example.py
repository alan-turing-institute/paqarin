"""Example of synthetic data generation using DoppleGANger."""

from paqarin.generators import TimeVaeGenerator, TimeVaeParameters
from paqarin.evaluation import EvaluationPipeline
from paqarin import multivariate_metrics
import pandas as pd

print("Defining Generator")
timevae_generator = TimeVaeGenerator(
    provider="synthcity",
    generator_parameters=TimeVaeParameters(
#        categorical_columns=["isp", "technology", "state"],
#        measurement_columns=["traffic_byte_counter", "ping_loss_rate"],
#        packing_degree=1,
#        sample_length=8,
#        steps_per_batch=1,
#        wgan_weight=2,
        item_id_column = "state",
#        layers_dimension: Optional[int] = None,
#        noise_dimension: Optional[int] = None,
#        number_of_sequences: Optional[int] = None,
#        timestamp_column: Optional[str] = None,
#
#
        batch_size=512,
        epochs=100,
        filename="timevae_generator",
        latent_dimension=20,
        learning_rate=0.001,
        numerical_columns=["traffic_byte_counter", "ping_loss_rate"],
        sequence_length=56,
#
        gamma = 1

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
