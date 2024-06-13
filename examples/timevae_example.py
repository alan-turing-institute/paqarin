"""Example of synthetic data generation using DoppleGANger."""

from paqarin.generators import TimeVaeGenerator, TimeVaeParameters
from paqarin.evaluation import EvaluationPipeline
from paqarin import multivariate_metrics
import pandas as pd

print("Defining Generator")

try:
    df = pd.read_csv("examples/stock_data.csv")
except:
    df = pd.read_csv("stock_data.csv")
df1 = df.copy()
df1 += 1
t1 = pd.to_datetime("1/1/2000")+ pd.Timedelta("1 day")*df.index 
df['time'] = t1.strftime('%d/%m/%Y')
df1['time'] = t1.strftime('%d/%m/%Y')
df['id'] = 1 
df1['id'] = 2 

df = pd.concat([df,df1])


numerical_columns = [x for x in df.columns if x not in ['time','id']]
timevae_generator = TimeVaeGenerator(
    provider="synthcity",
    generator_parameters=TimeVaeParameters(
##        item_id_column, timestamp_column, numerical_columns
#        categorical_columns=["isp", "technology", "state"],
#        measurement_columns=["traffic_byte_counter", "ping_loss_rate"],
#        packing_degree=1,
#        sample_length=8,
#        steps_per_batch=1,
#        wgan_weight=2,
        item_id_column = "id",
        timestamp_column = "time",
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
        numerical_columns=numerical_columns,
        sequence_length=56,
#
        gamma = 1

    )
)

print("Defining Pipeline")
evaluation_pipeline: EvaluationPipeline = EvaluationPipeline(
    generator_map={"timevae": timevae_generator},
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
evaluation_pipeline.fit(df)

print("Evaluation results")
if evaluation_pipeline.scoring_object is not None:
    print(evaluation_pipeline.scoring_object.summary_metrics)

print("Data Generation")
synthetic_data_blocks: list[pd.DataFrame] = timevae_generator.generate(
    number_of_sequences=10
)

print(synthetic_data_blocks[:2])
