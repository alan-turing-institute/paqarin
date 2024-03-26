"""Trains a synthetic time series generator, using artifacts gathered from GCP."""

import argparse
import glob
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import dask.dataframe as dd

# Setting up Cloud Logging.
import google.cloud.logging  # type: ignore
import pandas as pd
from google.cloud import storage  # type: ignore
from google.cloud.storage.bucket import Blob, Bucket  # type: ignore

from paqarin.evaluation import TrainingMetadata
from paqarin.generator import TimeSeriesGenerator

client = google.cloud.logging.Client()
client.setup_logging()
import logging  # noqa: E402

LOCAL_DIRECTORY: str = "~/temp/"


def process_gcs_uri(uri: str) -> tuple[str, str, str, str]:
    """Receives a Google Cloud Storage (GCS) uri and breaks it down.

    Divides it into the scheme, bucket, path and file. Adapted from:
    https://codelabs.developers.google.com/codelabs/vertex-ai-custom-code-training#3

    Arguments:
        uri (str): GCS uri

    Returns:
        scheme (str): uri scheme
        bucket (str): uri bucket
        path (str): uri path
        file (str): uri file
    """
    url_arr: list[str] = uri.split("/")
    if "." not in url_arr[-1]:
        file = ""
    else:
        file = url_arr.pop()
    scheme: str = url_arr[0]
    bucket: str = url_arr[2]
    path: str = "/".join(url_arr[3:])
    path = path[:-1] if path.endswith("/") else path

    return scheme, bucket, path, file


def load_data_from_gcs(data_gcs_path: str) -> pd.DataFrame:
    """Loads data from Google Cloud Storage (GCS) to a dataframe.

    Adapted from:
    https://codelabs.developers.google.com/codelabs/vertex-ai-custom-code-training#3

    Arguments:
        data_gcs_path (str): gs path for the location of the data.
        Wildcards are also supported. i.e
                gs://example_bucket/data/training-*.csv

    Returns:
        pandas.DataFrame: a dataframe with the data from GCP loaded
    """
    # using dask that supports wildcards to read multiple files. Then with
    #  dd.read_csv().compute
    # we create a pandas dataframe
    # Additionally I have noticed that some values for TotalCharges are missing and this
    # creates confusion regarding TotalCharges the data types.
    # to overcome this we manually define TotalCharges as object.
    # We will later fix this anomaly.
    logging.info("reading gs data: %s", data_gcs_path)
    return dd.read_csv(data_gcs_path).compute()


def export_generator_to_gcs(
    fitted_generator: TimeSeriesGenerator, model: str, gcs_bucket: str
):
    """Move the fitted model to a cloud bucket."""
    _, bucket_name, path, _ = process_gcs_uri(gcs_bucket)

    Path(LOCAL_DIRECTORY).mkdir(parents=True, exist_ok=True)
    fitted_generator.parameters.filename = LOCAL_DIRECTORY + model
    fitted_generator.save()
    bucket: Bucket = storage.Client().bucket(bucket_name)

    relative_paths: list = glob.glob(LOCAL_DIRECTORY + "/**", recursive=True)
    for local_file in relative_paths:
        remote_path: str = os.path.join(path, local_file[1 + len(LOCAL_DIRECTORY) :])

        if os.path.isfile(local_file):
            logging.info("Uploading %s to bucket %s", local_file, bucket_name)
            blob: Blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def get_generator_from_bucket(metadata_uri: str) -> TimeSeriesGenerator:
    """Given a pickle of metadata in a bucket, it returns the generator."""
    _, bucket_name, path, file = process_gcs_uri(metadata_uri)
    blob_name: str = os.path.join(path, file)
    logging.info(
        "Loading metadata configuration from bucket %s and blob %s",
        bucket_name,
        blob_name,
    )

    bucket: Bucket = storage.Client().bucket(bucket_name)
    blob: Blob = bucket.blob(blob_name)

    metadata: TrainingMetadata = pickle.loads(blob.download_as_string())
    return metadata.create_generator()


def main() -> None:
    """Starts the training process."""
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--training_metadata", type=str)
    parser.add_argument("--training_dataframe", type=str)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--target_bucket", type=str)

    arguments: dict = parser.parse_args().__dict__

    training_metadata: str = arguments["training_metadata"]
    generator: TimeSeriesGenerator = get_generator_from_bucket(training_metadata)

    training_dataframe: str = arguments["training_dataframe"]
    logging.info("Loading training data from %s", training_dataframe)
    dataframe: pd.DataFrame = load_data_from_gcs(training_dataframe)

    generator.fit(dataframe)

    model_file: str = arguments["model_file"]
    target_bucket: str = arguments["target_bucket"]
    export_generator_to_gcs(generator, model_file, target_bucket)

    logging.info(
        "Training completed. Model generated at %s exported to %s",
        model_file,
        target_bucket,
    )


if __name__ == "__main__":
    main()
