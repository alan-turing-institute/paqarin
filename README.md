# Paqarin
 A library for the generation of synthetic time series data.

## Installation

Paqarin was tested using **Python 3.10**.
We strongly suggest you to create a virtual environment for running this code.
For creating the `.venv` environment, do the following:

```
python -m venv .venv
```

Paqarin relies on AutoGluon for utility evaluation. As such, we expect you to have [OpenMP installed in your system](https://uk.mathworks.com/help/coder/ug/install-openmp-library-on-macos-platform.html).
Also, the installation of [LightGBM might be problematic, if working on a MacBook with M1](https://stackoverflow.com/questions/47098281/during-installation-of-lightgbm-it-says-that-you-should-install-cmake-first-whi).


Once created, activate it and install the Paqarin package using the `install_paqarin.bat` script:

```bash
.\pyenv\Scripts\activate
install_paqarin.bat <INDEX_URL> <PROVIDER_FLAGS>
```

Where `<INDEX_URL>` is the URL of the Python Package Index you want to use, and `<PROVIDER_FLAGS>` configure the provider libraries you want to install. Currently, we support the following libraries:

* For [ydata-synthetic](https://docs.synthetic.ydata.ai/1.3/), use the flag `/ydata`.
* For [sdv](https://docs.sdv.dev/sdv), use the flag `/sdv`.
* For [synthcity](https://synthcity.readthedocs.io/en/latest/), use the flag `/synthcity`.


Installation can take several minutes.
Depending on your connectivity, you might need several runs to have all the dependencies in place.

To verify the installation succeeded, you can try running one of our examples, using YData's implementation of the DoppleGANger algorithm:

```bash
cd examples
python doppleganger_example.py
```

## Usage

Paqarin exposes multiple synthetic time series generation algorithms, along with metrics to evaluate their performance. Using Paqarin, you can select which technique is better for your use case.

For example, to use the [DoppleGanger algorithm](https://dl.acm.org/doi/abs/10.1145/3419394.3423643), as implemented by [ydata-synthetic](https://docs.synthetic.ydata.ai/1.3/), we do the following:

```python
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
```

Then, to calculate the [predictive score](https://proceedings.neurips.cc/paper_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html), after training a forecasting model for multiple iterations, we can do:

```python
evaluation_pipeline: EvaluationPipeline = EvaluationPipeline(
    generator_map={"doppleganger": doppleganger_generator},
    scoring=PredictiveScorer(
        lstm_units=12,
        iterations=3,
        scorer_epochs=100,
        scorer_batch_size=128,
        number_of_features=2,
        numerical_columns=["traffic_byte_counter", "ping_loss_rate"],
        sequence_length=56,
        metric_value_key="mean_absolute_error")
)

evaluation_pipeline.fit(pd.read_csv("fcc_mba.csv"))
```
This will calculate the mean absolute error over multiple iterations, for both training over real data and using synthetic data.
For additional details, please refer to [our examples](examples/dopplenganger_example.py).

## Maturity

Paqarin should be considered **experimental**.
It comes with no support, but we are keen to receive feedback and suggestions on how to improve it.
Paqarin is not meant to be used in production environments, and the risks of its deployment are unknown.
