# How to Contribute to Paqarin

We welcome contributions from the community. To start, you will need the following software:

* Python **3.11**.
* Git.
* Make (or a Windows equivalent).
* Google Cloud SDK.


## Setting up the repository

1. You need to first clone the repository.

    ```
    git clone https://<TBA>/paqarin.git
    ```
2. From the root directory of the project, create a virtual environment named `.venv` using:

    ```
    python -m venv .venv
    ```
    **Make sure you're using the correct Python version!**

3. Activate the `.venv` virtual environment. In Windows, you should run:

    ```
    .venv\Scripts\activate
    ```

4. Once the `.venv` environment is active, run the following command to install **all** the project dependencies:

    ```
    make install
    make install-optional
    ```

    To access `make` from a Windows machine, you need to install `UnxUtils` first. Due to conflicting dependencies, **is highly likely that this process fails**. If this happens, run `python -m pip install -r env_state.txt` to get a exact copy of a working Python environment.

5. To make sure everything is working, run the tests with:

    ```
    make test
    ```

## Developing new Features

* We use [black](https://github.com/psf/black#:~:text=Black%20is%20the%20uncompromising%20Python,energy%20for%20more%20important%20matters.) to format our code. To apply it to your changes, just run `make format` from the project directory.
* Our style guide is determined by [flake8](https://github.com/PyCQA/flake8). To check if your code complies with our standards, run `make lint` from the project directory.
* We expect new features to include unit tests. To run [pytest](https://docs.pytest.org/en/7.4.x/) and produce a coverage report, run `make test` from the project directory.
* Before submitting changes, please run `make checklist` and verify your code does not produce errors. This routine executes flake8, the [mypy type checker](https://mypy.readthedocs.io/en/stable/), and all the unit tests.

## Code Structure

These are the main modules of the package. We can organise them in 3 groups:

### Algorithm Generators

* `generator.py`: This module contains abstractions for synthetic time series generation algorithms. If you're adding support for a new algorithm, you will need to extend the types defined in this module.

* `timegan.py`: This module contains components for generating time series using the [TimeGAN algorithm](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html?ref=https://githubhelp.com).

* `doppleganger.py`: This modules contains components for generating time series using the [DoppleGANger algorithm](https://dl.acm.org/doi/abs/10.1145/3419394.3423643).

* `par.py`: This module contains components for generating time series using the [CPAR algorithm](https://arxiv.org/abs/2207.14406).

### Implementation Adapters

* `adapter.py`: This module contains the abstractions for handling multiple libraries for synthetic time series generation. If you're adding support for a new library, make sure you register it in `get_generator_adapter`.

* `sdv_adapter.py`: This module contains the adapter code for running synthetic time series generation algorithms from [the SDV library](https://docs.sdv.dev/sdv).

* `synthcity_adapter.py`:  This module contains the adapter code for running synthetic time series generation algorithms from [the Synthcity library](https://synthcity.readthedocs.io/en/latest/).

* `ydata_adapter.py`: This module contains the adapter code for running synthetic time series generation algorithms from [the ydata-synthetic library](https://docs.synthetic.ydata.ai/1.3/).

### Synthetic Data Metrics

* `evaluation.py`: This modules contains components for gathering evaluation metrics of synthetic time series.

* `multivariate_metrics.py`:This module contains logic for evaluating synthetic time series on multivariate time series forecasting tasks, as proposed by [Yoon et al.](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html?ref=https://githubhelp.com).

* `univariate_metrics.py`: This module contains logic for evaluating synthetic time series on univariate time series forecasting tasks, using [the AutoGluon library](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-faq.html).

### Utilities

* `cloud_trainer.py`: This module contains the logic for training synthetic time series generation models in the Google Cloud Platform. **This is still work in progress, so use with care**.

* `data_plots.py`: This module contains plotting functionality for multivariate time series.

* `data_utils.py`: This module contains data processing functionality that's common to many synthetic data generation algorithms.

