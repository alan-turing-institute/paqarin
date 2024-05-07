install:
	python -m pip install --editable .

install-optional:
	python -m pip install .[dev]

.PHONY: lint
lint:
	python -m flake8 paqarin tests

format:
	python -m isort --use-parentheses --profile black paqarin tests
	python -m black paqarin tests

.PHONY: typehint
typehint:
	python -m mypy paqarin tests

.PHONY: test
test:
	python -m pytest -vv --cov=paqarin tests

.PHONY: checklist
checklist: format lint typehint test


