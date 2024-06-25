.PHONY: tests lint format doctest integration_tests_fast evals

tests:
	poetry run python -m pytest --disable-socket --allow-unix-socket -n auto --durations=10 tests/unit_tests

tests_watch:
	poetry run ptw --now . -- -vv -x  tests/unit_tests

integration_tests:
	poetry run python -m pytest -v --durations=10 --cov=langsmith --cov-report=term-missing --cov-report=html --cov-config=.coveragerc tests/integration_tests

integration_tests_fast:
	poetry run python -m pytest -n auto --durations=10 -v --cov=langsmith --cov-report=term-missing --cov-report=html --cov-config=.coveragerc tests/integration_tests

evals:
	poetry run python -m pytest tests/evaluation

lint:
	poetry run ruff check .
	poetry run mypy .

format:
	poetry run ruff format .
	poetry run ruff check . --fix

build:
	poetry build

publish:
	poetry publish --dry-run
