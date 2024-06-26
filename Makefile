.PHONY: tests lint format doctest integration_tests_fast evals

tests:
	poetry run python -m pytest --disable-socket --allow-unix-socket -n auto --durations=10 tests/unit_tests

tests_watch:
	poetry run ptw --now . -- -vv -x  tests/unit_tests

integration_tests:
	poetry run python -m pytest -v --durations=10 --cov=trustcall --cov-report=term-missing --cov-report=html --cov-config=.coveragerc tests/integration_tests

integration_tests_fast:
	poetry run python -m pytest -n auto --durations=10 -v --cov=trustcall --cov-report=term-missing --cov-report=html --cov-config=.coveragerc tests/integration_tests

evals:
	poetry run python -m pytest -p no:asyncio  --max_asyncio_tasks 4 tests/evals

lint:
	poetry run ruff check .
	poetry run mypy .

doctest:
	poetry run python -m pytest -n auto --durations=10 --doctest-modules trustcall

format:
	ruff check --select I --fix
	poetry run ruff format .
	poetry run ruff check . --fix

build:
	poetry build

publish:
	poetry publish --dry-run
