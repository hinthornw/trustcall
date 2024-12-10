.PHONY: tests lint format doctest integration_tests_fast evals

tests:
	uv run python -m pytest --disable-socket --allow-unix-socket -n auto --durations=10 tests/unit_tests

tests_watch:
	uv run ptw --now . -- -vv -x  tests/unit_tests

integration_tests:
	uv run python -m pytest -v --durations=10 --cov=trustcall --cov-report=term-missing --cov-report=html --cov-config=.coveragerc tests/integration_tests

integration_tests_fast:
	uv run python -m pytest -n auto --durations=10 -v --cov=trustcall --cov-report=term-missing --cov-report=html --cov-config=.coveragerc tests/integration_tests

evals:
	LANGCHAIN_TEST_CACHE=tests/evals/cassettes uv run pytest  tests/evals/test_evals.py

lint:
	uv run ruff check .
	uv run mypy .

doctest:
	uv run python -m pytest -n auto --durations=10 --doctest-modules trustcall

format:
	ruff check --select I --fix
	uv run ruff format .
	uv run ruff check . --fix

build:
	uv build

publish:
	uv publish --dry-run
