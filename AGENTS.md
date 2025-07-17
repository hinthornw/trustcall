<general_rules>
# General Development Rules

## Code Quality and Style
- Always run `make lint` before committing to ensure code passes ruff and mypy checks
- Use `make format` to automatically format code with ruff (includes import sorting and code formatting)
- Follow Google-style docstrings as configured in pyproject.toml
- Maintain type hints for all function parameters and return values
- When creating new functions in the trustcall package, first search existing modules (_base.py, _validation_node.py) to avoid duplication

## Development Workflow
- Use `uv` for all dependency management - never use pip directly
- Run `make tests` for unit tests before pushing changes
- Use `make tests_watch` for continuous testing during development
- For evaluation testing, use `make evals` (requires API keys)
- Always check that new code doesn't break existing functionality

## Import and Module Organization
- Public API should only expose necessary functions through trustcall/__init__.py
- Internal modules use underscore prefix (_base.py, _validation_node.py)
- Follow existing import patterns: langchain-core for LLM integration, langgraph for state management
- When adding new dependencies, update pyproject.toml and run `uv sync`
</general_rules>

<repository_structure>
# Repository Structure

## Core Package (trustcall/)
- `__init__.py`: Public API exposing create_extractor, ExtractionInputs, ExtractionOutputs
- `_base.py`: Main extraction logic, tool handling, JSON patch operations, and core extractor functionality
- `_validation_node.py`: ValidationNode class for tool call validation in LangGraph workflows
- `py.typed`: Indicates package supports type checking

## Testing Structure (tests/)
- `unit_tests/`: Core functionality tests (test_extraction.py, test_strict_existing.py, test_utils.py)
- `evals/`: Evaluation benchmarks using LangSmith for model comparison (test_evals.py)
- `cassettes/`: VCR cassettes for mocking API responses in tests
- `conftest.py`: Pytest configuration with asyncio backend setup

## Configuration and Build
- `pyproject.toml`: Project metadata, dependencies, tool configuration (ruff, mypy, pytest)
- `Makefile`: Common development commands (tests, lint, format, build, publish)
- `uv.lock`: Locked dependency versions managed by uv
- `.github/workflows/`: CI/CD with unit tests (test.yml) and daily evaluations (eval.yml)

## Documentation and Assets
- `README.md`: Comprehensive usage examples and API documentation
- `_static/`: Static assets (cover image)
- `LICENSE`: MIT license
</repository_structure>

<dependencies_and_installation>
# Dependencies and Installation

## Package Manager
- Uses `uv` for fast, reliable dependency management
- Never use pip directly - always use `uv run`, `uv sync`, or `uv add`
- Dependencies are defined in pyproject.toml with version constraints

## Core Dependencies
- `langgraph>=0.2.25`: State graph management for LLM workflows
- `dydantic<1.0.0,>=0.0.8`: Dynamic Pydantic model creation
- `jsonpatch<2.0,>=1.33`: JSON patch operations for efficient updates
- `langchain-core`: LLM integration and tool calling

## Development Dependencies
- Code quality: `ruff` (linting/formatting), `mypy` (type checking)
- Testing: `pytest`, `pytest-asyncio`, `pytest-socket`, `vcrpy`
- LLM providers: `langchain-openai`, `langchain-anthropic`, `langchain-fireworks`

## Installation Commands
- `uv sync --all-extras --dev`: Install all dependencies including dev tools
- `uv sync`: Install only production dependencies
- `uv add <package>`: Add new dependency
- `uv build`: Build distribution packages
</dependencies_and_installation>

<testing_instructions>
# Testing Instructions

## Test Framework and Structure
- Uses pytest with asyncio support for async/await testing patterns
- Socket access is disabled by default (`--disable-socket --allow-unix-socket`) to prevent external calls
- VCR cassettes in tests/cassettes/ and tests/evals/cassettes/ mock API responses

## Running Tests
- `make tests`: Run unit tests with socket restrictions and detailed output
- `make tests_watch`: Continuous testing during development (uses ptw)
- `make evals`: Run evaluation benchmarks (requires OPENAI_API_KEY, ANTHROPIC_API_KEY, LANGSMITH_API_KEY)
- `make doctest`: Run doctests in the trustcall module

## Test Categories
- **Unit Tests**: Core functionality testing without external API calls
  - test_extraction.py: Main extractor functionality and retry logic
  - test_strict_existing.py: Schema validation and existing data handling
  - test_utils.py: Utility functions like patch application and type conversion
- **Evaluation Tests**: LangSmith-integrated benchmarks comparing model performance
  - test_evals.py: Comparative evaluation across different LLM providers

## Writing Tests
- Use FakeExtractionModel for mocking LLM responses in unit tests
- Async tests should use pytest-asyncio decorators
- Mock external API calls using VCR cassettes or custom fake models
- Follow existing patterns for tool validation and schema testing
- Test both success and error scenarios, especially for validation failures
</testing_instructions>

