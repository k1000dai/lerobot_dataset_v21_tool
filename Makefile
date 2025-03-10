.PHONY: format lint ruff-fix ruff-check mypy test test-coverage

format: ruff-fix

ruff-fix:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

lint: ruff-check mypy

ruff-check:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

mypy:
	uv run mypy src/ tests/

test:
	uv run pytest tests/ -svv

test-coverage:
	uv run pytest tests/ --cov=src/ --cov=src --cov-report=term-missing
