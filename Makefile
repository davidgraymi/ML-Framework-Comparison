.PHONY: format check lint test

format:
	uv run --extra dev python scripts/format.py

check:
	uv run --extra dev python scripts/format.py --check

lint:
	uv run --extra dev python scripts/format.py --check

test:
	uv run --extra dev python -m pytest tests/
