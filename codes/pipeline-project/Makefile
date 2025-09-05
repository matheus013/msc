.PHONY: venv install test run lint format clean

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -U pip && pip install kedro>=0.19.0 pandas pytest ruff

test:
	. .venv/bin/activate && pytest -q

run:
	. .venv/bin/activate && kedro run

viz:
	. .venv/bin/activate && kedro viz

lint:
	. .venv/bin/activate && ruff check src tests

format:
	. .venv/bin/activate && ruff format src tests

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ .kedro-*/ .ruff_cache
