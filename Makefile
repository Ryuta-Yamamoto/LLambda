.PHONY: install test lint format check clean dist

help:
	@echo "    install"
	@echo "        Install dependencies."
	@echo "    test"
	@echo "        Run tests."
	@echo "    lint"
	@echo "        Lint project with mypy."
	@echo "    format"
	@echo "        Format project with ruff."
	@echo "    check"
	@echo "        Check the package for common errors with poetry check."
	@echo "    clean"
	@echo "        Clean the directory."
	@echo "    dist"
	@echo "        Build source and wheel distribution."

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run mypy llambda examples
	poetry run ruff llambda examples tests

format:
	poetry run ruff --fix llambda examples tests

check:
	make lint
	make test

clean:
	find . -type d -name '__pycache__' -delete
	rm -fr .pytest_cache/
	rm -fr .mypy_cache/
	rm -fr .ruff_cache/
	rm -fr dist/
	rm -fr build/
	rm -fr *.egg-info

# dist:
# 	poetry build
