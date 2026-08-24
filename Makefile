.PHONY: help install format lint test check clean

.DEFAULT_GOAL := help

UV ?= uv
SRC_DIR ?= src
TEST_DIR ?= tests
PYTHON_PATHS ?= $(SRC_DIR) $(TEST_DIR)
PYTHON_FILES = $(shell find $(PYTHON_PATHS) -type f -name '*.py')
FAIL_UNDER ?= 9.5
FAIL_UNDER_TESTS ?= 9.5
PYLINT_TEST_DISABLE ?= redefined-outer-name

help:
	@echo "Available targets:"
	@echo "  make install  Synchronize the virtual environment"
	@echo "  make format   Format source code and tests"
	@echo "  make lint     Run all static checks without modifying files"
	@echo "  make test     Run the test suite"
	@echo "  make check    Run lint and tests"
	@echo "  make clean    Remove generated Python and tool caches"

install:
	$(UV) sync

format:
	$(UV) run isort $(PYTHON_PATHS)
	@set -e; for file in $(PYTHON_FILES); do \
		$(UV) run black --quiet "$$file"; \
	done

lint:
	$(UV) run isort --check-only $(PYTHON_PATHS)
	@set -e; for file in $(PYTHON_FILES); do \
		$(UV) run black --check --quiet "$$file"; \
	done
	$(UV) run bandit -r $(SRC_DIR)
	$(UV) run bandit -r $(TEST_DIR) --skip B101
	$(UV) run mypy $(PYTHON_PATHS)
	$(UV) run flake8 $(PYTHON_PATHS)
	$(UV) run ruff check $(PYTHON_PATHS)
	$(UV) run complexipy $(PYTHON_PATHS)
	$(UV) run pylint --fail-under=$(FAIL_UNDER) $(SRC_DIR)
	$(UV) run pylint --fail-under=$(FAIL_UNDER_TESTS) \
		--disable=$(PYLINT_TEST_DISABLE) $(TEST_DIR)

test:
	$(UV) run pytest $(TEST_DIR)

check: lint test

clean:
	find $(PYTHON_PATHS) -type d \( \
		-name __pycache__ -o \
		-name .pytest_cache -o \
		-name .mypy_cache -o \
		-name .ruff_cache \
	\) -prune -exec rm -rf {} +
	find $(PYTHON_PATHS) -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	rm -f .coverage
