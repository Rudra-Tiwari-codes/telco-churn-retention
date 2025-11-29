.PHONY: help install install-dev lint format type-check test clean qa phase1 phase2

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	pip install -e .

install-dev: ## Install project with dev dependencies
	pip install -e ".[dev]"

lint: ## Run linting checks
	ruff check src/ tests/ scripts/

format: ## Format code with black
	black src/ tests/ scripts/

format-check: ## Check code formatting without making changes
	black --check src/ tests/ scripts/

type-check: ## Run type checking with mypy
	mypy src/ || true

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

qa: lint format-check type-check test ## Run all quality assurance checks

clean: ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true

phase1: ## Run Phase 1 data intake workflow
	python scripts/run_phase1_data_intake.py

phase2: ## Run Phase 2 feature engineering workflow
	python scripts/run_phase2_feature_engineering.py --use-latest

phase3: ## Run Phase 3 modeling workflow
	python scripts/run_phase3_modeling.py --use-latest

phase4: ## Run Phase 4 API service
	python scripts/run_phase4_api.py

phase4-streaming: ## Run Phase 4 streaming pipeline (simulator)
	python scripts/run_phase4_streaming.py --simulate

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

