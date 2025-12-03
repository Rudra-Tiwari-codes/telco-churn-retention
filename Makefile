.PHONY: help install install-dev lint format type-check test clean qa phase1 phase2

help: ## Show this help message
	@python -c "import re, sys; lines = [l.strip() for l in open('Makefile') if '##' in l]; [print(f\"  {re.search(r'^([a-zA-Z_-]+):', l).group(1):15s} {re.search(r'## (.*)', l).group(1)}\") for l in lines if re.search(r'^([a-zA-Z_-]+):', l)]" 2>/dev/null || (echo "Available commands:" && echo "  install          Install project dependencies" && echo "  install-dev      Install project with dev dependencies" && echo "  lint             Run linting checks" && echo "  format           Format code with black" && echo "  format-check     Check code formatting" && echo "  type-check       Run type checking" && echo "  test             Run tests" && echo "  test-cov         Run tests with coverage" && echo "  qa               Run all quality assurance checks" && echo "  clean            Clean build artifacts" && echo "  phase1           Run Phase 1 data intake" && echo "  phase2           Run Phase 2 feature engineering" && echo "  phase3           Run Phase 3 modeling" && echo "  phase4           Run Phase 4 API service" && echo "  phase4-streaming Run Phase 4 streaming pipeline" && echo "  phase5-monitoring Run Phase 5 monitoring" && echo "  phase5-retraining Run Phase 5 retraining" && echo "  pre-commit       Run pre-commit hooks")

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
	python -c "import shutil, pathlib, glob; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist'] + glob.glob('*.egg-info')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc') if p.is_file()]"
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.mypy_cache', '.ruff_cache'] if pathlib.Path(p).exists()]"

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

phase5-monitoring: ## Run Phase 5 monitoring workflow
	python scripts/run_phase5_monitoring_auto.py

phase5-retraining: ## Run Phase 5 retraining pipeline
	python scripts/run_phase5_retraining.py

phase6: ## Run Phase 6 business intelligence and executive delivery
	python scripts/run_phase6_business.py

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

