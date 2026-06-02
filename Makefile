.PHONY: help setup lint format type test test-cov run ingest clean docker-build docker-up docker-down

help:
	@echo "Available targets:"
	@echo "  make setup          Install dev dependencies"
	@echo "  make lint           Lint with ruff"
	@echo "  make format         Format with ruff"
	@echo "  make type           Type-check with mypy"
	@echo "  make test           Run pytest"
	@echo "  make test-cov       Run pytest with coverage"
	@echo "  make ingest         Ingest sample documents"
	@echo "  make run            Run the FastAPI server locally"
	@echo "  make docker-build   Build the Docker image"
	@echo "  make docker-up      Start docker-compose (postgres + app)"
	@echo "  make docker-down    Stop docker-compose"
	@echo "  make clean          Remove .pyc, cache, etc."

setup:
	pip install -q -e ".[dev]"

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

type:
	mypy

test:
	pytest -q

test-cov:
	pytest --cov=prag --cov-report=html

ingest:
	PYTHONPATH=src python scripts/ingest_samples.py data/sample

run:
	uvicorn prag.api.app:app --reload --port 8000

docker-build:
	docker build -t prag:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -f prag_checkpoints.sqlite
