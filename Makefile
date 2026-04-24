.PHONY: help install test train calibrate run lint format docker-build docker-run

# Variables
PYTHON := python
VENV_BIN := venv/Scripts
PIP := $(VENV_BIN)/pip
UVICORN := $(VENV_BIN)/uvicorn
PYTEST := $(VENV_BIN)/pytest
RUFF := $(VENV_BIN)/ruff

help:
	@echo "Folsom AQI Backend Makefile"
	@echo "Usage:"
	@echo "  make install       Install dependencies in a virtual environment"
	@echo "  make run           Run the FastAPI server locally"
	@echo "  make test          Run unit tests"
	@echo "  make lint          Lint the code using ruff"
	@echo "  make format        Format the code using ruff"
	@echo "  make train         Train the ensemble model"
	@echo "  make calibrate     Run conformal prediction calibration"
	@echo "  make docker-build  Build the Docker container"
	@echo "  make docker-run    Run the Docker container locally"

install:
	$(PYTHON) -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest ruff mypy

run:
	$(UVICORN) api:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTEST) test_*.py -v

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

train:
	$(PYTHON) train_ensemble.py

calibrate:
	$(PYTHON) calibrate_coverage.py

docker-build:
	docker build -t folsom-aqi-backend .

docker-run:
	docker run -p 8000:8000 --env-file .env folsom-aqi-backend
