.PHONY: help install data lint fmt test check-endpoint experiment train predict evaluate docker-build docker-up clean

help:
	@echo "Targets:"
	@echo "  install       Install package + dev deps (editable)"
	@echo "  data          Download the Recruit dataset into data/raw (needs Kaggle creds)"
	@echo "  lint          ruff check"
	@echo "  fmt           black + ruff --fix"
	@echo "  test          pytest"
	@echo "  train         Train all models (application/train.py)"
	@echo "  predict       Batch forecast with the latest model (application/predict.py)"
	@echo "  evaluate      Temporal CV + comparison report/plots (application/evaluate.py)"
	@echo "  check-endpoint Probe the TimesFM HF Space (health + a sample forecast)"
	@echo "  experiment    Full model selection + report (find-best-model); logs to MLflow"
	@echo "  docker-build  Build the image (CPU-only; TimesFM is served over HTTP)"
	@echo "  docker-up     Run the full pipeline via docker compose"

install:
	pip install -e ".[dev]"

data:
	bash scripts/download_data.sh

lint:
	ruff check .

fmt:
	black .
	ruff check --fix .

test:
	pytest -q

train:
	python -m forecasting train

predict:
	python -m forecasting predict

evaluate:
	python -m forecasting evaluate

# Probe the Space. Reads FORECAST_TIMESFM_ENDPOINT[_TOKEN] from .env (sourced here) or the shell.
check-endpoint:
	set -a; [ -f .env ] && . ./.env || true; set +a; \
	python scripts/check_timesfm_endpoint.py "$$FORECAST_TIMESFM_ENDPOINT" "$$FORECAST_TIMESFM_ENDPOINT_TOKEN"

# Full 4-step selection + holdout report. python -m forecasting auto-loads .env (see config.py).
experiment:
	python -m forecasting find-best-model

docker-build:
	docker build -t gstock-forecasting .

docker-up:
	docker compose up

clean:
	rm -rf artifacts mlruns .pytest_cache .ruff_cache
