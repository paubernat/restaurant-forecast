.PHONY: help install data lint fmt test check-endpoint train predict evaluate docker-up clean

help:
	@echo "Targets:"
	@echo "  install       Install package + dev deps (editable)"
	@echo "  data          Download the Recruit dataset into data/raw (needs Kaggle creds)"
	@echo "  lint          ruff check"
	@echo "  fmt           black + ruff --fix"
	@echo "  test          pytest"
	@echo "  evaluate      Temporal CV + selection + comparison report/plots; logs to MLflow"
	@echo "  train         Refit the selected winner on all data -> best_model.pkl"
	@echo "  predict       Batch forecast with the latest model -> forecasts.csv"
	@echo "  check-endpoint Probe the TimesFM HF Space (health + a sample forecast)"
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

docker-up:
	docker compose up

clean:
	rm -rf artifacts mlruns .pytest_cache .ruff_cache
