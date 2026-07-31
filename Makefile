.DEFAULT_GOAL := help
PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin
RELEASES ?= var/releases
DEVICE ?= var/device

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install with dev tools
	$(BIN)/pip install -e '.[dev]'

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: lint
lint: ## Lint and format-check
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

.PHONY: fmt
fmt: ## Auto-format
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

.PHONY: demo
demo: ## The whole down-link: v1 ships, v2 is adopted, v3 is refused and quarantined
	$(BIN)/edge-runtime demo

.PHONY: quantize
quantize: ## int8 fidelity, per-tensor vs per-channel, on the same data
	$(BIN)/edge-runtime quantise

.PHONY: bench
bench: ## Single-observation latency against a control deadline
	$(BIN)/edge-runtime bench --steps 5000

.PHONY: serve
serve: ## Development release + telemetry endpoint on :8720
	$(BIN)/edge-runtime serve-releases --releases $(RELEASES)

.PHONY: clean
clean: ## Remove device state, releases and build output
	rm -rf var build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean ## Also remove the virtualenv
	rm -rf $(VENV) .pytest_cache .ruff_cache
