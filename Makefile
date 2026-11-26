# Convenience targets. Override CONFIG=... DOMAIN=... as needed.
CONFIG ?= configs/default.yaml
DOMAIN ?= beijing_housing
PY ?= python

.PHONY: install install-gpu test lint prepare index run ablation tables clean

install:            ## install core + llm deps (Mac/dev)
	pip install -e ".[llm,dev]"

install-gpu:        ## install everything incl. GPU extras (server)
	pip install -e ".[gpu,llm,dev]"

test:               ## run unit tests (pure-Python, no GPU/LLM needed)
	pytest -q

lint:
	ruff check src tests scripts

prepare:            ## raw datasets/ -> canonical benchmark
	$(PY) scripts/prepare_data.py --config $(CONFIG)

index:              ## GPU: build dense + sparse index
	$(PY) scripts/build_index.py --config $(CONFIG) --domain $(DOMAIN)

run:                ## run one method on one domain
	$(PY) scripts/run_experiment.py --config $(CONFIG) --domain $(DOMAIN) --method asof

ablation:           ## run all ablation variants
	$(PY) scripts/run_ablation.py --config $(CONFIG) --domain $(DOMAIN)

tables:             ## results -> paper LaTeX tables
	$(PY) scripts/make_tables.py --runs data/runs --out data/tables

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
