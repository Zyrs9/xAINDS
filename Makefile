# ==============================================================
# NIDS MVP — Makefile
#
# Build, train, and deploy the eBPF-powered NIDS system.
#
# Usage:
#   make install        — Install Python dependencies
#   make train          — Train model on UNSW-NB15 dataset
#   make compile-ebpf   — Compile eBPF C code (Linux only)
#   make api            — Start FastAPI dev server
#   make demo           — Run counterfactual SHAP demo
#   make build          — Build Docker images
#   make deploy         — Deploy with Docker Compose
#   make benchmark      — Run NVP vs MVP performance benchmark
#   make clean          — Remove build artifacts
# ==============================================================

.PHONY: install train api demo build deploy benchmark \
        stop clean test offline-test help

# Variables
PYTHON      ?= python3
PIP         ?= pip
INTERFACE   ?= eth0
DATASET     ?= NF-UNSW-NB15-v2.csv
COMPOSE     := docker-compose -f infrastructure/docker-compose.yml

# ── Help ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  NIDS MVP — Available Commands"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make install        Install Python dependencies"
	@echo "  make train          Train Isolation Forest on UNSW-NB15"
	@echo "  make api            Start FastAPI dev server"
	@echo "  make demo           Run counterfactual SHAP demo"
	@echo "  make offline-test   Test eBPF pipeline without kernel"
	@echo "  make build          Build Docker images"
	@echo "  make deploy         Start Docker Compose stack"
	@echo "  make stop           Stop Docker Compose stack"
	@echo "  make benchmark      Run NVP vs MVP benchmark"
	@echo "  make clean          Remove build artifacts"
	@echo ""

# ── Phase 0: Setup ────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt


# ── Phase 2: ML Training ─────────────────────────────────────
train:
	@echo "Training model on $(DATASET)..."
	$(PYTHON) -m src.ml.train_baseline
	@echo "[✔] Training complete. Check .pkl artifacts."

# ── Phase 2: XAI Demo ────────────────────────────────────────
demo:
	$(PYTHON) -m src.xai.counterfactual_demo

demo-auto:
	$(PYTHON) -m src.xai.counterfactual_demo --auto

# ── Phase 3: API Server ──────────────────────────────────────
api:
	$(PYTHON) -m src.api.server --reload

# ── Phase 1: eBPF Listener ───────────────────────────────────
listen:
	sudo $(PYTHON) -m src.ebpf.user_space --interface $(INTERFACE)

offline-test:
	$(PYTHON) -m src.ebpf.user_space --offline

# ── Phase 3: Docker ──────────────────────────────────────────
build:
	$(COMPOSE) build

deploy:
	$(COMPOSE) up -d
	@echo "[✔] NIDS MVP deployed. API at http://localhost:8000"

stop:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

# ── Phase 4: Benchmark ───────────────────────────────────────
benchmark:
	@echo "Starting NVP vs MVP performance benchmark..."
	$(PYTHON) benchmark/metrics_collector.py

# ── Cleanup ───────────────────────────────────────────────────
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "[✔] Clean complete."
