"""
NIDS MVP — FastAPI Backend Server

Clean, production-grade API for the MVP architecture.
Replaces the legacy main.py and backend/app.py with a unified server.

Endpoints:
    GET  /health         — System status and readiness check
    POST /analyze        — Analyze a 10-feature flow vector
    GET  /alerts         — Retrieve recent alerts with optional level filter
    GET  /alerts/stats   — Aggregated alert statistics
    POST /explain        — Full SHAP + NLG explanation for a flow

Author: NIDS Research Team
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ml.engine import NIDSEngine, CANONICAL_FEATURES
from src.xai.explainer import NIDSExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("nids.api")

# ---------------------------------------------------------------------------
# Global Instances
# ---------------------------------------------------------------------------

engine: Optional[NIDSEngine] = None
explainer: Optional[NIDSExplainer] = None

ARTIFACTS_DIR = "."  # Override via environment or CLI


# ---------------------------------------------------------------------------
# Lifespan (Startup / Shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ML engine and SHAP explainer on startup."""
    global engine, explainer

    logger.info("=" * 60)
    logger.info("NIDS MVP API — Starting Up")
    logger.info("=" * 60)

    engine = NIDSEngine(artifacts_dir=ARTIFACTS_DIR)
    try:
        engine.load_artifacts()
    except FileNotFoundError as e:
        logger.error(f"Failed to load model artifacts: {e}")
        logger.error("Run 'python -m src.ml.train_baseline' first.")
        raise

    explainer = NIDSExplainer.from_artifacts(ARTIFACTS_DIR)
    logger.info("[✔] API ready to receive flow data.")

    yield

    logger.info("NIDS MVP API — Shutting Down")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-Driven NIDS — MVP Inference Engine",
    description=(
        "Explainable AI-Driven Lightweight Network Intrusion Detection System. "
        "Uses eBPF for data plane, cost-sensitive Isolation Forest for detection, "
        "and SHAP with NLG for explainability."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response Models (Pydantic)
# ---------------------------------------------------------------------------

class FlowInput(BaseModel):
    """Input model for flow analysis."""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of 10 flow features matching the UNSW-NB15 feature set.",
        json_schema_extra={
            "example": {
                "MIN_TTL": 64.0,
                "MAX_TTL": 64.0,
                "SHORTEST_FLOW_PKT": 54.0,
                "LONGEST_FLOW_PKT": 1500.0,
                "MIN_IP_PKT_LEN": 40.0,
                "MAX_IP_PKT_LEN": 1480.0,
                "OUT_BYTES": 2500.0,
                "OUT_PKTS": 15.0,
                "DST_TO_SRC_SECOND_BYTES": 45000.0,
                "NUM_PKTS_UP_TO_128_BYTES": 8.0,
            }
        }
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features_count: int
    threshold: float
    uptime_seconds: float


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

_start_time = time.time()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """System status and model readiness check."""
    return HealthResponse(
        status="online" if engine and engine.is_initialized else "degraded",
        model_loaded=engine.is_initialized if engine else False,
        features_count=len(engine.top_features) if engine else 0,
        threshold=engine.threshold if engine else 0.0,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/analyze")
def analyze_flow(data: FlowInput):
    """
    Analyze a network flow for anomalies.

    Accepts a 10-feature dictionary, runs cost-sensitive Isolation Forest,
    and returns detection result with alert level.
    """
    if not engine or not engine.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="ML engine not initialized. Check server logs."
        )

    # Validate all required features are present
    missing = [f for f in engine.top_features if f not in data.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}"
        )

    try:
        # Build feature vector in canonical order
        feature_vector = np.array(
            [data.features[f] for f in engine.top_features],
            dtype=np.float64
        )

        result = engine.analyze(feature_vector, feature_dict=data.features)

        return {
            "timestamp": result.timestamp,
            "prediction": -1 if result.is_anomaly else 1,
            "status": "Anomaly" if result.is_anomaly else "Normal",
            "anomaly_score": result.anomaly_score,
            "threshold": result.threshold,
            "alert_level": result.alert_level,
            "spike_info": result.spike_info,
        }

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/explain")
def explain_flow(data: FlowInput):
    """
    Full SHAP + NLG explanation for a network flow.

    Returns ranked feature attributions with human-readable insights
    and MITRE ATT&CK technique mappings.
    """
    if not engine or not explainer:
        raise HTTPException(
            status_code=503,
            detail="Engine or explainer not initialized."
        )

    missing = [f for f in engine.top_features if f not in data.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}"
        )

    try:
        feature_vector = np.array(
            [data.features[f] for f in engine.top_features],
            dtype=np.float64
        )

        # Scale for SHAP
        X_scaled = engine.scaler.transform(np.atleast_2d(feature_vector))

        # Get full explanation report
        report = explainer.explain(
            X_scaled,
            feature_values=data.features,
            top_k=10
        )

        # Also run detection for context
        result = engine.analyze(feature_vector, feature_dict=data.features)

        return {
            "detection": {
                "status": "Anomaly" if result.is_anomaly else "Normal",
                "anomaly_score": result.anomaly_score,
                "alert_level": result.alert_level,
            },
            "explanation": report.to_dict(),
        }

    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.get("/alerts")
def get_alerts(
    n: int = Query(50, description="Number of recent alerts to retrieve"),
    level: Optional[str] = Query(None, description="Filter by alert level: NORMAL, WARNING, CRITICAL")
):
    """Retrieve recent alerts with optional level filtering."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    alerts = engine.get_recent_alerts(n=n, level_filter=level)
    return {
        "count": len(alerts),
        "alerts": alerts,
    }


@app.get("/alerts/stats")
def get_alert_stats():
    """Aggregated alert statistics."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    all_alerts = engine.get_recent_alerts(n=10000)
    total = len(all_alerts)

    if total == 0:
        return {"total": 0, "by_level": {}, "anomaly_rate": 0.0}

    by_level = {}
    for alert in all_alerts:
        level = alert["alert_level"]
        by_level[level] = by_level.get(level, 0) + 1

    anomaly_count = by_level.get("WARNING", 0) + by_level.get("CRITICAL", 0)

    return {
        "total": total,
        "by_level": by_level,
        "anomaly_rate": round(anomaly_count / total, 4),
    }


# ---------------------------------------------------------------------------
# Server Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIDS MVP API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--artifacts-dir", default=".", help="Model artifacts directory")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    ARTIFACTS_DIR = args.artifacts_dir

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
