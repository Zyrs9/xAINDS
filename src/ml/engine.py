"""
NIDS ML Engine — Cost-Sensitive Anomaly Detection with Alert Fatigue Management

This module replaces the naive `prediction == -1` logic with:
1. A cost-sensitive threshold derived from the ROC curve and a FP/FN cost matrix.
2. A rolling-window spike detector that suppresses low-severity noise and only
   escalates sustained anomaly bursts to CRITICAL.

Academic Reference:
    - Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning."
    - UNSW-NB15 Dataset: Moustafa & Slay (2015).

Author: NIDS Research Team
"""

import time
import json
import logging
import numpy as np
import joblib
import threading
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from enum import Enum

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

# Default cost ratio: missing an attack is 10x worse than a false alarm
DEFAULT_COST_FP = 1.0    # Cost of a False Positive (analyst time wasted)
DEFAULT_COST_FN = 10.0   # Cost of a False Negative (attack undetected)

# Spike detection parameters
ROLLING_WINDOW_SECONDS = 60   # Time window for moving average
SPIKE_SIGMA_MULTIPLIER = 1.5  # Spike = score > mean + (multiplier * std_dev)

# The canonical 10-feature set from UNSW-NB15 (trained via train_baseline.py)
CANONICAL_FEATURES = [
    "MIN_TTL",
    "MAX_TTL",
    "SHORTEST_FLOW_PKT",
    "LONGEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "OUT_BYTES",
    "OUT_PKTS",
    "DST_TO_SRC_SECOND_BYTES",
    "NUM_PKTS_UP_TO_128_BYTES",
]

logger = logging.getLogger("nids.engine")


class AlertLevel(Enum):
    """Three-tier alert classification to combat SOC alert fatigue."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DetectionResult:
    """Structured output of a single flow analysis."""
    timestamp: float
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    alert_level: str            # AlertLevel value
    features: Dict[str, float]
    spike_info: Optional[Dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Cost-Sensitive Threshold Calculator
# ---------------------------------------------------------------------------

class CostSensitiveThreshold:
    """
    Computes the optimal decision boundary for Isolation Forest by minimizing
    the total misclassification cost on a validation set.

    The Isolation Forest `decision_function` returns a signed anomaly score:
        - Large positive → clearly normal
        - Near zero      → borderline
        - Negative        → clearly anomalous

    Instead of using the default 0.0 cutoff, we sweep thresholds on a
    labelled validation set and pick the one that minimizes:

        TotalCost = C_FP * FP_count + C_FN * FN_count

    This is the standard Elkan (2001) cost-sensitive approach.
    """

    def __init__(self, cost_fp: float = DEFAULT_COST_FP,
                 cost_fn: float = DEFAULT_COST_FN):
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.optimal_threshold = 0.0  # Will be calibrated

    def calibrate(self, scores: np.ndarray, labels: np.ndarray,
                  n_thresholds: int = 500) -> float:
        """
        Sweep thresholds across the score range and find the cost-optimal one.

        Args:
            scores: Anomaly scores from model.decision_function(X_val).
            labels: Ground truth (0 = normal, 1 = attack).
            n_thresholds: Resolution of the threshold sweep.

        Returns:
            The optimal threshold value.
        """
        # Isolation Forest convention: score < threshold → anomaly
        # We sweep from min(scores) to max(scores)
        thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

        best_cost = np.inf
        best_threshold = 0.0
        cost_curve = []

        for t in thresholds:
            # Predict: anything below threshold is flagged as anomaly
            predicted_anomaly = scores < t

            # Ground truth: label == 1 means actual attack
            actual_attack = labels == 1

            # False Positive: predicted anomaly but actually normal
            fp = np.sum(predicted_anomaly & ~actual_attack)
            # False Negative: predicted normal but actually attack
            fn = np.sum(~predicted_anomaly & actual_attack)

            total_cost = (self.cost_fp * fp) + (self.cost_fn * fn)
            cost_curve.append((float(t), float(total_cost), int(fp), int(fn)))

            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = float(t)

        self.optimal_threshold = best_threshold
        logger.info(
            f"Cost-sensitive calibration complete. "
            f"Optimal threshold: {best_threshold:.6f} "
            f"(C_FP={self.cost_fp}, C_FN={self.cost_fn}, "
            f"MinCost={best_cost:.1f})"
        )

        return best_threshold

    def calibrate_multi_ratio(self, scores: np.ndarray, labels: np.ndarray,
                               ratios: List[Tuple[float, float]] = None
                               ) -> Dict[str, float]:
        """
        Calibrate across multiple cost ratios for the IEEE paper's
        sensitivity analysis graph.

        Args:
            scores: Anomaly scores from decision_function.
            labels: Ground truth labels.
            ratios: List of (C_FP, C_FN) tuples. Defaults to 1:5, 1:10, 1:20.

        Returns:
            Dictionary mapping "ratio_label" → optimal_threshold.
        """
        if ratios is None:
            ratios = [(1.0, 5.0), (1.0, 10.0), (1.0, 20.0)]

        results = {}
        for c_fp, c_fn in ratios:
            calc = CostSensitiveThreshold(cost_fp=c_fp, cost_fn=c_fn)
            t = calc.calibrate(scores, labels)
            key = f"1:{int(c_fn / c_fp)}"
            results[key] = t
            logger.info(f"  Ratio {key} → threshold = {t:.6f}")

        return results


# ---------------------------------------------------------------------------
# Spike Detector (Alert Fatigue Management)
# ---------------------------------------------------------------------------

class SpikeDetector:
    """
    Implements a rolling-window anomaly rate tracker to distinguish between
    persistent threats (CRITICAL) and transient noise (WARNING).

    Mechanism:
        - Maintains a deque of (timestamp, score) tuples within the window.
        - Computes moving average (μ) and standard deviation (σ) of scores.
        - A score is classified as a SPIKE if:  score > μ + (k * σ)
        - Only spikes produce CRITICAL alerts.

    This directly addresses SOC alert fatigue — the #1 operational complaint
    in enterprise security operations centers.
    """

    def __init__(self, window_seconds: float = ROLLING_WINDOW_SECONDS,
                 sigma_multiplier: float = SPIKE_SIGMA_MULTIPLIER):
        self.window_seconds = window_seconds
        self.sigma_multiplier = sigma_multiplier
        self._history: deque = deque()
        self._running_sum = 0.0
        self._running_sq_sum = 0.0

    def _evict_expired(self, now: float) -> None:
        """Remove entries older than the rolling window."""
        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            _, old_intensity = self._history.popleft()
            self._running_sum -= old_intensity
            self._running_sq_sum -= (old_intensity * old_intensity)

    def record_and_classify(self, score: float, is_anomaly: bool,
                             timestamp: float = None) -> Tuple[AlertLevel, Dict]:
        """
        Record a new anomaly score and determine the alert level.

        Args:
            score: The raw anomaly score (from decision_function; lower = more anomalous).
            is_anomaly: Whether the cost-sensitive threshold flagged this as anomaly.
            timestamp: Optional explicit timestamp (defaults to time.time()).

        Returns:
            (AlertLevel, spike_info_dict)
        """
        now = timestamp or time.time()
        self._evict_expired(now)

        # We track the absolute deviation from zero (anomaly intensity).
        # More negative scores = more anomalous, so we negate for spike math.
        intensity = -score  # Higher intensity = more anomalous

        self._history.append((now, intensity))
        self._running_sum += intensity
        self._running_sq_sum += (intensity * intensity)

        # Calculate rolling statistics
        n = len(self._history)
        if n < 3:
            # Not enough data for meaningful statistics
            spike_info = {
                "window_size": n,
                "mean": float(intensity),
                "std": 0.0,
                "spike_threshold": float(intensity),
                "is_spike": False,
            }
            level = AlertLevel.WARNING if is_anomaly else AlertLevel.NORMAL
            return level, spike_info

        mean = float(self._running_sum / n)
        # Variance calculation with fallback for numerical precision
        var = (self._running_sq_sum - (self._running_sum ** 2) / n) / n
        std = float(max(0, var) ** 0.5)
        spike_threshold = mean + (self.sigma_multiplier * std)

        is_spike = intensity > spike_threshold

        spike_info = {
            "window_size": len(self._history),
            "window_seconds": self.window_seconds,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "spike_threshold": round(spike_threshold, 6),
            "current_intensity": round(float(intensity), 6),
            "is_spike": is_spike,
        }

        # Three-tier classification
        if is_anomaly and is_spike:
            level = AlertLevel.CRITICAL
        elif is_anomaly:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.NORMAL

        return level, spike_info


# ---------------------------------------------------------------------------
# Main Inference Engine
# ---------------------------------------------------------------------------

class NIDSEngine:
    """
    The core ML inference engine for the MVP NIDS.

    Lifecycle:
        1. Load pre-trained artifacts (model, scaler, threshold, features)
        2. Accept 10-feature flow vectors from eBPF user-space
        3. Scale → predict → cost-threshold → spike-classify
        4. Return structured DetectionResult
    """

    def __init__(self, artifacts_dir: str = ".",
                 cost_fp: float = DEFAULT_COST_FP,
                 cost_fn: float = DEFAULT_COST_FN,
                 window_seconds: float = ROLLING_WINDOW_SECONDS):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.scaler = None
        self.top_features: List[str] = []
        self.threshold = 0.0
        self.cost_calculator = CostSensitiveThreshold(cost_fp, cost_fn)
        self.spike_detector = SpikeDetector(window_seconds=window_seconds)
        self._initialized = False
        self._alert_log: deque = deque(maxlen=100000)
        self._log_lock = threading.Lock()

    def load_artifacts(self,
                       model_path: str = None,
                       scaler_path: str = None,
                       features_path: str = None,
                       threshold_path: str = None) -> None:
        """
        Load pre-trained Isolation Forest artifacts from disk.

        These are produced by `src/ml/train_baseline.py` (offline training
        on UNSW-NB15 — no synthetic data).
        """
        base = self.artifacts_dir

        model_path = model_path or str(base / "isolation_forest_model.pkl")
        scaler_path = scaler_path or str(base / "scaler.pkl")
        features_path = features_path or str(base / "top_features.pkl")
        threshold_path = threshold_path or str(base / "optimal_threshold.pkl")

        logger.info("Loading model artifacts...")

        self.model = joblib.load(model_path)
        logger.info(f"  [✔] Model loaded: {model_path}")

        self.scaler = joblib.load(scaler_path)
        logger.info(f"  [✔] Scaler loaded: {scaler_path}")

        self.top_features = joblib.load(features_path)
        logger.info(f"  [✔] Features ({len(self.top_features)}): {self.top_features}")

        # Load calibrated threshold if available; otherwise use default 0.0
        threshold_file = Path(threshold_path)
        if threshold_file.exists():
            self.threshold = joblib.load(threshold_path)
            logger.info(f"  [✔] Cost-sensitive threshold: {self.threshold:.6f}")
        else:
            self.threshold = 0.0
            logger.warning(
                f"  [!] Threshold file not found at {threshold_path}. "
                f"Using default threshold=0.0. Run train_baseline.py to calibrate."
            )

        self._initialized = True
        logger.info("Engine initialized successfully.")

    def analyze(self, feature_vector: np.ndarray,
                feature_dict: Dict[str, float] = None) -> DetectionResult:
        """
        Run full inference pipeline on a single flow.

        Args:
            feature_vector: Raw 10-feature numpy array (unscaled).
            feature_dict: Optional pre-built {name: value} dict for the result.

        Returns:
            DetectionResult with score, threshold, alert level, and spike info.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call load_artifacts() first.")

        # Ensure 2D
        X = np.atleast_2d(feature_vector)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Anomaly score (lower = more anomalous)
        score = float(self.model.decision_function(X_scaled)[0])

        # Cost-sensitive decision: anomaly if score < calibrated threshold
        is_anomaly = score < self.threshold

        # Spike detection
        alert_level, spike_info = self.spike_detector.record_and_classify(
            score=score, is_anomaly=is_anomaly
        )

        # Build feature dict if not provided
        if feature_dict is None:
            feature_dict = {
                name: float(X[0, i])
                for i, name in enumerate(self.top_features)
            }

        result = DetectionResult(
            timestamp=time.time(),
            anomaly_score=round(score, 6),
            threshold=round(self.threshold, 6),
            is_anomaly=is_anomaly,
            alert_level=alert_level.value,
            features=feature_dict,
            spike_info=spike_info,
        )

        with self._log_lock:
            self._alert_log.append(result)

        return result

    def get_recent_alerts(self, n: int = 50,
                          level_filter: str = None) -> List[dict]:
        """Retrieve recent alerts, optionally filtered by level."""
        with self._log_lock:
            # deque sliding needs to be converted to list
            alerts = list(self._alert_log)[-n:]
        if level_filter:
            alerts = [a for a in alerts if a.alert_level == level_filter]
        return [a.to_dict() for a in alerts]

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# Module-level convenience (for direct script testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

    engine = NIDSEngine(artifacts_dir=".")
    engine.load_artifacts()

    # Quick smoke test with a synthetic flow
    test_flow = np.array([64, 128, 40, 1500, 20, 1480, 50000, 30, 12000, 45],
                         dtype=np.float64)

    result = engine.analyze(test_flow)
    print("\n--- Detection Result ---")
    print(result.to_json())
