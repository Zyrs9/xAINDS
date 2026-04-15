"""
NIDS Training Pipeline — Offline Model Training on UNSW-NB15

This script is the single source of truth for model training. It replaces
ALL synthetic data generation with real-world labelled NetFlow data.

Pipeline Steps:
    1. Load and clean NF-UNSW-NB15-v2.csv
    2. Feature selection via RandomForest importance (Top 10)
    3. Train Isolation Forest on normal traffic only (unsupervised baseline)
    4. Calibrate cost-sensitive threshold on a validation split
    5. Export: model, scaler, features list, optimal threshold

Output Artifacts:
    - isolation_forest_model.pkl   (trained Isolation Forest)
    - scaler.pkl                   (StandardScaler fitted on normal data)
    - top_features.pkl             (ordered list of 10 feature names)
    - optimal_threshold.pkl        (cost-calibrated decision boundary)
    - threshold_sensitivity.json   (multi-ratio analysis for IEEE paper)

Author: NIDS Research Team
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our cost-sensitive threshold module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ml.engine import CostSensitiveThreshold, DEFAULT_COST_FP, DEFAULT_COST_FN

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("nids.train")


def train_nids_pipeline(
    dataset_path: str = "NF-UNSW-NB15-v2.csv",
    output_dir: str = ".",
    n_top_features: int = 10,
    contamination: float = 0.01,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Full training pipeline: load → clean → select → train → calibrate → export.

    Args:
        dataset_path: Path to the NF-UNSW-NB15-v2.csv dataset.
        output_dir:   Directory for exported .pkl artifacts.
        n_top_features: Number of features to select (default: 10).
        contamination:  Isolation Forest contamination parameter.
        cost_fp:        False Positive cost for threshold calibration.
        cost_fn:        False Negative cost for threshold calibration.
        test_size:      Fraction of data held out for threshold calibration.
        random_state:   Random seed for reproducibility.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # STEP 1: DATA LOADING & CLEANING
    # ===================================================================
    logger.info(f"Loading dataset: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logger.error(
            f"Dataset '{dataset_path}' not found. "
            f"Download NF-UNSW-NB15-v2.csv and place it in the project root."
        )
        return

    logger.info(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Drop identity columns that cause memorization instead of behavioral learning
    identity_cols = [
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
        "L2_IPV4_CLIENT_MAC", "L2_IPV4_SERVER_MAC",
        "id", "Attack",
    ]
    existing_drops = [c for c in identity_cols if c in df.columns]
    df_cleaned = df.drop(columns=existing_drops)

    # Separate target label
    y = df_cleaned["Label"]
    X = df_cleaned.drop(columns=["Label"])

    # One-Hot encode categoricals, handle infinities and NaN
    X = pd.get_dummies(X)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    logger.info(f"Cleaned feature matrix: {X.shape[0]:,} × {X.shape[1]} features")

    # ===================================================================
    # STEP 2: FEATURE SELECTION (LIGHTWEIGHT ARCHITECTURE)
    # ===================================================================
    logger.info(f"Selecting Top {n_top_features} features via RandomForest importance...")

    rf = RandomForestClassifier(
        n_estimators=50, n_jobs=-1, random_state=random_state
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(n_top_features).index.tolist()

    logger.info("Selected features:")
    for i, feat in enumerate(top_features, 1):
        score = importances[feat]
        logger.info(f"  {i:2d}. {feat:<35s} (importance: {score:.6f})")

    X_optimized = X[top_features]

    # ===================================================================
    # STEP 3: TRAIN/VALIDATION SPLIT
    # ===================================================================
    # We split BEFORE training so the threshold calibration uses unseen data.
    X_train, X_val, y_train, y_val = train_test_split(
        X_optimized, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    logger.info(
        f"Split: {X_train.shape[0]:,} train / {X_val.shape[0]:,} validation  "
        f"(Attack ratio in val: {y_val.mean():.2%})"
    )

    # ===================================================================
    # STEP 4: ISOLATION FOREST TRAINING (NORMAL TRAFFIC ONLY)
    # ===================================================================
    logger.info("Training Isolation Forest on normal traffic...")

    # Filter to normal traffic (Label == 0) for unsupervised training
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    # Fit scaler on normal traffic
    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)

    logger.info(f"  Normal training samples: {X_train_normal_scaled.shape[0]:,}")

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_normal_scaled)
    logger.info("  [✔] Isolation Forest fitted.")

    # ===================================================================
    # STEP 5: COST-SENSITIVE THRESHOLD CALIBRATION
    # ===================================================================
    logger.info(
        f"Calibrating cost-sensitive threshold "
        f"(C_FP={cost_fp}, C_FN={cost_fn})..."
    )

    # Score the validation set
    X_val_scaled = scaler.transform(X_val)
    val_scores = model.decision_function(X_val_scaled)

    # Primary calibration
    calibrator = CostSensitiveThreshold(cost_fp=cost_fp, cost_fn=cost_fn)
    optimal_threshold = calibrator.calibrate(val_scores, y_val.values)

    # Multi-ratio sensitivity analysis (for the IEEE paper)
    logger.info("Running sensitivity analysis across cost ratios...")
    sensitivity_results = calibrator.calibrate_multi_ratio(
        val_scores, y_val.values,
        ratios=[(1.0, 5.0), (1.0, 10.0), (1.0, 20.0)]
    )

    # Quick performance summary at the chosen threshold
    predicted_anomaly = val_scores < optimal_threshold
    actual_attack = y_val.values == 1

    tp = int(np.sum(predicted_anomaly & actual_attack))
    fp = int(np.sum(predicted_anomaly & ~actual_attack))
    fn = int(np.sum(~predicted_anomaly & actual_attack))
    tn = int(np.sum(~predicted_anomaly & ~actual_attack))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info(f"\n  === Validation Performance (threshold={optimal_threshold:.6f}) ===")
    logger.info(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")

    # ===================================================================
    # STEP 6: EXPORT ARTIFACTS
    # ===================================================================
    logger.info("\nExporting artifacts...")

    artifacts = {
        "isolation_forest_model.pkl": model,
        "scaler.pkl": scaler,
        "top_features.pkl": top_features,
        "optimal_threshold.pkl": optimal_threshold,
    }

    for name, obj in artifacts.items():
        path = output_path / name
        joblib.dump(obj, path)
        logger.info(f"  [✔] {name}")

    # Export sensitivity analysis as JSON (for paper graphs)
    sensitivity_path = output_path / "threshold_sensitivity.json"
    sensitivity_data = {
        "cost_ratios": sensitivity_results,
        "validation_metrics": {
            "threshold": optimal_threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        },
    }
    with open(sensitivity_path, "w") as f:
        json.dump(sensitivity_data, f, indent=2)
    logger.info(f"  [✔] threshold_sensitivity.json")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"Optimal Threshold: {optimal_threshold:.6f}")
    logger.info(f"Artifacts saved to: {output_path.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    train_nids_pipeline()
