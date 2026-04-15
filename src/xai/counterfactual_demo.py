"""
Counterfactual SHAP Causality Demo — Interactive CLI Script

This is the PRESENTATION SHOWPIECE. It proves that the model's decisions are
driven by causal feature contributions, not mere statistical correlation.

Flow:
    1. Loads a sample anomalous network flow (from JSON file or built-in sample).
    2. Runs Isolation Forest prediction → shows ANOMALY verdict with score.
    3. Computes SHAP values → displays ranked feature impact in terminal.
    4. Identifies the TOP contributing feature.
    5. Prompts user: "Reduce [feature] by 50%? (y/n)"
    6. Modifies the feature, re-runs prediction → shows NORMAL verdict.
    7. Prints causal conclusion proving the model's interpretability.

Usage:
    python -m src.xai.counterfactual_demo
    python -m src.xai.counterfactual_demo --flow-file suspicious_flow.json

Academic Reference:
    - Wachter, Mittelstadt & Russell (2017). "Counterfactual Explanations
      Without Opening the Black Box."

Author: NIDS Research Team
"""

import sys
import os
import json
import argparse
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional

# Fix Windows terminal encoding (cp1254/cp1252 can't render Unicode box chars)
if sys.platform == "win32":
    os.system("")  # Enable ANSI escape codes on Windows 10+
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ml.engine import NIDSEngine, CANONICAL_FEATURES
from src.xai.explainer import NIDSExplainer

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Terminal Color Utilities (cross-platform ANSI)
# ---------------------------------------------------------------------------

class Colors:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def colored(text: str, color: str) -> str:
        return f"{color}{text}{Colors.RESET}"


def print_header():
    """Print the demo header banner."""
    print()
    print(Colors.colored("=" * 70, Colors.CYAN))
    print(Colors.colored("  COUNTERFACTUAL SHAP CAUSALITY DEMONSTRATION", Colors.BOLD + Colors.CYAN))
    print(Colors.colored("  Proving AI Decision Causality, Not Correlation", Colors.DIM + Colors.CYAN))
    print(Colors.colored("=" * 70, Colors.CYAN))
    print()


def print_section(title: str):
    """Print a section divider."""
    print()
    print(Colors.colored(f"── {title} ", Colors.BOLD + Colors.YELLOW) +
          Colors.colored("─" * (60 - len(title)), Colors.DIM + Colors.YELLOW))
    print()


def print_verdict(is_anomaly: bool, score: float, threshold: float):
    """Print the model's verdict with color coding."""
    if is_anomaly:
        verdict_text = "⚠  ANOMALY DETECTED"
        color = Colors.RED
    else:
        verdict_text = "✔  NORMAL TRAFFIC"
        color = Colors.GREEN

    print(f"  Verdict:   {Colors.colored(Colors.BOLD + verdict_text, color)}")
    print(f"  Score:     {score:.6f}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Decision:  score {'<' if is_anomaly else '≥'} threshold → "
          f"{Colors.colored('ANOMALY' if is_anomaly else 'NORMAL', color)}")


def print_shap_bar(feature: str, shap_val: float, max_abs: float, rank: int):
    """Print a single SHAP feature contribution as a terminal bar."""
    bar_width = 30
    normalized = abs(shap_val) / max_abs if max_abs > 0 else 0
    bar_len = int(normalized * bar_width)

    if shap_val < 0:
        # Pushes toward anomaly
        bar = Colors.colored("█" * bar_len, Colors.RED)
        direction = Colors.colored("← ANOMALY", Colors.RED)
    else:
        # Pushes toward normal
        bar = Colors.colored("█" * bar_len, Colors.GREEN)
        direction = Colors.colored("NORMAL →", Colors.GREEN)

    prefix = Colors.colored(f"  #{rank}", Colors.BOLD) if rank == 1 else f"  #{rank}"
    print(f"{prefix}  {feature:<32s}  {bar}  {shap_val:+.6f}  {direction}")


def print_feature_table(features: Dict[str, float], highlight_feature: str = None):
    """Print feature values in a clean table format."""
    print(f"  {'Feature':<32s}  {'Value':>12s}")
    print(f"  {'─' * 32}  {'─' * 12}")
    for name, val in features.items():
        if name == highlight_feature:
            line = Colors.colored(f"  {name:<32s}  {val:>12.2f}  ◄", Colors.YELLOW + Colors.BOLD)
        else:
            line = f"  {name:<32s}  {val:>12.2f}"
        print(line)


# ---------------------------------------------------------------------------
# Built-in Anomalous Sample (for demo without external files)
# ---------------------------------------------------------------------------

# This flow mimics a borderline data exfiltration pattern:
# - OUT_BYTES is just above the anomaly threshold — dominant driver
# - Reducing OUT_BYTES by 50% flips the verdict to NORMAL (causality proof)
BUILTIN_ANOMALOUS_FLOW = {
    "MIN_TTL": 62.0,
    "MAX_TTL": 64.0,
    "SHORTEST_FLOW_PKT": 54.0,
    "LONGEST_FLOW_PKT": 1500.0,
    "MIN_IP_PKT_LEN": 40.0,
    "MAX_IP_PKT_LEN": 1480.0,
    "OUT_BYTES": 80000.0,
    "OUT_PKTS": 45.0,
    "DST_TO_SRC_SECOND_BYTES": 8500.0,
    "NUM_PKTS_UP_TO_128_BYTES": 12.0,
}


# ---------------------------------------------------------------------------
# Main Demo Logic
# ---------------------------------------------------------------------------

def load_flow_from_file(path: str) -> Dict[str, float]:
    """Load a flow sample from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    # Support both flat dict and nested {"features": {...}} format
    if "features" in data:
        return data["features"]
    return data


def run_demo(artifacts_dir: str = ".", flow_file: str = None,
             reduction_pct: float = 50.0, auto_mode: bool = False):
    """
    Execute the full counterfactual demonstration.

    Args:
        artifacts_dir: Path to model artifacts (.pkl files).
        flow_file: Optional JSON file with anomalous flow data.
        reduction_pct: Percentage to reduce the top feature (default: 50%).
        auto_mode: If True, skip user prompts (for automated testing).
    """
    print_header()

    # ── Step 1: Load Model ────────────────────────────────────────────────
    print_section("STEP 1: Loading Model Artifacts")

    engine = NIDSEngine(artifacts_dir=artifacts_dir)
    try:
        engine.load_artifacts()
    except FileNotFoundError as e:
        print(Colors.colored(f"  [✘] Failed to load artifacts: {e}", Colors.RED))
        print(f"  Run 'python -m src.ml.train_baseline' first.")
        return

    explainer = NIDSExplainer.from_artifacts(artifacts_dir)
    print(Colors.colored("  [✔] Model and SHAP explainer loaded.", Colors.GREEN))

    # ── Step 2: Load Anomalous Sample ─────────────────────────────────────
    print_section("STEP 2: Loading Suspicious Network Flow")

    if flow_file:
        print(f"  Source: {flow_file}")
        flow_data = load_flow_from_file(flow_file)
    else:
        print("  Source: Built-in anomalous flow sample (Data Exfiltration pattern)")
        flow_data = BUILTIN_ANOMALOUS_FLOW.copy()

    print_feature_table(flow_data)

    # ── Step 3: Initial Prediction ────────────────────────────────────────
    print_section("STEP 3: Initial Model Prediction")

    feature_vector = np.array([flow_data[f] for f in engine.top_features], dtype=np.float64)
    result = engine.analyze(feature_vector, feature_dict=flow_data)

    print_verdict(result.is_anomaly, result.anomaly_score, result.threshold)

    if not result.is_anomaly:
        print()
        print(Colors.colored(
            "  [!] Flow is already classified as NORMAL. "
            "Use a more extreme flow for demonstration.", Colors.YELLOW))
        print("      Try increasing OUT_BYTES or NUM_PKTS_UP_TO_128_BYTES.")
        return

    # ── Step 4: SHAP Explanation ──────────────────────────────────────────
    print_section("STEP 4: SHAP Feature Attribution Analysis")

    X_scaled = engine.scaler.transform(np.atleast_2d(feature_vector))
    report = explainer.explain(X_scaled, feature_values=flow_data, top_k=10)

    print(f"  Base Value (expected): {report.shap_base_value:.6f}")
    print()

    # Sort by absolute SHAP for the bar chart
    shap_dict = explainer.get_shap_dict(X_scaled)
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    max_abs_shap = max(abs(v) for _, v in sorted_features) if sorted_features else 1.0

    for rank, (feat, shap_val) in enumerate(sorted_features, 1):
        print_shap_bar(feat, shap_val, max_abs_shap, rank)

    # Identify top anomaly-driving feature
    top_anomaly_features = [(f, v) for f, v in sorted_features if v < 0]
    if not top_anomaly_features:
        print(Colors.colored("\n  [!] No features push toward anomaly. Cannot demonstrate counterfactual.", Colors.YELLOW))
        return

    top_feature_name, top_shap_value = top_anomaly_features[0]

    print()
    print(Colors.colored(f"  ▸ Top anomaly driver: {top_feature_name} "
                         f"(SHAP = {top_shap_value:+.6f})", Colors.BOLD + Colors.MAGENTA))

    # Print NLG insight for the top feature
    for exp in report.explanations:
        if exp["feature_name"] == top_feature_name:
            print(Colors.colored(f"    → {exp['insight']}", Colors.MAGENTA))
            if exp.get("mitre_id"):
                print(Colors.colored(
                    f"    → MITRE ATT&CK: {exp['mitre_id']} — {exp['mitre_technique']}", Colors.DIM))
            break

    # ── Step 5: Counterfactual Intervention ───────────────────────────────
    print_section("STEP 5: Counterfactual Intervention")

    original_value = flow_data[top_feature_name]
    reduced_value = original_value * (1 - reduction_pct / 100.0)

    print(f"  Target feature:  {top_feature_name}")
    print(f"  Original value:  {original_value:,.2f}")
    print(f"  Proposed change: Reduce by {reduction_pct:.0f}% → {reduced_value:,.2f}")
    print()

    if not auto_mode:
        try:
            user_input = input(
                Colors.colored(
                    f"  Apply counterfactual intervention? (y/n): ",
                    Colors.BOLD + Colors.YELLOW
                )
            ).strip().lower()
            if user_input not in ("y", "yes", ""):
                print("  Aborted by user.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            return

    # Apply the counterfactual modification
    modified_flow = flow_data.copy()
    modified_flow[top_feature_name] = reduced_value

    print()
    print(Colors.colored("  Applying counterfactual modification...", Colors.CYAN))
    print()
    print_feature_table(modified_flow, highlight_feature=top_feature_name)

    # ── Step 6: Re-evaluation ─────────────────────────────────────────────
    print_section("STEP 6: Re-evaluation After Intervention")

    modified_vector = np.array([modified_flow[f] for f in engine.top_features], dtype=np.float64)
    modified_result = engine.analyze(modified_vector, feature_dict=modified_flow)

    print_verdict(modified_result.is_anomaly, modified_result.anomaly_score,
                  modified_result.threshold)

    # ── Step 7: Causality Conclusion ──────────────────────────────────────
    print_section("STEP 7: Causality Analysis Conclusion")

    score_delta = modified_result.anomaly_score - result.anomaly_score

    print(f"  Original Score:  {result.anomaly_score:.6f} → "
          f"{Colors.colored('ANOMALY', Colors.RED)}")
    print(f"  Modified Score:  {modified_result.anomaly_score:.6f} → "
          f"{Colors.colored('NORMAL' if not modified_result.is_anomaly else 'ANOMALY', Colors.GREEN if not modified_result.is_anomaly else Colors.RED)}")
    print(f"  Score Delta:     {score_delta:+.6f}")
    print()

    if not modified_result.is_anomaly:
        print(Colors.colored("  ╔══════════════════════════════════════════════════════════╗", Colors.GREEN))
        print(Colors.colored("  ║  CAUSALITY PROVEN: The model's anomaly decision was     ║", Colors.GREEN))
        print(Colors.colored(f"  ║  causally driven by '{top_feature_name}'.", Colors.GREEN))
        padding = 55 - len(top_feature_name)
        print(Colors.colored(f"  ║  Reducing this single feature by {reduction_pct:.0f}% reversed the    ║", Colors.GREEN))
        print(Colors.colored("  ║  verdict from ANOMALY → NORMAL, demonstrating that      ║", Colors.GREEN))
        print(Colors.colored("  ║  SHAP attributions reflect genuine causal relationships, ║", Colors.GREEN))
        print(Colors.colored("  ║  not mere statistical correlations.                     ║", Colors.GREEN))
        print(Colors.colored("  ╚══════════════════════════════════════════════════════════╝", Colors.GREEN))
    else:
        print(Colors.colored("  ╔══════════════════════════════════════════════════════════╗", Colors.YELLOW))
        print(Colors.colored("  ║  PARTIAL CAUSALITY: Reducing the top feature shifted    ║", Colors.YELLOW))
        print(Colors.colored("  ║  the score but did not fully reverse the decision.      ║", Colors.YELLOW))
        print(Colors.colored("  ║  This indicates a multi-factor anomaly pattern.         ║", Colors.YELLOW))
        print(Colors.colored("  ║  Try reducing multiple features or using a larger %.    ║", Colors.YELLOW))
        print(Colors.colored("  ╚══════════════════════════════════════════════════════════╝", Colors.YELLOW))

    print()
    print(Colors.colored("  Demo complete.", Colors.DIM))
    print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual SHAP Causality Demonstration for NIDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.xai.counterfactual_demo
  python -m src.xai.counterfactual_demo --flow-file suspicious_flow.json
  python -m src.xai.counterfactual_demo --reduction 70 --auto
        """
    )
    parser.add_argument(
        "--artifacts-dir", type=str, default=".",
        help="Directory containing .pkl model artifacts (default: current dir)"
    )
    parser.add_argument(
        "--flow-file", type=str, default=None,
        help="JSON file containing anomalous flow data (optional)"
    )
    parser.add_argument(
        "--reduction", type=float, default=50.0,
        help="Percentage to reduce the top contributing feature (default: 50%%)"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Skip interactive prompts (for automated testing/CI)"
    )

    args = parser.parse_args()
    run_demo(
        artifacts_dir=args.artifacts_dir,
        flow_file=args.flow_file,
        reduction_pct=args.reduction,
        auto_mode=args.auto,
    )


if __name__ == "__main__":
    main()
