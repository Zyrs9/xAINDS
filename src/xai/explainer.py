"""
NIDS Explainable AI Module — SHAP + Natural Language Generation (NLG)

This module extends the base SHAP TreeExplainer with:
1. Human-readable Natural Language Generation (NLG) for SOC analysts.
2. MITRE ATT&CK technique mapping based on feature contribution patterns.
3. Structured explanation output for API consumption and terminal display.

Academic Reference:
    - Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions."
    - MITRE ATT&CK Framework: https://attack.mitre.org/

Author: NIDS Research Team
"""

import logging
import numpy as np
import shap
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger("nids.xai")

# ---------------------------------------------------------------------------
# NLG Rule Engine — Maps SHAP feature contributions to human language
# ---------------------------------------------------------------------------

# Each rule maps a feature name to interpretations based on the SHAP value's
# direction (positive SHAP = pushes toward "normal", negative = toward "anomaly"
# in Isolation Forest's decision_function space).
#
# For IsolationForest TreeExplainer:
#   - Negative SHAP value → feature pushes score DOWN → more anomalous
#   - Positive SHAP value → feature pushes score UP → more normal

NLG_RULES: Dict[str, Dict[str, dict]] = {
    "OUT_BYTES": {
        "anomalous": {
            "insight": "Abnormally high outbound data volume detected",
            "detail": "The volume of data leaving the network significantly exceeds the baseline profile. This pattern is consistent with bulk data transfer or staging.",
            "mitre_id": "T1048",
            "mitre_tactic": "Exfiltration",
            "mitre_technique": "Exfiltration Over Alternative Protocol",
            "severity": "HIGH",
        },
        "normal": {
            "insight": "Outbound data volume within expected range",
            "detail": "Egress traffic volume aligns with learned baseline behavior.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
    "OUT_PKTS": {
        "anomalous": {
            "insight": "Excessive outbound packet count",
            "detail": "High-frequency outbound packet bursts typically indicate scanning, C2 beaconing, or flood-based exfiltration.",
            "mitre_id": "T1071",
            "mitre_tactic": "Command and Control",
            "mitre_technique": "Application Layer Protocol",
            "severity": "HIGH",
        },
        "normal": {
            "insight": "Outbound packet rate within baseline",
            "detail": "Packet transmission rate matches normal operational patterns.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
    "DST_TO_SRC_SECOND_BYTES": {
        "anomalous": {
            "insight": "Anomalous reverse-direction data flow detected",
            "detail": "Unusually high data volume flowing from destination back to source. This may indicate data retrieval from a compromised asset or C2 response payload.",
            "mitre_id": "T1105",
            "mitre_tactic": "Command and Control",
            "mitre_technique": "Ingress Tool Transfer",
            "severity": "HIGH",
        },
        "normal": {
            "insight": "Reverse data flow within baseline",
            "detail": "Return traffic volume consistent with normal request-response patterns.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
    "NUM_PKTS_UP_TO_128_BYTES": {
        "anomalous": {
            "insight": "High volume of small packets detected",
            "detail": "Excessive small-packet traffic is a hallmark of DDoS amplification, SYN floods, or protocol-based reconnaissance scanning.",
            "mitre_id": "T1498",
            "mitre_tactic": "Impact",
            "mitre_technique": "Network Denial of Service",
            "severity": "CRITICAL",
        },
        "normal": {
            "insight": "Small packet ratio within baseline",
            "detail": "TCP ACK/control packet frequency is consistent with normal operations.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
    "MIN_TTL": {
        "anomalous": {
            "insight": "Anomalous minimum TTL value",
            "detail": "Unusual TTL floor may indicate spoofed packets, tunneling, or traffic from unexpected network distances.",
            "mitre_id": "T1090",
            "mitre_tactic": "Command and Control",
            "mitre_technique": "Proxy",
            "severity": "MEDIUM",
        },
        "normal": {
            "insight": "TTL minimum within expected range",
            "detail": "Minimum TTL value consistent with known network topology.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
    "MAX_TTL": {
        "anomalous": {
            "insight": "Anomalous maximum TTL value",
            "detail": "Elevated TTL ceiling may indicate forged packets or multi-hop evasion chains.",
            "mitre_id": "T1090.003",
            "mitre_tactic": "Command and Control",
            "mitre_technique": "Multi-hop Proxy",
            "severity": "MEDIUM",
        },
        "normal": {
            "insight": "TTL maximum within expected range",
            "detail": "Maximum TTL aligns with standard operating system defaults.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
    "SHORTEST_FLOW_PKT": {
        "anomalous": {
            "insight": "Anomalous minimum packet size in flow",
            "detail": "Unexpectedly small packets may indicate crafted probe packets or protocol manipulation attacks.",
            "mitre_id": "T1046",
            "mitre_tactic": "Discovery",
            "mitre_technique": "Network Service Scanning",
            "severity": "MEDIUM",
        },
        "normal": {
            "insight": "Minimum packet size within expected range",
            "detail": "Smallest packet size in flow is consistent with standard protocol headers.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
    "LONGEST_FLOW_PKT": {
        "anomalous": {
            "insight": "Anomalous maximum packet size in flow",
            "detail": "Oversized packets may indicate buffer overflow exploitation attempts or data smuggling via MTU manipulation.",
            "mitre_id": "T1190",
            "mitre_tactic": "Initial Access",
            "mitre_technique": "Exploit Public-Facing Application",
            "severity": "HIGH",
        },
        "normal": {
            "insight": "Maximum packet size within expected range",
            "detail": "Largest packet in flow is within standard MTU boundaries.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
    "MIN_IP_PKT_LEN": {
        "anomalous": {
            "insight": "Anomalous minimum IP packet length",
            "detail": "Minimal IP datagrams may indicate crafted packets used in OS fingerprinting or firewall probing.",
            "mitre_id": "T1018",
            "mitre_tactic": "Discovery",
            "mitre_technique": "Remote System Discovery",
            "severity": "MEDIUM",
        },
        "normal": {
            "insight": "Minimum IP packet length within baseline",
            "detail": "IP datagram sizes are consistent with standard protocol behavior.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
    "MAX_IP_PKT_LEN": {
        "anomalous": {
            "insight": "Anomalous maximum IP packet length",
            "detail": "Large IP datagrams may indicate payload smuggling, tunneling, or fragmentation evasion attacks.",
            "mitre_id": "T1572",
            "mitre_tactic": "Command and Control",
            "mitre_technique": "Protocol Tunneling",
            "severity": "HIGH",
        },
        "normal": {
            "insight": "Maximum IP packet length within baseline",
            "detail": "Largest IP datagram fits within expected protocol parameters.",
            "mitre_id": None,
            "mitre_tactic": None,
            "mitre_technique": None,
            "severity": "LOW",
        },
    },
}


@dataclass
class FeatureExplanation:
    """Structured explanation for a single feature's SHAP contribution."""
    feature_name: str
    feature_value: float
    shap_value: float
    contribution_direction: str  # "anomalous" or "normal"
    rank: int                    # 1 = most influential
    insight: str
    detail: str
    severity: str
    mitre_id: Optional[str] = None
    mitre_tactic: Optional[str] = None
    mitre_technique: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExplanationReport:
    """Complete SHAP + NLG explanation for one flow analysis."""
    shap_base_value: float
    total_features: int
    top_contributor: str
    explanations: List[dict]
    summary: str  # One-line human-readable summary

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# SHAP Explainer with NLG
# ---------------------------------------------------------------------------

class NIDSExplainer:
    """
    Wraps SHAP TreeExplainer with NLG mapping for SOC-ready outputs.

    Usage:
        explainer = NIDSExplainer.from_artifacts("./")
        report = explainer.explain(X_scaled, feature_dict)
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: Trained sklearn IsolationForest.
            feature_names: Ordered list of feature names matching training columns.
        """
        self.tree_explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
        logger.info(f"SHAP TreeExplainer initialized for {len(feature_names)} features.")

    @classmethod
    def from_artifacts(cls, artifacts_dir: str = ".") -> "NIDSExplainer":
        """Factory method to load model and features from disk."""
        base = Path(artifacts_dir)
        model = joblib.load(base / "isolation_forest_model.pkl")
        features = joblib.load(base / "top_features.pkl")
        return cls(model, features)

    def compute_shap_values(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute raw SHAP values for a scaled feature vector.

        Args:
            X_scaled: Scaled 2D numpy array (1, n_features).

        Returns:
            (shap_values_1d, base_value)
        """
        X_scaled = np.atleast_2d(X_scaled)
        shap_values = self.tree_explainer.shap_values(X_scaled)

        # Handle varying SHAP output shapes across versions
        if isinstance(shap_values, list):
            current_shap = shap_values[0].flatten()
        elif len(shap_values.shape) == 3:
            current_shap = shap_values[0, :, 0]
        else:
            current_shap = shap_values.flatten()

        ev = self.tree_explainer.expected_value
        if isinstance(ev, np.ndarray):
            base_value = float(ev.flat[0])
        else:
            base_value = float(ev)

        return current_shap, base_value

    def explain(self, X_scaled: np.ndarray,
                feature_values: Dict[str, float] = None,
                top_k: int = 5) -> ExplanationReport:
        """
        Generate a full SHAP + NLG explanation report.

        Args:
            X_scaled: Scaled feature vector (1, n_features).
            feature_values: Original (unscaled) feature values as {name: value}.
            top_k: Number of top contributors to include in the report.

        Returns:
            ExplanationReport with ranked, NLG-enriched feature explanations.
        """
        shap_values, base_value = self.compute_shap_values(X_scaled)

        # Rank features by absolute SHAP contribution
        abs_shap = np.abs(shap_values)
        ranked_indices = np.argsort(abs_shap)[::-1]

        explanations: List[FeatureExplanation] = []

        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            feat_name = self.feature_names[idx]
            shap_val = float(shap_values[idx])
            feat_val = float(feature_values.get(feat_name, 0.0)) if feature_values else 0.0

            # Negative SHAP → pushes toward anomaly
            direction = "anomalous" if shap_val < 0 else "normal"

            # Look up NLG rule
            rule = NLG_RULES.get(feat_name, {}).get(direction, {})

            explanation = FeatureExplanation(
                feature_name=feat_name,
                feature_value=round(feat_val, 4),
                shap_value=round(shap_val, 6),
                contribution_direction=direction,
                rank=rank,
                insight=rule.get("insight", f"{feat_name} contributed toward {direction}"),
                detail=rule.get("detail", "No detailed mapping available."),
                severity=rule.get("severity", "UNKNOWN"),
                mitre_id=rule.get("mitre_id"),
                mitre_tactic=rule.get("mitre_tactic"),
                mitre_technique=rule.get("mitre_technique"),
            )
            explanations.append(explanation)

        # Build one-line summary from the top contributor
        top = explanations[0] if explanations else None
        if top and top.contribution_direction == "anomalous":
            summary = (
                f"Primary anomaly driver: {top.feature_name} "
                f"(SHAP={top.shap_value:+.4f}) → {top.insight}"
            )
            if top.mitre_id:
                summary += f" [MITRE {top.mitre_id}]"
        else:
            summary = "Flow behavior consistent with baseline — no significant anomaly drivers."

        return ExplanationReport(
            shap_base_value=round(base_value, 6),
            total_features=len(self.feature_names),
            top_contributor=top.feature_name if top else "N/A",
            explanations=[e.to_dict() for e in explanations],
            summary=summary,
        )

    def get_shap_dict(self, X_scaled: np.ndarray) -> Dict[str, float]:
        """Simple {feature_name: shap_value} dictionary (backward compatible)."""
        shap_values, _ = self.compute_shap_values(X_scaled)
        return {
            name: round(float(val), 6)
            for name, val in zip(self.feature_names, shap_values)
        }
