import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "model_metadata.json"


def load_metadata():
    with METADATA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_artifacts(metadata):
    model_path = BASE_DIR / metadata["model_file"]
    scaler_path = BASE_DIR / metadata["scaler_file"]

    with model_path.open("rb") as fh:
        model = pickle.load(fh)
    with scaler_path.open("rb") as fh:
        scaler = pickle.load(fh)

    return model, scaler


def build_feature_frame(records, feature_names):
    frame = pd.DataFrame(records)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    frame = frame[feature_names].copy()
    for column in feature_names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame.isna().any().any():
        bad_columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Non-numeric or empty values found in: {bad_columns}")

    return frame


def predict_records(records):
    metadata = load_metadata()
    feature_names = metadata["feature_names"]
    threshold = float(metadata["threshold"])
    model, scaler = load_artifacts(metadata)

    frame = build_feature_frame(records, feature_names)
    scaled = scaler.transform(frame)
    scores = model.decision_function(scaled)

    results = []
    for score in scores:
        prediction = 1 if score <= threshold else 0
        results.append(
            {
                "score": float(score),
                "threshold": threshold,
                "prediction": prediction,
                "label": "anomaly" if prediction == 1 else "benign",
            }
        )
    return results


def read_json_records(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("JSON input must be an object or a list of objects.")


def read_csv_records(path):
    return pd.read_csv(path).to_dict(orient="records")


def self_test():
    metadata = load_metadata()
    model, scaler = load_artifacts(metadata)
    feature_names = metadata["feature_names"]

    if getattr(model, "n_features_in_", None) != metadata["feature_count"]:
        raise RuntimeError("Model feature count does not match metadata.")
    if getattr(scaler, "n_features_in_", None) != metadata["feature_count"]:
        raise RuntimeError("Scaler feature count does not match metadata.")

    scaler_features = list(getattr(scaler, "feature_names_in_", []))
    if scaler_features != feature_names:
        raise RuntimeError("Scaler feature order does not match metadata.")

    sample = [{name: 0 for name in feature_names}]
    result = predict_records(sample)[0]

    print("Model loaded successfully")
    print("Scaler loaded successfully")
    print(f"Experiment: {metadata['experiment_id']}")
    print(f"Seed: {metadata['seed']}")
    print(f"Feature count: {metadata['feature_count']}")
    print(f"Self-test score: {result['score']}")
    print(f"Self-test label: {result['label']}")
    print("Prediction test passed")


def main():
    parser = argparse.ArgumentParser(description="Run NIDS anomaly inference with the seed 52 model.")
    parser.add_argument("--json", help="Path to a JSON object or list of objects.")
    parser.add_argument("--csv", help="Path to a CSV file containing the required features.")
    parser.add_argument("--self-test", action="store_true", help="Load artifacts and run a synthetic prediction.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.json:
        records = read_json_records(args.json)
    elif args.csv:
        records = read_csv_records(args.csv)
    else:
        parser.error("Use --self-test, --json INPUT.json, or --csv INPUT.csv.")

    print(json.dumps(predict_records(records), indent=2))


if __name__ == "__main__":
    main()
