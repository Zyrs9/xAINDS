# NIDS Deployment Package

This folder contains the VM/demo inference package. It does not retrain the model.

## Files

- `models/model_seed_52.pkl`: selected Isolation Forest model.
- `models/scaler_seed_52.pkl`: matching StandardScaler for seed 52.
- `model_metadata.json`: threshold, feature order, and evaluation metrics.
- `predict.py`: CLI inference and self-test script.
- `api.py`: FastAPI inference service.
- `live_monitor.py`: live packet capture and flow classification script.
- `requirements.txt`: deployment dependencies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Health Check

```bash
python predict.py --self-test
```

## Example Prediction

```bash
python predict.py --json sample_input.json
```

## API

Start the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```text
http://SERVER_IP:8000/docs
```

Open the dashboard from another VM or the host browser:

```text
http://SERVER_IP:8000/dashboard
```

The dashboard includes a global explainability section with:

- SHAP summary plot.
- SHAP feature importance plot.
- Top global feature importance table.

These are global model interpretation artifacts, not per-event local XAI explanations.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Single prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": $(cat sample_input.json)}"
```

## Live Monitor

Find the gateway interface with:

```bash
ip -br a
```

Run the monitor on the attacker or victim side interface:

```bash
sudo .venv/bin/python live_monitor.py --interface eth1
```

Use JSON lines output when logs need to be parsed:

```bash
sudo .venv/bin/python live_monitor.py --interface eth1 --json
```

## tcpdump Monitor

Use this when Python raw socket capture is unreliable but `tcpdump` works:

```bash
sudo .venv/bin/python tcpdump_monitor.py --interface eth1 --flow-timeout 3 --min-packets 5
```

For quick validation with ping or hping3, lower the thresholds:

```bash
sudo .venv/bin/python tcpdump_monitor.py --interface eth1 --flow-timeout 1 --min-packets 1
```

For DDoS-style tests from Parrot to Debian, keep the monitor running on Ubuntu and generate traffic from Parrot:

```bash
hping3 -S -c 1000 -p 80 10.10.20.10
```

The tcpdump monitor groups DDoS-like traffic by source host, destination host, destination port, and protocol by default. Use `--no-aggregate-hosts` only when full 5-tuple flow separation is needed.

Every classified flow is appended to:

```text
events.jsonl
```

The dashboard reads this file through the `/events` endpoint.

## Prediction Rule

```text
anomaly if score <= -0.054220406182326486
```

## Notes

Use the seed 52 model and seed 52 scaler together. Do not pair `best_model.pkl` with `best_scaler.pkl`.
