import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from predict import load_metadata, predict_records


BASE_DIR = Path(__file__).resolve().parent
EVENTS_PATH = BASE_DIR / "events.jsonl"
EXPLAINABILITY_DIR = BASE_DIR / "explainability"
FEATURE_IMPORTANCE_PATH = EXPLAINABILITY_DIR / "feature_importance.csv"

app = FastAPI(
    title="Explainable Lightweight NIDS API",
    description="Inference API for the seed 52 Isolation Forest NIDS model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Network-flow feature object matching model_metadata.json feature_names.",
    )


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        description="List of network-flow feature objects.",
    )


@app.get("/health")
def health():
    metadata = load_metadata()
    return {
        "status": "ok",
        "experiment_id": metadata["experiment_id"],
        "seed": metadata["seed"],
        "feature_count": metadata["feature_count"],
    }


@app.get("/metadata")
def metadata():
    return load_metadata()


def read_events(limit=100):
    if not EVENTS_PATH.exists():
        return []
    lines = EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def read_feature_importance(limit=10):
    if not FEATURE_IMPORTANCE_PATH.exists():
        return []
    rows = []
    lines = FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        try:
            rows.append({"feature": parts[0], "mean_abs_shap": float(parts[1])})
        except ValueError:
            continue
    return rows[:limit]


@app.get("/events")
def events(limit: int = 100):
    return {
        "count": len(read_events(limit)),
        "events": read_events(limit),
    }


@app.get("/explainability/feature-importance")
def feature_importance(limit: int = 10):
    return {
        "count": len(read_feature_importance(limit)),
        "features": read_feature_importance(limit),
    }


@app.get("/explainability/assets/{filename}")
def explainability_asset(filename: str):
    allowed = {
        "shap_summary.png",
        "shap_feature_importance.png",
        "feature_importance.csv",
    }
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Asset not found")
    path = EXPLAINABILITY_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    metadata = load_metadata()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NIDS Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --line: #334155;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --ok: #22c55e;
      --bad: #ef4444;
      --warn: #f59e0b;
      --blue: #38bdf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, sans-serif;
    }}
    header {{
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{ font-size: 20px; margin: 0; }}
    main {{ padding: 20px 24px 28px; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .card {{ padding: 14px; min-height: 82px; }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .bad {{ color: var(--bad); }}
    .ok {{ color: var(--ok); }}
    .warn {{ color: var(--warn); }}
    .blue {{ color: var(--blue); }}
    .panels {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 14px;
    }}
    .explainability {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .panel {{ padding: 14px; overflow: hidden; }}
    h2 {{ font-size: 15px; margin: 0 0 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .timeline {{
      height: 140px;
      display: flex;
      align-items: end;
      gap: 4px;
      border-left: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 8px;
    }}
    .bar {{
      width: 10px;
      min-height: 2px;
      background: var(--blue);
      border-radius: 2px 2px 0 0;
    }}
    .bar.anomaly {{ background: var(--bad); }}
    .empty {{ color: var(--muted); padding: 12px 0; }}
    .explainability img {{
      width: 100%;
      max-height: 420px;
      object-fit: contain;
      background: #020617;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .note {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: repeat(2, 1fr); }}
      .panels {{ grid-template-columns: 1fr; }}
      .explainability {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Explainable Lightweight NIDS Dashboard</h1>
      <div class="meta">Experiment {metadata["experiment_id"]} | Seed {metadata["seed"]} | Feature count {metadata["feature_count"]}</div>
    </div>
    <div class="meta" id="updated">Waiting for events</div>
  </header>
  <main>
    <section class="grid">
      <div class="card"><div class="label">Total Events</div><div class="value" id="total">0</div></div>
      <div class="card"><div class="label">Anomalies</div><div class="value bad" id="anomalies">0</div></div>
      <div class="card"><div class="label">Benign</div><div class="value ok" id="benign">0</div></div>
      <div class="card"><div class="label">Last Decision</div><div class="value" id="lastDecision">-</div></div>
      <div class="card"><div class="label">Last Score</div><div class="value blue" id="lastScore">-</div></div>
    </section>
    <section class="panels">
      <div class="panel">
        <h2>Recent Events</h2>
        <div style="overflow:auto">
          <table>
            <thead>
              <tr><th>Time</th><th>Label</th><th>Source</th><th>Destination</th><th>Port</th><th>Packets</th><th>Bytes</th><th>Score</th></tr>
            </thead>
            <tbody id="eventsBody"></tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <h2>Score Timeline</h2>
        <div class="timeline" id="timeline"></div>
        <h2 style="margin-top:18px">Top Sources</h2>
        <table><tbody id="topSources"></tbody></table>
      </div>
    </section>
    <section class="explainability">
      <div class="panel">
        <h2>Global SHAP Summary</h2>
        <img src="/explainability/assets/shap_summary.png" alt="Global SHAP summary">
        <p class="note">Global SHAP analysis explains the model's overall behavior across evaluation samples. It is separate from per-event local XAI.</p>
      </div>
      <div class="panel">
        <h2>SHAP Feature Importance</h2>
        <img src="/explainability/assets/shap_feature_importance.png" alt="SHAP feature importance">
        <h2 style="margin-top:18px">Top Global Features</h2>
        <table>
          <thead><tr><th>Feature</th><th>Mean |SHAP|</th></tr></thead>
          <tbody id="featureImportance"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    function fmtScore(value) {{
      return Number(value).toFixed(4);
    }}
    function cls(label) {{
      return label === "anomaly" ? "bad" : "ok";
    }}
    async function refresh() {{
      const response = await fetch("/events?limit=100");
      const payload = await response.json();
      const events = payload.events || [];
      const anomalies = events.filter(e => e.label === "anomaly").length;
      const benign = events.filter(e => e.label === "benign").length;
      const last = events[events.length - 1];

      document.getElementById("total").textContent = events.length;
      document.getElementById("anomalies").textContent = anomalies;
      document.getElementById("benign").textContent = benign;
      document.getElementById("lastDecision").textContent = last ? last.label.toUpperCase() : "-";
      document.getElementById("lastDecision").className = "value " + (last ? cls(last.label) : "");
      document.getElementById("lastScore").textContent = last ? fmtScore(last.score) : "-";
      document.getElementById("updated").textContent = "Updated " + new Date().toLocaleTimeString();

      const body = document.getElementById("eventsBody");
      body.innerHTML = "";
      if (!events.length) {{
        body.innerHTML = "<tr><td colspan='8' class='empty'>No events yet</td></tr>";
      }}
      events.slice(-30).reverse().forEach(event => {{
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${{event.timestamp || ""}}</td>
          <td class="${{cls(event.label)}}">${{(event.label || "").toUpperCase()}}</td>
          <td>${{event.src || ""}}</td>
          <td>${{event.dst || ""}}</td>
          <td>${{event.dport || 0}}</td>
          <td>${{event.packets || 0}}</td>
          <td>${{event.bytes || 0}}</td>
          <td>${{fmtScore(event.score || 0)}}</td>
        `;
        body.appendChild(row);
      }});

      const timeline = document.getElementById("timeline");
      timeline.innerHTML = "";
      events.slice(-40).forEach(event => {{
        const bar = document.createElement("div");
        const scoreMagnitude = Math.min(Math.abs(Number(event.score || 0)) * 420, 120);
        bar.className = "bar " + (event.label === "anomaly" ? "anomaly" : "");
        bar.style.height = Math.max(4, scoreMagnitude) + "px";
        bar.title = `${{event.label}} ${{fmtScore(event.score || 0)}}`;
        timeline.appendChild(bar);
      }});

      const counts = {{}};
      events.forEach(event => {{
        counts[event.src] = (counts[event.src] || 0) + 1;
      }});
      const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 5);
      const topBody = document.getElementById("topSources");
      topBody.innerHTML = top.length ? "" : "<tr><td class='empty'>No sources yet</td></tr>";
      top.forEach(([src, count]) => {{
        const row = document.createElement("tr");
        row.innerHTML = `<td>${{src}}</td><td>${{count}} events</td>`;
        topBody.appendChild(row);
      }});
    }}
    async function refreshFeatureImportance() {{
      const response = await fetch("/explainability/feature-importance?limit=10");
      const payload = await response.json();
      const rows = payload.features || [];
      const body = document.getElementById("featureImportance");
      body.innerHTML = rows.length ? "" : "<tr><td colspan='2' class='empty'>No feature importance data</td></tr>";
      rows.forEach(item => {{
        const row = document.createElement("tr");
        row.innerHTML = `<td>${{item.feature}}</td><td>${{Number(item.mean_abs_shap).toFixed(4)}}</td>`;
        body.appendChild(row);
      }});
    }}
    refresh();
    refreshFeatureImportance();
    setInterval(refresh, 2000);
  </script>
</body>
</html>"""


@app.post("/predict")
def predict_one(request: PredictionRequest):
    try:
        result = predict_records([request.features])[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "input_type": "single",
        "result": result,
    }


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    try:
        results = predict_records(request.records)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "input_type": "batch",
        "count": len(results),
        "results": results,
    }
