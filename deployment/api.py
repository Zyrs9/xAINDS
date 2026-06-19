import json
import os
import textwrap
from datetime import datetime
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
REPORTS_DIR = BASE_DIR / "reports"
ENV_PATH = BASE_DIR / ".env"

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ENV_PATH)

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


class InvestigationRequest(BaseModel):
    event: dict[str, Any] = Field(
        ...,
        description="Selected dashboard event from events.jsonl.",
    )
    related_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent events used for historical correlation.",
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


def protocol_name(value):
    mapping = {1: "ICMP", 6: "TCP", 17: "UDP"}
    try:
        return mapping.get(int(value), str(value or "UNKNOWN"))
    except (TypeError, ValueError):
        return str(value or "UNKNOWN")


def risk_level(event, metadata):
    threshold = float(event.get("threshold", metadata.get("threshold", -0.054220406182326486)))
    score = float(event.get("score", 0.0))
    if str(event.get("label", "")).lower() == "anomaly" or score <= threshold:
        return "ANOMALY"
    if score <= threshold + abs(threshold) * 0.65:
        return "SUSPICIOUS"
    return "BENIGN"


def incident_id(event):
    seed = f"{event.get('timestamp', '')}-{event.get('src', '')}-{event.get('dst', '')}-{event.get('dport', '')}"
    value = abs(hash(seed)) % 1000000
    return f"AI-NIDS-{value:06d}"


def top_feature_drivers(event, limit=6):
    features = event.get("features") or {}
    preferred = [
        "Packets_per_Second",
        "Flow_Intensity",
        "Byte_Rate",
        "Total_Packets",
        "Total_Bytes",
        "Bytes_per_Packet",
        "FLOW_DURATION_MILLISECONDS",
    ]
    drivers = []
    for name in preferred:
        try:
            value = abs(float(features.get(name, 0)))
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            drivers.append({"feature": name, "value": value})
    if not drivers:
        drivers = read_feature_importance(limit)
        return [{"feature": row["feature"], "value": row["mean_abs_shap"]} for row in drivers]
    return sorted(drivers, key=lambda item: item["value"], reverse=True)[:limit]


def correlate_events(event, related_events):
    related = [
        item for item in related_events
        if item.get("src") == event.get("src") or item.get("dport") == event.get("dport")
    ]
    ports = sorted({str(item.get("dport")) for item in related if item.get("dport")})
    ips = sorted({str(item.get("src")) for item in related if item.get("src")})
    return {
        "similar_events": len(related),
        "last_occurrence": related[-1].get("timestamp") if related else None,
        "event_frequency": f"{len(related)} events in current dashboard window",
        "related_source_ips": ips[:8],
        "related_ports": ports[:8],
    }


def fallback_analysis(event, related_events, metadata):
    features = event.get("features") or {}
    risk = risk_level(event, metadata)
    proto = protocol_name(event.get("proto"))
    pps = float(features.get("Packets_per_Second", 0) or 0)
    byte_rate = float(features.get("Byte_Rate", 0) or 0)
    flow_intensity = float(features.get("Flow_Intensity", 0) or 0)
    attack_type = "Anomalous network behavior"
    if proto == "TCP" and pps > 50:
        attack_type = "Reconnaissance or brute-force activity"
    elif proto == "UDP" and pps > 50:
        attack_type = "UDP burst or service abuse"
    elif proto == "ICMP":
        attack_type = "ICMP burst or availability probing"
    confidence = "87%" if risk == "ANOMALY" else "64%" if risk == "SUSPICIOUS" else "38%"
    return {
        "root_cause_analysis": (
            f"Isolation Forest score {event.get('score')} was evaluated against threshold "
            f"{metadata.get('threshold')}. The flow shows {pps:.2f} packets per second, "
            f"{byte_rate:.2f} bytes per second, and flow intensity {flow_intensity:.2f}."
        ),
        "possible_attack_type": attack_type,
        "confidence_score": confidence,
        "threat_assessment": risk,
        "recommended_actions": [
            "Review source IP reputation and recent activity.",
            "Correlate with repeated attempts against the same destination or port.",
            "Preserve this event as evidence for analyst review.",
            "Apply temporary blocking only if repeated anomalous behavior continues.",
        ],
        "ai_investigation_summary": (
            "The event is assessed with model evidence, flow-derived drivers, and global SHAP context. "
            "Traffic rate and flow intensity are the strongest available explanatory signals for this incident."
        ),
        "shap_evidence": top_feature_drivers(event),
        "historical_correlation": correlate_events(event, related_events),
        "model_source": "local-fallback",
    }


def build_gemini_prompt(event, related_events, metadata, feature_importance):
    language = os.environ.get("LANGUAGE", "Turkish")
    prompt_payload = {
        "project": "Explainable AI-Driven Lightweight Network Intrusion Detection System (AI-NIDS)",
        "required_language": language,
        "model": {
            "type": metadata.get("model_type"),
            "threshold": metadata.get("threshold"),
            "prediction_rule": metadata.get("prediction_rule"),
            "metrics": metadata.get("metrics"),
        },
        "selected_event": event,
        "risk_level": risk_level(event, metadata),
        "top_feature_drivers": top_feature_drivers(event),
        "global_shap_feature_importance": feature_importance,
        "historical_correlation": correlate_events(event, related_events),
    }
    return (
        "You are a cybersecurity SOC investigation assistant for an academic AI-NIDS project. "
        "Analyze the selected network event using Isolation Forest score evidence, SHAP-style feature drivers, "
        "and historical correlation. Return ONLY valid JSON with these keys: "
        "root_cause_analysis, possible_attack_type, confidence_score, threat_assessment, "
        "recommended_actions, ai_investigation_summary, shap_evidence, historical_correlation. "
        "Use concise, professional Turkish suitable for an academic incident report.\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )


def call_gemini_analysis(event, related_events, metadata):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model_name = os.environ.get("MODEL_NAME", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed. Run: pip install -r deployment/requirements.txt") from exc

    prompt = build_gemini_prompt(event, related_events, metadata, read_feature_importance(10))
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", "") or ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = fallback_analysis(event, related_events, metadata)
        payload["ai_investigation_summary"] = text.strip() or payload["ai_investigation_summary"]
    payload["model_source"] = model_name
    return payload


def report_payload(event, related_events):
    metadata = load_metadata()
    analysis = call_gemini_analysis(event, related_events, metadata)
    if analysis is None:
        analysis = fallback_analysis(event, related_events, metadata)
    return {
        "incident_id": incident_id(event),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "risk_level": risk_level(event, metadata),
        "metadata": metadata,
        "analysis": analysis,
        "feature_drivers": top_feature_drivers(event),
        "feature_importance": read_feature_importance(10),
    }


def pdf_font_name():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "calibri.ttf",
    ]
    for path in font_candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont("AINIDSFont", str(path)))
                return "AINIDSFont"
            except Exception:
                continue
    return "Helvetica"


def add_wrapped_text(canvas, text, x, y, width, font_name, font_size=9, leading=12):
    canvas.setFont(font_name, font_size)
    max_chars = max(int(width / (font_size * 0.48)), 24)
    for paragraph in str(text or "").splitlines() or [""]:
        for line in textwrap.wrap(paragraph, max_chars) or [""]:
            if y < 56:
                canvas.showPage()
                y = 780
                canvas.setFont(font_name, font_size)
            canvas.drawString(x, y, line)
            y -= leading
    return y


def write_pdf_report(payload):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("reportlab is not installed. Run: pip install -r deployment/requirements.txt") from exc

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{payload['incident_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = REPORTS_DIR / filename
    c = canvas.Canvas(str(path), pagesize=A4)
    font_name = pdf_font_name()
    width, height = A4
    y = height - 48

    def section(title, body):
        nonlocal y
        if y < 110:
            c.showPage()
            y = height - 48
        c.setFont(font_name, 13)
        c.drawString(44, y, title)
        y -= 18
        y = add_wrapped_text(c, body, 54, y, width - 96, font_name)
        y -= 10

    event = payload["event"]
    analysis = payload["analysis"]
    metadata = payload["metadata"]
    c.setFont(font_name, 18)
    c.drawString(44, y, "AI-NIDS Incident Investigation Report")
    y -= 24
    c.setFont(font_name, 9)
    c.drawString(44, y, f"Incident ID: {payload['incident_id']} | Generated: {payload['generated_at']}")
    y -= 22

    section(
        "Incident Information",
        "\n".join([
            f"Timestamp: {event.get('timestamp', '-')}",
            f"Source IP: {event.get('src', '-')}",
            f"Destination IP: {event.get('dst', '-')}",
            f"Protocol: {protocol_name(event.get('proto'))}",
            f"Source Port: {event.get('sport', '-')}",
            f"Destination Port: {event.get('dport', '-')}",
        ]),
    )
    section(
        "Detection Details",
        "\n".join([
            f"Model: {metadata.get('model_type')} with {metadata.get('scaler_type')}",
            f"Anomaly Score: {event.get('score')}",
            f"Threshold: {metadata.get('threshold')}",
            f"Risk Level: {payload['risk_level']}",
            f"Prediction Rule: {metadata.get('prediction_rule')}",
        ]),
    )
    section("SHAP Evidence", json.dumps(payload["feature_drivers"], ensure_ascii=False, indent=2))
    section("AI Investigation Findings", analysis.get("ai_investigation_summary", ""))
    section("Root Cause Analysis", analysis.get("root_cause_analysis", ""))
    section(
        "Historical Correlation",
        json.dumps(analysis.get("historical_correlation", {}), ensure_ascii=False, indent=2),
    )
    section(
        "Recommendations",
        "\n".join(analysis.get("recommended_actions", [])) if isinstance(analysis.get("recommended_actions"), list) else analysis.get("recommended_actions", ""),
    )
    section("Analyst Notes", "Human analyst review area. Validate AI output before operational enforcement.")
    c.save()
    return filename, path


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


@app.post("/investigate/report")
def investigate_report(request: InvestigationRequest):
    try:
        payload = report_payload(request.event, request.related_events)
        filename, _ = write_pdf_report(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "incident_id": payload["incident_id"],
        "generated_at": payload["generated_at"],
        "risk_level": payload["risk_level"],
        "analysis": payload["analysis"],
        "pdf_url": f"/reports/{filename}",
        "pdf_filename": filename,
    }


@app.get("/reports/{filename}")
def report_file(filename: str):
    if "/" in filename or "\\" in filename or not filename.endswith(".pdf"):
        raise HTTPException(status_code=404, detail="Report not found")
    path = REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="application/pdf", filename=filename)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI-NIDS SOC Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0B1020;
      --panel: #111827;
      --line: #334155;
      --text: #F8FAFC;
      --muted: #CBD5E1;
      --ok: #22C55E;
      --bad: #EF4444;
      --warn: #F59E0B;
      --blue: #38bdf8;
      --soft: #172033;
      --shadow: rgba(0, 0, 0, 0.28);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #0B1020;
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }
    button {
      border: 0;
      border-radius: 6px;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      min-height: 34px;
      padding: 8px 12px;
    }
    .app-shell { min-height: 100vh; }
    .topbar {
      align-items: center;
      background: rgba(11, 16, 32, 0.96);
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 18px;
      min-height: 68px;
      padding: 0 24px;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .brand {
      align-items: center;
      display: flex;
      gap: 12px;
      min-width: 220px;
    }
    .logo-mark {
      align-items: center;
      background: #0F2745;
      border: 1px solid rgba(56, 189, 248, 0.55);
      border-radius: 8px;
      color: var(--blue);
      display: grid;
      font-weight: 800;
      height: 40px;
      letter-spacing: 0;
      place-items: center;
      width: 40px;
    }
    .brand-title { font-size: 18px; font-weight: 800; }
    .brand-subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .nav {
      align-items: center;
      display: flex;
      gap: 4px;
      overflow-x: auto;
    }
    .nav button {
      background: transparent;
      color: var(--muted);
      white-space: nowrap;
    }
    .nav button.active, .nav button:hover {
      background: #142136;
      color: var(--text);
    }
    main { padding: 22px 24px 30px; }
    .view { display: none; }
    .view.active { display: block; }
    .page-head {
      align-items: end;
      display: flex;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 26px; letter-spacing: 0; }
    h2 { font-size: 16px; margin-bottom: 12px; }
    h3 { font-size: 14px; margin-bottom: 10px; }
    .lede { color: var(--muted); font-size: 14px; margin-top: 6px; max-width: 760px; }
    .card, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 12px 28px var(--shadow);
    }
    .label { color: var(--muted); font-size: 12px; }
    .value { font-size: 25px; font-weight: 800; line-height: 1.1; }
    .trend { color: var(--ok); font-size: 12px; margin-top: 8px; }
    .timestamp { color: var(--muted); font-size: 11px; margin-top: 10px; }
    .soc-grid {
      display: block;
    }
    .panel { overflow: hidden; padding: 16px; }
    .panel-head {
      align-items: center;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      margin-bottom: 10px;
    }
    .section-note { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .table-wrap { overflow: auto; }
    table { border-collapse: collapse; font-size: 13px; width: 100%; }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 11px 8px;
      text-align: left;
      white-space: nowrap;
    }
    th { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
    tr:hover td { background: rgba(56, 189, 248, 0.04); }
    .badge {
      border-radius: 999px;
      display: inline-flex;
      font-size: 11px;
      font-weight: 800;
      justify-content: center;
      min-width: 92px;
      padding: 5px 9px;
    }
    .badge.benign { background: rgba(34, 197, 94, 0.12); color: var(--ok); }
    .badge.suspicious { background: rgba(245, 158, 11, 0.14); color: var(--warn); }
    .badge.anomaly { background: rgba(239, 68, 68, 0.14); color: var(--bad); }
    .btn-primary { background: var(--blue); color: #05101E; font-weight: 800; }
    .btn-muted { background: #1D293D; color: var(--muted); }
    .action-group { display: flex; gap: 8px; }
    .chart-metrics {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(3, 1fr);
      margin-bottom: 14px;
    }
    .mini-stat {
      background: #0F172A;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
    }
    .mini-stat strong { display: block; font-size: 17px; margin-top: 5px; }
    .detail-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .detail {
      background: #0F172A;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
    }
    .detail span { color: var(--muted); display: block; font-size: 11px; margin-bottom: 5px; }
    .detail strong { font-size: 14px; word-break: break-word; }
    .shap-assets {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 12px;
    }
    .shap-assets img {
      background: #020617;
      border: 1px solid var(--line);
      border-radius: 7px;
      max-height: 300px;
      object-fit: contain;
      width: 100%;
    }
    .report-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .report-section {
      background: #0F172A;
      border: 1px solid var(--line);
      border-radius: 7px;
      color: var(--muted);
      min-height: 80px;
      padding: 12px;
    }
    .report-section strong { color: var(--text); display: block; margin-bottom: 6px; }
    .report-link {
      display: inline-flex;
      margin-top: 12px;
      text-decoration: none;
    }
    .status-line {
      color: var(--muted);
      font-size: 12px;
      margin-top: 12px;
    }
    .empty { color: var(--muted); padding: 16px 0; }
    .bad { color: var(--bad); }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .blue { color: var(--blue); }
    @media (max-width: 1280px) {
      .soc-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      .topbar, .page-head { align-items: flex-start; flex-direction: column; }
      .detail-grid, .report-grid, .shap-assets { grid-template-columns: 1fr; }
      main { padding: 16px; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <div class="logo-mark">AI</div>
        <div>
          <div class="brand-title">AI-NIDS</div>
          <div class="brand-subtitle">Explainable AI-Driven Lightweight Network Intrusion Detection System</div>
        </div>
      </div>
      <nav class="nav" aria-label="Primary navigation">
        <button class="active" data-nav="dashboardView">Dashboard</button>
        <button data-nav="modelView">Model Information</button>
      </nav>
    </header>

    <main>
      <section class="view active" id="dashboardView">
        <section class="soc-grid">
          <div class="chart-metrics">
            <div class="mini-stat"><span class="label">Current Threshold</span><strong id="chartThreshold">--</strong></div>
            <div class="mini-stat"><span class="label">Current Average Score</span><strong id="chartAverage">--</strong></div>
            <div class="mini-stat"><span class="label">Highest Score</span><strong id="chartHighest">--</strong></div>
          </div>

          <div class="panel">
            <div class="panel-head">
              <div>
                <h2>Real-Time Alerts</h2>
                <p class="section-note">Analyst-ready event queue with risk scoring and investigation actions.</p>
              </div>
            </div>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr><th>Timestamp</th><th>Source IP</th><th>Destination IP</th><th>Protocol</th><th>Anomaly Score</th><th>Risk Level</th><th>Status</th><th>Actions</th></tr>
                </thead>
                <tbody id="alertsBody"></tbody>
              </table>
            </div>
          </div>
        </section>
      </section>

      <section class="view" id="reportsView">
        <div class="page-head">
          <div>
            <h1>PDF Report Preview</h1>
            <p class="lede">Professional incident report structure for academic evaluation and analyst handoff.</p>
          </div>
        </div>
        <div class="panel">
          <div class="report-grid">
            <div class="report-section"><strong>Incident Information</strong><span id="reportIncident">No incident selected.</span></div>
            <div class="report-section"><strong>Detection Details</strong><span id="reportDetection">Isolation Forest score and threshold evidence.</span></div>
            <div class="report-section"><strong>SHAP Evidence</strong><span>Top feature drivers and global explainability assets.</span></div>
            <div class="report-section"><strong>AI Investigation Findings</strong><span id="reportAi">Root cause, possible attack type, and confidence score.</span></div>
            <div class="report-section"><strong>Historical Correlation</strong><span id="reportHistory">Similar event frequency and related ports.</span></div>
            <div class="report-section"><strong>Recommendations</strong><span id="reportRecommendations">Analyst action guidance appears after investigation.</span></div>
            <div class="report-section"><strong>Analyst Notes</strong><span>Reserved for human review and academic traceability.</span></div>
          </div>
          <div class="status-line" id="reportStatus">No generated PDF yet.</div>
          <a class="btn-primary report-link" id="reportPdfLink" href="#" target="_blank" style="display:none">Export PDF</a>
        </div>
      </section>

      <section class="view" id="modelView">
        <div class="page-head"><div><h1>Model Information</h1><p class="lede">Isolation Forest artifact status, threshold policy, and evaluation metrics.</p></div></div>
        <div class="panel"><div class="detail-grid" id="modelDetails"></div></div>
      </section>

    </main>
  </div>

  <script>
    let allEvents = [];
    let metadata = {};
    let selectedEvent = null;
    let selectedReport = null;

    const protocolMap = {1: "ICMP", 6: "TCP", 17: "UDP"};

    function fmt(value, digits = 4) {
      const number = Number(value || 0);
      return Number.isFinite(number) ? number.toFixed(digits) : "--";
    }

    function riskLevel(event) {
      if (!event) return "BENIGN";
      const threshold = Number(metadata.threshold || event.threshold || -0.054220406182326486);
      const score = Number(event.score || 0);
      if ((event.label || "").toLowerCase() === "anomaly" || score <= threshold) return "ANOMALY";
      if (score <= threshold + Math.abs(threshold) * 0.65) return "SUSPICIOUS";
      return "BENIGN";
    }

    function badgeClass(risk) {
      return risk === "ANOMALY" ? "anomaly" : risk === "SUSPICIOUS" ? "suspicious" : "benign";
    }

    function showView(id) {
      document.querySelectorAll(".view").forEach(view => view.classList.remove("active"));
      document.getElementById(id).classList.add("active");
      document.querySelectorAll(".nav button").forEach(button => button.classList.toggle("active", button.dataset.nav === id));
    }

    document.querySelectorAll(".nav button").forEach(button => {
      button.addEventListener("click", () => showView(button.dataset.nav));
    });

    async function loadMetadata() {
      const response = await fetch("/metadata");
      metadata = await response.json();
      renderModelDetails();
    }

    async function refresh() {
      const response = await fetch("/events?limit=120");
      const payload = await response.json();
      allEvents = payload.events || [];
      renderDashboard();
      if (!selectedEvent && allEvents.length) setSelectedEvent(allEvents[allEvents.length - 1], false);
    }

    function renderDashboard() {
      const scores = allEvents.map(event => Number(event.score || 0));
      const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
      const highest = scores.length ? Math.min(...scores) : 0;
      document.getElementById("chartThreshold").textContent = fmt(metadata.threshold);
      document.getElementById("chartAverage").textContent = fmt(avg);
      document.getElementById("chartHighest").textContent = fmt(highest);

      const body = document.getElementById("alertsBody");
      body.innerHTML = "";
      if (!allEvents.length) {
        body.innerHTML = "<tr><td colspan='8' class='empty'>No alerts yet. Start tcpdump_monitor.py to stream classified flows.</td></tr>";
      }
      allEvents.slice(-35).reverse().forEach((event, index) => {
        const risk = riskLevel(event);
        const row = document.createElement("tr");
        const id = allEvents.length - 1 - index;
        row.innerHTML = `
          <td>${event.timestamp || ""}</td>
          <td>${event.src || ""}</td>
          <td>${event.dst || ""}</td>
          <td>${protocolMap[event.proto] || event.proto || "UNKNOWN"}</td>
          <td>${fmt(event.score)}</td>
          <td><span class="badge ${badgeClass(risk)}">${risk}</span></td>
          <td>${risk === "ANOMALY" ? "Open" : "Monitored"}</td>
          <td><div class="action-group"><button class="btn-primary" onclick="investigateEvent(${id})">Investigate & Export PDF Report</button></div></td>
        `;
        body.appendChild(row);
      });
    }

    function renderModelDetails() {
      const metrics = metadata.metrics || {};
      const details = [
        ["Model Type", metadata.model_type || "IsolationForest"],
        ["Scaler", metadata.scaler_type || "StandardScaler"],
        ["Decision Threshold", fmt(metadata.threshold)],
        ["Feature Count", metadata.feature_count || "--"],
        ["Recall", metrics.recall ? `${(metrics.recall * 100).toFixed(1)}%` : "--"],
        ["Precision", metrics.precision ? `${(metrics.precision * 100).toFixed(1)}%` : "--"],
        ["F1 Score", metrics.f1 ? `${(metrics.f1 * 100).toFixed(1)}%` : "--"],
        ["False Positive Rate", metrics.fpr ? `${(metrics.fpr * 100).toFixed(2)}%` : "--"]
      ];
      document.getElementById("modelDetails").innerHTML = details.map(([label, value]) => `<div class="detail"><span>${label}</span><strong>${value}</strong></div>`).join("");
    }

    function investigateEvent(index) {
      setSelectedEvent(allEvents[index]);
      showView("reportsView");
      generateInvestigationReport();
    }

    function setSelectedEvent(event) {
      selectedEvent = event || null;
      selectedReport = null;
      renderReportPreview();
      resetReportLinks();
    }

    function incidentId(event) {
      if (!event) return "AI-NIDS-0000";
      const seed = `${event.timestamp || ""}-${event.src || ""}-${event.dst || ""}-${event.dport || ""}`;
      let hash = 0;
      for (let i = 0; i < seed.length; i++) hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0;
      return `AI-NIDS-${Math.abs(hash).toString().slice(0, 6).padStart(6, "0")}`;
    }

    function renderReportPreview() {
      const event = selectedEvent;
      const risk = riskLevel(event);
      const features = event && event.features ? event.features : {};
      const packetsPerSecond = Number(features.Packets_per_Second || 0);
      const proto = event ? protocolMap[event.proto] || event.proto || "network" : "network";
      let attackType = "Anomalous network behavior";
      if (proto === "TCP" && packetsPerSecond > 50) attackType = "Reconnaissance or brute-force activity";
      if (proto === "UDP" && packetsPerSecond > 50) attackType = "UDP burst or service abuse";
      if (proto === "ICMP") attackType = "ICMP burst or availability probing";
      const confidence = risk === "ANOMALY" ? "87%" : risk === "SUSPICIOUS" ? "64%" : "38%";
      const related = event ? allEvents.filter(item => item.src === event.src || item.dport === event.dport) : [];
      const action = risk === "ANOMALY" ? "Review source activity, preserve evidence, and consider temporary blocking if repeated anomalous behavior continues." : risk === "SUSPICIOUS" ? "Monitor and correlate with repeated attempts." : "No immediate containment required.";
      document.getElementById("reportIncident").textContent = event ? `${incidentId(event)} | ${event.src} to ${event.dst} | ${event.timestamp}` : "No incident selected.";
      document.getElementById("reportDetection").textContent = event ? `Score ${fmt(event.score)} compared with threshold ${fmt(metadata.threshold)}. Risk level: ${risk}.` : "Isolation Forest score and threshold evidence.";
      document.getElementById("reportAi").textContent = event ? `${attackType} with confidence ${confidence}.` : "Root cause, possible attack type, and confidence score.";
      document.getElementById("reportHistory").textContent = event ? `${related.length} related events in current dashboard window.` : "Similar event frequency and related ports.";
      document.getElementById("reportRecommendations").textContent = event ? action : "Analyst action guidance appears after investigation.";
    }

    function resetReportLinks() {
      document.getElementById("reportStatus").textContent = "No generated PDF yet.";
      document.getElementById("reportPdfLink").style.display = "none";
    }

    function applyAiReport(report) {
      selectedReport = report;
      const analysis = report.analysis || {};
      const actions = Array.isArray(analysis.recommended_actions) ? analysis.recommended_actions.join(" ") : analysis.recommended_actions;
      document.getElementById("reportAi").textContent = analysis.root_cause_analysis || analysis.ai_investigation_summary || "AI investigation completed.";
      document.getElementById("reportRecommendations").textContent = actions || "Review the generated PDF report.";
      document.getElementById("reportStatus").textContent = `PDF ready: ${report.pdf_filename}`;
      document.getElementById("reportPdfLink").href = report.pdf_url;
      document.getElementById("reportPdfLink").style.display = "inline-flex";
    }

    async function generateInvestigationReport() {
      if (!selectedEvent) {
        document.getElementById("reportStatus").textContent = "Select an event before generating a report.";
        showView("dashboardView");
        return;
      }
      document.getElementById("reportStatus").textContent = "Generating PDF report...";
      try {
        const response = await fetch("/investigate/report", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            event: selectedEvent,
            related_events: allEvents
          })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Report generation failed");
        applyAiReport(payload);
      } catch (error) {
        document.getElementById("reportStatus").textContent = `Report generation failed: ${error.message}`;
      }
    }

    loadMetadata().then(refresh);
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
