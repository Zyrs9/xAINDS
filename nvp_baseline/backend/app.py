from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import time
import uvicorn

# Project imports (assumes they are in the PYTHONPATH)
from features.extractor import FlowFeatureExtractor
from ml.model import AnomalyDetector
from xai.explainer import SHAPExplainer
from capture.sniffer import LiveSniffer

# Global instances
detector = AnomalyDetector(contamination=0.1)
sniffer = LiveSniffer()
explainer = None # Initialized after training

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print("--- NIDS System Startup ---")
    
    # 1. Train model with synthetic 'normal' baseline
    detector.train_synthetic(n_samples=2000)
    
    # 2. Initialize SHAP explainer
    global explainer
    explainer = SHAPExplainer(detector.get_model(), FlowFeatureExtractor.get_feature_names())
    
    # 3. Start background sniffer
    sniffer.start()
    
    yield
    # --- Shutdown ---
    print("--- NIDS System Shutdown ---")
    sniffer.stop()

app = FastAPI(title="AI-Driven NIDS with Explainable AI", lifespan=lifespan)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_fitted": detector.is_fitted,
        "sniffer_running": sniffer.running
    }

@app.get("/analyze")
def analyze_last_flow():
    """
    Retrieves the most recent captured flow from the sniffer queue and analyzes it.
    """
    flow_packets = sniffer.get_next_flow(timeout=5)
    if not flow_packets:
        return {"message": "No new network flows captured recently. Try generating some traffic."}

    # Extract features
    features = FlowFeatureExtractor.extract(flow_packets)
    
    # Predict anomaly
    label, score = detector.predict(features)
    
    # Explain prediction
    explanation = explainer.explain(features)
    
    # Prepare result
    results = {
        "flow_metadata": {
            "packet_count": len(flow_packets),
            "timestamp": time.time()
        },
        "features": {name: float(val) for name, val in zip(FlowFeatureExtractor.get_feature_names(), features)},
        "score": score,
        "prediction": "Anomaly (-1)" if label == -1 else "Normal (1)",
        "explanation": explanation
    }
    
    return results

@app.post("/analyze")
def analyze_direct_input(data: dict):
    """
    Accepts manual feature data and returns detection + explanation.
    Expected format: {"packets_per_second": 100, "avg_payload_size": 1500, "flow_duration": 1.0, "byte_rate": 150000}
    """
    try:
        # Convert dictionary to numpy vector in correct order
        feature_names = FlowFeatureExtractor.get_feature_names()
        X = [data.get(name, 0.0) for name in feature_names]
        
        # Predict
        label, score = detector.predict(X)
        
        # Explain
        explanation = explainer.explain(X)
        
        return {
            "prediction": int(label),
            "score": float(score),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
