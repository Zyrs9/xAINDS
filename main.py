import pandas as pd
import numpy as np
import joblib
import shap
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import uvicorn

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# 1. INITIALIZATION & MODEL LOADING
app = FastAPI(
    title="AI-Driven NIDS Inference Engine",
    description="A FastAPI backend that detects network intrusions using Isolation Forest and explains results with SHAP."
)

# Global variables for models and config
model = None
scaler = None
top_features = None
explainer = None

@app.on_event("startup")
def load_artifacts():
    """Load the trained ML models and configuration on startup."""
    global model, scaler, top_features, explainer
    print("--- Initializing NIDS Inference Engine ---")
    try:
        # Load artifacts exported from Step 2
        model = joblib.load('isolation_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        top_features = joblib.load('top_features.pkl')
        
        # Initialize SHAP TreeExplainer
        # TreeExplainer is optimized for scikit-learn tree-based models like IsolationForest
        explainer = shap.TreeExplainer(model)
        
        print("[✔] Model artifacts loaded successfully.")
        print(f"[✔] Monitoring Top {len(top_features)} features.")
    except FileNotFoundError as e:
        print(f"[✘] CRITICAL ERROR: Could not find model artifacts. {e}")
        print("Please run 'train_baseline.py' first to generate .pkl files.")
    except Exception as e:
        print(f"[✘] UNEXPECTED STARTUP ERROR: {e}")

# 2. DATA VALIDATION (PYDANTIC)
class FlowData(BaseModel):
    # Expects a dictionary of feature names and their numeric values
    features: Dict[str, float]

# 3. API ENDPOINTS
@app.get("/")
def health_check():
    """Simple endpoint to verify the API is online."""
    return {
        "status": "Online",
        "model_loaded": model is not None,
        "features_count": len(top_features) if top_features else 0
    }

@app.post("/analyze")
def analyze_flow(data: FlowData):
    """
    Main inference endpoint. 
    Detects anomalies and generates SHAP explanations.
    """
    if model is None or scaler is None or top_features is None:
        raise HTTPException(status_code=500, detail="Model engine not initialized. Check server logs.")

    try:
        # 4. INFERENCE LOGIC
        # Convert incoming dictionary to DataFrame
        input_dict = data.features
        df_input = pd.DataFrame([input_dict])

        # Validation: Check if all required features are present
        missing_features = [f for f in top_features if f not in df_input.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required behavioral features for NIDS analysis: {missing_features}"
            )

        # Ensure column order exactly matches the training phase
        df_input = df_input[top_features]

        # Scale the input features
        X_scaled = scaler.transform(df_input)

        # Run Prediction (1: Normal, -1: Anomaly)
        prediction_label = int(model.predict(X_scaled)[0])
        status = "Anomaly" if prediction_label == -1 else "Normal"

        # Calculate Anomaly Score (Lower = More Anomalous)
        anomaly_score = float(model.decision_function(X_scaled)[0])

        # 5. EXPLAINABLE AI (SHAP) LOGIC
        # Calculate SHAP values for the current input
        # IsolationForest TreeExplainer returns values explaining the decision_function
        shap_values = explainer.shap_values(X_scaled)

        # Handle structural differences in SHAP output across versions
        # For IsolationForest, we expect a 2D array [sample, feature]
        if isinstance(shap_values, list):
            # In some cases SHAP returns a list (e.g. for multi-output)
            current_shap = shap_values[0].flatten()
        elif len(shap_values.shape) == 3:
            # If 3D, take specific sample
            current_shap = shap_values[0, :, 0]
        else:
            # Standard 2D output
            current_shap = shap_values.flatten()

        # Map SHAP values to feature names for readable output
        explanation_dict = {
            feature: float(shap_val) 
            for feature, shap_val in zip(top_features, current_shap)
        }

        # 6. RESPONSE FORMAT
        return {
            "prediction": prediction_label,
            "status": status,
            "anomaly_score": round(anomaly_score, 4),
            "explanation": explanation_dict
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[✘] INFERENCE ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during inference: {str(e)}")

# 7. SERVER EXECUTION
if __name__ == "__main__":
    # Note: Using 127.0.0.1 for local inference. Change to 0.0.0.0 for containerized deployment.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
