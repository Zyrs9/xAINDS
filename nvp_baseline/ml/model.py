import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd

class AnomalyDetector:
    """
    ML model for anomaly detection using Isolation Forest.
    Fits a baseline synthetic 'normal' dataset to provide context for SHAP.
    """
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            n_estimators=100, 
            contamination=contamination, 
            random_state=42
        )
        self.is_fitted = False

    def train_synthetic(self, n_samples=1000):
        """
        Generates Gaussian distributed synthetic 'normal' network flow behavior 
        to train the model initially.
        Features: [packets_per_second, avg_payload_size, flow_duration, byte_rate]
        """
        # Normal behavior (e.g., browsing, SSH, etc.)
        # Lower packets/sec, moderate payload, some duration
        normal_data = np.random.normal(
            loc=[50, 250, 10, 12500],   # Means: ~50 pps, 250 avg payload, 10 duration, 12.5 KB/s
            scale=[10, 50, 2, 2500],     # Variances
            size=(n_samples, 4)
        )
        
        # Ensure non-negative
        normal_data = np.clip(normal_data, 0, None)
        
        # Fit the model
        self.model.fit(normal_data)
        self.is_fitted = True
        print(f"Model trained on {n_samples} synthetic normal flow samples.")

    def predict(self, X):
        """
        Predicts if the input sample is an anomaly.
        X: numpy array or list of features [packets_per_second, avg_payload_size, duration, byte_rate]
        Returns: label (1: normal, -1: anomaly) and anomaly score.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction.")
        
        # Ensure 2D array
        X = np.atleast_2d(X)
        
        # Isolation Forest prediction: 1 for normal, -1 for anomaly
        label = self.model.predict(X)[0]
        # In sklearn, lower scores are more abnormal. We use decision_function.
        # decision_function returns float in [offset_low, offset_high]
        score = self.model.decision_function(X)[0]
        
        return int(label), float(score)

    def get_model(self):
        return self.model
