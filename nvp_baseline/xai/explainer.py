import shap
import pandas as pd
import numpy as np

class SHAPExplainer:
    """
    Explainer for the Isolation Forest model using SHAP.
    Maps numeric feature contributions back to human-readable feature names.
    For IsolationForest, TreeExplainer explains the decision_function result.
    Lower decision_function values mean 'more anomalous'.
    """
    def __init__(self, model, feature_names):
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names

    def explain(self, X):
        """
        Calculates SHAP values for a single flow sample.
        X: list or numpy array of features.
        Returns: A dictionary of {feature_name: shap_value}.
        """
        # Ensure 2D for SHAP
        X = np.atleast_2d(X)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # IsolationForest TreeExplainer returns a single set of SHAP values 
        # that sum to the decision_function(X) - expected_value.
        # shap_values[0] will be a 1D array of feature contributions for the sample.
        
        explanation = {}
        # shap_values can be 3D if multiple outcomes, but IsolationForest is usually 2D [sample, feature]
        current_shap = shap_values[0] if len(shap_values.shape) == 2 else shap_values[0][0]
        
        for i, name in enumerate(self.feature_names):
            explanation[name] = float(current_shap[i])
            
        return explanation
