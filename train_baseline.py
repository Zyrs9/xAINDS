import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

def train_nids_baseline(file_path="NF-UNSW-NB15-v2.csv"):
    """
    Step 2: Training the unsupervised baseline using Isolation Forest.
    1. Loads and cleans NetFlow data.
    2. Performs feature selection (Top 10).
    3. Trains model only on normal traffic.
    4. Exports model, scaler, and feature list.
    """
    print(f"--- Loading dataset: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: Dataset '{file_path}' not found. Please ensure the file exists in the directory.")
        return

    # 1. DATA LOADING & CLEANING
    # Drop columns that lead to identity memorization instead of behavioral learning
    drop_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L2_IPV4_CLIENT_MAC', 'L2_IPV4_SERVER_MAC', 'id', 'Attack']
    # If any of these columns don't exist, ignore them
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df_cleaned = df.drop(columns=existing_drop_cols)
    
    # Separate the target 'Label' for training but keep it for filtering later
    y = df_cleaned['Label']
    X = df_cleaned.drop(columns=['Label'])

    # 2. PREPROCESSING
    print("--- Preprocessing traffic data ---")
    # Apply One-Hot Encoding to categorical variables
    X = pd.get_dummies(X)
    
    # Replace Infinity values with NaN and fill all NaN with 0
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # 3. FEATURE SELECTION (LIGHTWEIGHT ARCHITECTURE)
    print("--- Selecting Top 10 Most Important Behavioral Features ---")
    # Fit a RandomForest to identify feature importances
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    
    # Extract feature importances and get Top 10 names
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_10_features = importances.sort_values(ascending=False).head(10).index.tolist()
    
    print("\n[✔] Selected Top 10 Features:")
    for i, feature in enumerate(top_10_features, 1):
        print(f"  {i}. {feature}")
    
    # Create optimized dataframe with only selected features
    X_optimized = X[top_10_features]

    # 4. BASELINE TRAINING (ISOLATION FOREST - UNSUPERVISED)
    print("\n--- Training Isolation Forest Baseline (Normal Traffic Only) ---")
    # IMPORTANT: Isolation Forest learns from 'normal' (Label == 0) traffic
    normal_indices = df[df['Label'] == 0].index
    X_normal = X_optimized.loc[normal_indices]
    
    # Standardize the normal dataset
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    # Fit Isolation Forest
    # contamination=0.01 assumes 1% of the 'normal' training data might still be outliers
    model = IsolationForest(
        n_estimators=100, 
        contamination=0.01, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_normal_scaled)

    # 5. EXPORT ARTIFACTS
    print("\n--- Exporting Model Artifacts ---")
    joblib.dump(model, 'isolation_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(top_10_features, 'top_features.pkl')
    
    print("[✔] Success! Artifacts saved:")
    print("  - isolation_forest_model.pkl")
    print("  - scaler.pkl")
    print("  - top_features.pkl")
    print("Baseline training complete.")

if __name__ == "__main__":
    train_nids_baseline()
