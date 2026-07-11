import os
import numpy as np
import xgboost as xgb
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .feature_extractor import AudioFeatureExtractor

def start_training():
    extractor = AudioFeatureExtractor()
    X, y = [], []
    print("[INFO] Checking audio dataset via kagglehub...")
    try:
        download_path = kagglehub.dataset_download("radhaparikh/asvspoof2019-subset")
        print(f"[SUCCESS] Dataset located at: {download_path}")
        for root, dirs, files in os.walk(download_path):
            for file in files:
                if file.endswith(('.wav', '.flac')):
                    file_path = os.path.join(root, file)
                    is_spoof = "spoof" in root.lower() or "spoof" in file.lower()
                    features = extractor.extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(1 if is_spoof else 0)
    except Exception as e: print(f"[ERROR] Kagglehub failed: {e}")

    # CRITICAL FIX: Ensure both class 0 and class 1 exist before training
    unique_classes = np.unique(y)
    if len(X) == 0 or len(unique_classes) < 2:
        raise RuntimeError(
            f"ASVspoof subset unavailable or single-class (n={len(X)}). "
            "Refusing to train on fake data."
        )
    else:
        X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    print("[SUCCESS] XGBoost Audio Model Trained Successfully!")
    return model
