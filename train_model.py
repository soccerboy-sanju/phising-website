"""
Model Training & Evaluation Script for Phishing Website Detector
Trains and compares Logistic Regression, Decision Tree, and Random Forest models.
Saves the best performing model, scaler, and evaluation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Import local feature extraction module
from feature_extraction import extract_features, FEATURE_NAMES

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'phishing_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess_data():
    """Load dataset CSV and extract URL feature vectors."""
    print("==================================================")
    print(" 1. LOADING DATASET & EXTRACTING FEATURES ")
    print("==================================================")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"[*] Dataset loaded successfully. Total samples: {len(df)}")
    
    # Drop invalid rows
    df = df.dropna(subset=['url', 'label'])
    
    print("[*] Extracting URL features using local feature_extraction.py...")
    X_list = []
    y_list = []

    for idx, row in df.iterrows():
        url = str(row['url'])
        label = int(row['label'])
        features = extract_features(url)
        X_list.append(features)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"[*] Feature matrix shape: {X.shape}")
    print(f"[*] Target distribution: {np.bincount(y)} (0: Legitimate, 1: Phishing)")

    return X, y


def train_and_evaluate():
    """Train Logistic Regression, Decision Tree, and Random Forest models."""
    X, y = load_and_preprocess_data()

    print("\n==================================================")
    print(" 2. SPLITTING & PREPROCESSING DATA ")
    print("==================================================")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models definition
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    fitted_models = {}

    print("\n==================================================")
    print(" 3. TRAINING & EVALUATING ALGORITHMS ")
    print("==================================================")

    best_model_name = None
    best_f1 = -1.0

    for name, model in models.items():
        print(f"\n[*] Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]

        results[name] = {
            'accuracy': round(float(acc) * 100, 2),
            'precision': round(float(prec) * 100, 2),
            'recall': round(float(rec) * 100, 2),
            'f1_score': round(float(f1) * 100, 2),
            'confusion_matrix': cm
        }
        fitted_models[name] = model

        print(f"    - Accuracy  : {results[name]['accuracy']}%")
        print(f"    - Precision : {results[name]['precision']}%")
        print(f"    - Recall    : {results[name]['recall']}%")
        print(f"    - F1-Score  : {results[name]['f1_score']}%")
        print(f"    - Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

        # Favor Random Forest if scores tie
        if results[name]['f1_score'] > best_f1 or (results[name]['f1_score'] == best_f1 and name == 'Random Forest'):
            best_f1 = results[name]['f1_score']
            best_model_name = name

    print("\n==================================================")
    print(" 4. MODEL COMPARISON & SUMMARY ")
    print("==================================================")
    comparison_df = pd.DataFrame(results).T[['accuracy', 'precision', 'recall', 'f1_score']]
    print(comparison_df.to_string())

    print(f"\n[*] Selected Best Performing Model: {best_model_name} (F1-Score: {best_f1}%)")

    # Save Best Model & Scaler
    best_model = fitted_models[best_model_name]
    model_save_path = os.path.join(MODELS_DIR, 'phishing_model.pkl')
    scaler_save_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    metrics_save_path = os.path.join(MODELS_DIR, 'metrics.json')

    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    metrics_payload = {
        'best_model': best_model_name,
        'feature_names': FEATURE_NAMES,
        'models': results
    }

    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=4)

    print("\n==================================================")
    print(" 5. ARTIFACTS SAVED ")
    print("==================================================")
    print(f"[+] Model saved to    : {model_save_path}")
    print(f"[+] Scaler saved to   : {scaler_save_path}")
    print(f"[+] Metrics saved to  : {metrics_save_path}")
    print("\nTraining completed successfully! You can now run app.py.")


if __name__ == '__main__':
    train_and_evaluate()
