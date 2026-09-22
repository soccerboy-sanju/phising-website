"""
Flask Backend Web Application for Phishing Website Detection
Serves the web dashboard interface and handles prediction API requests.
Analyzes URL characteristics locally without making outbound network calls.
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

# Import feature extraction module
from feature_extraction import (
    extract_features, 
    get_human_readable_features, 
    FEATURE_NAMES
)

app = Flask(__name__)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'phishing_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')

# Global variables for model and scaler
model = None
scaler = None


def load_ml_artifacts():
    """Load trained ML model and scaler if available."""
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("[+] ML model and scaler loaded successfully.")
        else:
            print("[!] Warning: Model files not found. Please run train_model.py first.")
    except Exception as e:
        print(f"[!] Error loading model artifacts: {str(e)}")


# Load model on startup
load_ml_artifacts()


@app.route('/')
def home():
    """Render the main Cybersecurity Dashboard UI."""
    return render_template('index.html')


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return model evaluation metrics generated during training."""
    if not os.path.exists(METRICS_PATH):
        return jsonify({
            'status': 'error',
            'message': 'Metrics not found. Please run train_model.py to train models and generate evaluation metrics.'
        }), 404

    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        return jsonify({
            'status': 'success',
            'data': metrics_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to load metrics: {str(e)}'
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether a given URL is LEGITIMATE or PHISHING.
    Expects JSON body: { "url": "https://example.com" }
    """
    global model, scaler
    
    # Reload artifacts if not previously loaded
    if model is None or scaler is None:
        load_ml_artifacts()
        if model is None or scaler is None:
            return jsonify({
                'status': 'error',
                'message': 'Trained ML Model not found. Please run "python train_model.py" in terminal first.'
            }), 500

    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Please enter a valid website URL.'
            }), 400

        raw_url = str(data['url']).strip()

        if not raw_url:
            return jsonify({
                'status': 'error',
                'message': 'Please enter a valid website URL.'
            }), 400

        # Perform local feature extraction
        feature_vector = extract_features(raw_url)
        human_features = get_human_readable_features(raw_url)

        # Scale features using pre-fitted scaler
        X = np.array([feature_vector])
        X_scaled = scaler.transform(X)

        # Predict class (0: Legitimate, 1: Phishing)
        prediction_code = int(model.predict(X_scaled)[0])
        
        # Calculate prediction probability/confidence score
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)[0]
            confidence = float(probabilities[prediction_code]) * 100
        else:
            confidence = 95.0

        confidence = round(confidence, 1)
        prediction_text = "PHISHING" if prediction_code == 1 else "LEGITIMATE"

        # Generate contextual explanation
        if prediction_code == 1:
            explanation = (
                "High Risk: This URL exhibits patterns typical of phishing attacks, "
                "such as suspicious keyword combinations, non-standard domain structure, "
                "or missing encryption."
            )
        else:
            explanation = (
                "Low Risk: This URL exhibits characteristics consistent with legitimate "
                "websites, including standard domain depth, valid structure, and clean parameter patterns."
            )

        return jsonify({
            'status': 'success',
            'url': raw_url,
            'prediction': prediction_text,
            'prediction_code': prediction_code,
            'confidence': confidence,
            'explanation': explanation,
            'features': human_features,
            'disclaimer': 'ML-based security assessment. Does not guarantee safety.'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error during prediction: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("\n==================================================")
    print(" STARTING FLASK PHISHING DETECTOR SERVER ")
    print("==================================================")
    print("[*] Running on http://127.0.0.1:5000 (Press CTRL+C to quit)")
    app.run(host='127.0.0.1', port=5000, debug=True)
