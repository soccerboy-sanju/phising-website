# Phishing Website Detection Using Machine Learning

> **Computer Science & Engineering Mini-Project**  
> Built with Python, Flask, Scikit-learn, Pandas, HTML5/CSS3, JavaScript, and Chart.js.

---

## 📌 Problem Statement

Phishing is one of the most prevalent vector attacks in cybersecurity, where malicious actors mimic legitimate web applications to steal user credentials, financial information, and personal data. Traditional blacklist-based detection techniques struggle against newly generated phishing domains and dynamic URLs. 

This project solves the issue by leveraging **Machine Learning (ML)** to extract lexical and structural URL characteristics locally, classifying URLs as either **LEGITIMATE** or **PHISHING** without visiting or crawling dangerous target websites.

---

## 🎯 Objective

1. Extract 12 URL characteristics (length, dot count, subdomain count, hyphen count, special symbols, `@` symbol, IP routing, HTTPS encryption, numeric digits, query parameter count, path depth, and suspicious security keywords) locally.
2. Train, evaluate, and compare three machine learning classification algorithms:
   - **Logistic Regression**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
3. Select the highest-performing model based on F1-Score and Accuracy and serialize it using `joblib`.
4. Build a responsive, cybersecurity-themed web dashboard with Flask backend REST APIs and Chart.js analytics.

---

## 🛠️ Technology Stack

- **Backend:** Python 3.x, Flask (RESTful Web Application Framework)
- **Machine Learning & Data Processing:** Scikit-learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, CSS3 (Cyber Dark Theme), JavaScript (Fetch API / Async DOM updates)
- **Data Visualization:** Chart.js (Model metrics & performance visualization)
- **Environment:** 100% Local (Windows compatible, PyCharm ready, 0 external paid APIs)

---

## 🔬 Machine Learning Pipeline & Features

### Extracted URL Features
The system performs local string analysis on submitted URLs to extract 12 numerical features:
1. `url_length`: Total character count of the URL string.
2. `num_dots`: Count of `.` characters.
3. `num_subdomains`: Calculated subdomain depth from domain hostname.
4. `num_hyphens`: Count of `-` characters.
5. `num_special_chars`: Count of special characters (`@`, `?`, `=`, `%`, `!`, `&`, `#`, `+`).
6. `has_at_symbol`: Binary indicator (`1` if `@` is present).
7. `has_ip_address`: Binary indicator checking for IPv4 or IPv6 address routing in domain.
8. `use_https`: Binary indicator checking for `https://` protocol encryption.
9. `num_digits`: Total count of numeric digits (`0-9`).
10. `num_params`: Count of query parameter delimiters (`&` and `=`).
11. `url_depth`: Number of slash `/` path segments.
12. `has_suspicious_keyword`: Binary indicator detecting high-risk security keywords (`login`, `verify`, `bank`, `update`, `account`, `secure`, `webscr`, `cmd`, `signin`, `banking`, `paypal`, `ebay`, `amazon`, `credential`, `pay`, `security`, `wallet`).

---

## 📂 Project Structure

```text
ML/
│
├── app.py                     # Flask web application server & API endpoints
├── train_model.py             # ML dataset cleaner, feature extractor, trainer & evaluator
├── feature_extraction.py      # Local URL string feature extraction engine
├── requirements.txt           # Required Python packages
├── README.md                  # Project documentation & viva guide
│
├── dataset/
│   └── phishing_dataset.csv   # Dataset containing URLs and ground truth labels (0/1)
│
├── models/
│   ├── phishing_model.pkl     # Serialized best-performing trained ML model
│   ├── scaler.pkl             # Serialized StandardScaler normalization object
│   └── metrics.json           # Evaluation metrics & confusion matrices saved from training
│
├── templates/
│   └── index.html             # Web dashboard interface template
│
└── static/
    ├── style.css              # Cyberpunk dark mode styling & layout
    └── script.js              # Fetch API handler, Chart.js initializer & DOM manager
```

---

## 🚀 How to Run in PyCharm / Windows Terminal

### Step 1: Open Terminal in Project Folder
```cmd
cd ML
```

### Step 2: Install Required Packages
```cmd
pip install -r requirements.txt
```

### Step 3: Train & Compare Machine Learning Models
Execute `train_model.py` to process the dataset, train all three models, display metrics, and save the best model:
```cmd
python train_model.py
```

**Expected Output:**
- Tabular metric output for **Logistic Regression**, **Decision Tree**, and **Random Forest**.
- Best model saved to `models/phishing_model.pkl`.
- Scaler saved to `models/scaler.pkl`.
- Exact metric numbers saved to `models/metrics.json`.

### Step 4: Run Flask Web Server
Launch the Flask backend application:
```cmd
python app.py
```

### Step 5: Access Web Application
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 📊 Model Evaluation Metrics

When `train_model.py` is executed, the models are evaluated on test data using:
- **Accuracy:** Overall proportion of correctly identified URLs.
- **Precision:** Proportion of identified phishing URLs that were actually phishing.
- **Recall:** Proportion of actual phishing URLs correctly identified.
- **F1-Score:** Harmonic mean of Precision and Recall.
- **Confusion Matrix:** $2 \times 2$ matrix tracking True Negatives, False Positives, False Negatives, and True Positives.

The web dashboard automatically loads these empirical numbers from `models/metrics.json` to display the interactive Chart.js bar graph and confusion matrix grid.

---

## ⚠️ Limitations

1. **URL Structural Focus:** Analysis is strictly based on URL pattern characteristics. It does not inspect rendered DOM HTML, JavaScript behavior, or SSL certificate authority validity.
2. **Obfuscation Techniques:** Advanced shortened links (e.g. `bit.ly`) or redirection chains are evaluated as structural strings rather than expanded endpoints.

---

## 🔮 Future Enhancements

1. **WHOIS & Domain Age Lookup:** Incorporate domain registration age and registrar details via WHOIS API query.
2. **Browser Extension Integration:** Port the local feature extractor into a WebExtension for real-time URL scanning in Google Chrome / Firefox.
3. **Deep Learning Model (LSTM/1D-CNN):** Train character-level Deep Learning models to learn latent sequence representations of raw URLs.

---

## 💡 Conceptual Disclaimer

*This application is an automated Machine Learning security assessment tool created for academic demonstration. Predictions are probabilistic assessments based on feature extraction and do not constitute a 100% guarantee that a URL is safe or malicious.*
