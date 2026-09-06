# MediSmart ⚕️
### AI-Powered Smart Medical Inventory Management System with Deep Learning-Based Disease Prediction and Medication Recommendation

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![SQLite/MySQL](https://img.shields.io/badge/Database-SQLite%20%7C%20MySQL-blue.svg)](https://www.sqlite.org/)
[![Status](https://img.shields.io/badge/Phases%201--8-COMPLETED-brightgreen.svg)]()
[![Zero Running Cost](https://img.shields.io/badge/Running%20Cost-%240%20(Open%20Source)-success.svg)]()

---

## 📌 Executive Summary

**MediSmart** is an end-to-end intelligent hospital and pharmacy management platform engineered to tackle two critical healthcare bottlenecks simultaneously:
1. **Clinical Decision Support**: Accurate, deep learning-based disease screening that fuses quantitative clinical vitals (blood glucose, blood pressure, BMI) with patient-reported symptoms to predict conditions with calibrated confidence, advise first-line medication classes, and recommend specialist doctor referrals.
2. **Smart Inventory & Wastage Elimination**: Proactive First-Expiry-First-Out (FEFO) dispensing integrated with dual-head AI expiry risk prediction and 14-day LSTM therapeutic demand forecasting, eliminating pharmaceutical expiry write-offs.

A conversational AI assistant operates across all subsystems, providing instant offline answers on stock availability, near-expiry alerts, doctor referrals, and dispensing rules with **zero running cost**.

> ⚠️ **Medical Safety Disclaimer:** MediSmart is a clinical decision-support system designed for educational and pharmacy assistance. It is not an autonomous diagnostic medical device. Predictions include confidence intervals and always direct patients to qualified medical professionals.

---

## 🏛️ System Architecture

```
                                    ┌──────────────────────────────────────────┐
                                    │          Browser Web Interface           │
                                    │   Dashboard • Inventory • POS • AI UI    │
                                    └────────────────────┬─────────────────────┘
                                                         │ HTTP / REST
                                                         ▼
                                    ┌──────────────────────────────────────────┐
                                    │          Flask Application Core          │
                                    │    CORS • Flask-Login • SQLAlchemy ORM   │
                                    └────┬───────────────────┬───────────────┬─┘
                                         │                   │               │
                    ┌────────────────────┘                   │               └─────────────────────┐
                    ▼                                        ▼                                     ▼
       ┌─────────────────────────┐             ┌───────────────────────────┐             ┌───────────────────┐
       │    Inventory Service    │             │         AI Engine         │             │  Chatbot Service  │
       │  • FEFO Batch Deduction │             │  • 12x Keras DL Models    │             │  • Stock Queries  │
       │  • Dynamic Stock Audits │             │  • Multimodal Fusion      │             │  • Expiry Alerts  │
       │  • Supplier Intake      │             │  • Treatment Knowledge DB │             │  • Doctor Lookup  │
       └────────────┬────────────┘             └─────────────┬─────────────┘             └─────────┬─────────┘
                    │                                        │                                     │
                    ▼                                        ▼                                     ▼
       ┌─────────────────────────┐             ┌───────────────────────────┐             ┌───────────────────┐
       │   SQLite / MySQL DB     │             │ 41-Disease Clinical DB    │             │ Offline Knowledge │
       │  (data/medismart.db)    │             │ (WHO / NHS / Mayo Clinic) │             │ Graph (Zero Cost) │
       └─────────────────────────┘             └───────────────────────────┘             └───────────────────┘
```

---

## 🚀 Key Subsystems & Features

### 1. Multimodal Disease Prediction & Clinical Decision Support
- **Multimodal Clinical Fusion**: Evaluates both quantitative laboratory vitals and patient-reported symptoms.
- **131-Symptom Vocabulary**: Interactive multi-select tags with severity weighting (1–7) and emergency red-flag triggers (🚨).
- **Curated Treatment Database**: Covers all 41 diseases with medication classes, first-line dosages, and specialist doctor referrals.
- **Worked Diagnostic Example Verified (1-Click Test Button in UI)**:
  - **Input Vitals**: Age: `45` | Glucose: `180.0 mg/dL` | BMI: `29.0 kg/m²` | Blood Pressure: `140/90 mmHg`
  - **Input Symptoms**: `polyuria` (frequent urination), `excessive_hunger` (increased thirst/appetite), `fatigue`, `weight_loss`, `irregular_sugar_level`
  - **System Output**:
    - **Predicted Condition**: **Diabetes**
    - **Model Confidence**: **100.0%**
    - **Clinical Urgency**: `MODERATE`
    - **Recommended Medications**: Metformin, Sulfonylureas (Glimepiride), DPP-4 inhibitors, Insulin
    - **First-Line Regimen**: Metformin 500mg twice daily (start low, titrate up)
    - **Specialist Doctor**: **Endocrinologist / Diabetologist** (`Department of Endocrinology`)
    - **Precautions**: Balanced low-sugar diet, 150 min/week exercise, regular glucose monitoring

### 2. First-Expiry-First-Out (FEFO) Inventory & Point of Sale (POS)
- **Automatic FEFO Allocation**: The dispensing engine automatically consumes stock strictly from batches with the **earliest expiry date first**, eliminating dead stock.
- **Live FEFO Preview**: Cashiers see exactly which batches are decremented before confirming checkout.
- **Atomic Sales Ledger**: Generates itemized invoices, handles payment methods (Cash, UPI, Card), and logs audit adjustments.

### 3. AI Batch Expiry Risk Prediction (Model 2b)
- **Proactive Wastage Prevention**: Neural network evaluates batch cover ratio, days to expiry, and historical sales velocity.
- **Risk Classification**: Categorizes batches into `SAFE`, `WATCH`, `HIGH`, or `CRITICAL`.
- **Value-at-Risk (Rs)**: Computes financial exposure of unsold stock and issues automated directives (*"Apply 30% discount"*, *"Return to supplier"*).

### 4. 14-Day LSTM Demand Forecasting (Model 2a)
- **Deep Sequence Modeling**: Per-group sliding-window LSTMs forecast daily consumption across 8 ATC drug categories, accounting for seasonality (e.g. winter peaks in asthma inhalers, spring peaks in antihistamines).

### 5. Conversational AI Pharmacy Assistant (Chatbot)
- **Zero-Cost & Offline**: Grounded in the local database and clinical knowledge graph with zero external API fees.
- Answers queries like:
  - *"Do we have Paracetamol in stock?"* -> Real-time units on hand across active batches.
  - *"Which batches are near expiry?"* -> Batches expiring in ≤30 days with risk scores.
  - *"Which doctor should I consult for Diabetes?"* -> Specialist doctor and precautions.
  - *"Explain how FEFO dispensing works"* -> Educational inventory guidance.

---

## 📊 Development Phases & Current Status

| Phase | Milestone | Deliverable | Status |
|:---:|---|---|:---:|
| **1** | **Research & System Design** | System blueprint, dataset audit, safety framework, pinned requirements | **COMPLETE** ✅ |
| **2** | **Dataset Engineering** | 4 raw datasets downloaded, de-duplicated, 8-store 6-year simulation, EDA figures | **COMPLETE** ✅ |
| **3** | **Machine Learning Training** | 12 Keras models + scikit-learn benchmarks (Symptom, Diabetes, Heart, Expiry, 8x LSTMs) | **COMPLETE** ✅ |
| **4** | **Treatment Knowledge Base** | Curated clinical mapping for 41 diseases, 154 red-flag rules, specialist referrals | **COMPLETE** ✅ |
| **5** | **Flask Backend REST API** | REST API endpoints (`/api/auth`, `/api/inventory`, `/api/predict`, `/api/expiry`, `/api/dashboard`) | **COMPLETE** ✅ |
| **6** | **Database & ORM** | SQLAlchemy schema (`data/medismart.db`), seed script (66 batches, 22 medicines), FEFO engine | **COMPLETE** ✅ |
| **7** | **Frontend Web Application** | Responsive Jinja2 UI (`app.css`, `app.js`), Dashboard, POS, Expiry Grid, Diagnosis UI, AI Chatbot | **COMPLETE** ✅ |
| **8** | **System Integration & Output** | End-to-end integration demo (`scripts/integration_demo.py`), live server verification | **COMPLETE** ✅ |
| **9** | **Automated Testing Suite** | Pytest coverage, regression tests, edge-case assertions | *Next* |
| **10**| **Deployment Preparation** | Production Gunicorn, Docker containerization, cloud deployment guide | *Pending* |
| **11**| **Academic Documentation** | Final project report, defense presentation slides, viva defense guide | *Pending* |

---

## 🧠 Machine Learning Scorecard

| # | Model | Target | Architecture | Best Metric | Benchmark / Baseline |
|:---:|---|---|---|:---:|:---:|
| **1a** | **Symptom Classifier** | 41 Disease Classes | Deep Neural Net (256→128→41) | **94.69% Test Acc** (99.0% Top-3) | Beats 70.7% de-duplicated baseline |
| **1b** | **Diabetes Screening** | Binary Risk | Dense DNN (64→32→1) | **0.8435 ROC-AUC** | Outperforms Random Forest & Logistic Reg |
| **1c** | **Heart Disease Risk** | Binary Risk | Dense DNN (64→32→1) | **0.9053 ROC-AUC** | Competitive with RF (0.9096) on 302 rows |
| **2a** | **Demand Forecasting** | Daily Sales (8 Groups) | 60-Day Sliding Window LSTM | **Lowest MAE on 5/8 groups** | Beats 7-day and 30-day Moving Averages |
| **2b** | **Expiry Risk Model** | Wastage & Value-at-Risk | Dual-Head DNN (128→64→32) | **0.9811 ROC-AUC** | Dual classification + regression head |

All trained model weights (`.keras`), feature maps (`.json`), and scalers (`.pkl`) are saved in `artifacts/models/`.

---

## 💻 Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.12 | Industry standard for AI & Web backends |
| **Deep Learning** | TensorFlow / Keras 2.x | Neural networks, LSTM sequence modeling |
| **Machine Learning** | Scikit-learn | Multi-model benchmarking, preprocessing |
| **Data Processing** | Pandas, NumPy | High-performance tabular transformation |
| **Backend** | Flask | Lightweight, modular blueprint architecture |
| **Database** | SQLite / MySQL via SQLAlchemy | Zero-config local SQLite default, instant MySQL switch |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | Fast, responsive, dependency-free Jinja2 templates |
| **Data Visualization** | Chart.js | Interactive donut, bar, and demand line charts |
| **Authentication** | Flask-Login & Werkzeug | Role-based access control (Admin, Pharmacist, Cashier) |

---

## 🛠️ Quick Start & Installation

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/GJayamohan/AI-Powered-Smart-Medical-Inventory-Management-System.git
cd AI-Powered-Smart-Medical-Inventory-Management-System

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows
# source .venv/bin/activate  # On Linux / macOS
```

### 2. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Initialize & Seed Database
Initializes `data/medismart.db` and populates 3 default users, 4 suppliers, 22 catalog medicines, and 66 active batches:
```bash
python scripts/seed_db.py
```

### 4. Run Verification Suites
```bash
# Verify Backend APIs & FEFO engine
python scripts/test_backend.py

# Verify Frontend Routes & Chatbot queries
python scripts/test_frontend.py

# Run Complete End-to-End System Demonstration
python scripts/integration_demo.py
```

### 5. Launch Web Server
```bash
python wsgi.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 🔑 Demo Credentials

| Role | Username | Password | Access Privileges |
|---|---|---|---|
| **Admin** | `admin` | `admin123` | Full access, user management, financial audits |
| **Pharmacist** | `pharmacist` | `pharma123` | Inventory, stock intake, diagnosis, expiry alerts |
| **Cashier** | `cashier` | `cash123` | Point of sale, FEFO checkout, sales invoices |

*(The login page includes 1-click credential buttons for instant demonstration!)*

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|:---:|---|
| `/api/health` | `GET` | Health check & database connection status |
| `/api/auth/login` | `POST` | Authenticate user session |
| `/api/inventory/medicines` | `GET` | List all catalog medicines with real-time stock |
| `/api/inventory/batches` | `GET` | Filter batches by risk bucket, status, or medicine |
| `/api/inventory/fefo-preview` | `POST` | Calculate FEFO batch deduction plan for quantity |
| `/api/inventory/sale` | `POST` | Execute atomic sale transaction with FEFO deductions |
| `/api/predict/diagnose` | `POST` | Multimodal disease diagnosis from vitals + symptoms |
| `/api/predict/symptoms` | `GET` | List all 131 symptom strings with severity weights |
| `/api/expiry/analytics` | `GET` | Pharmacy wastage analytics & critical batch list |
| `/api/expiry/assess-batch/<id>` | `POST` | Run live AI risk model on a specific batch |
| `/api/expiry/forecast/<atc>` | `GET` | 14-day LSTM consumption forecast for drug category |
| `/api/chat/message` | `POST` | Conversational AI assistant query endpoint |
| `/api/dashboard/summary` | `GET` | High-level KPI metrics across all subsystems |

---

## 📁 Repository Structure

```
MMS/
├── artifacts/
│   ├── figures/             # 11 EDA and ML comparison plots
│   ├── models/              # 12 trained Keras models & scalers
│   └── reports/             # JSON evaluation benchmarks
├── data/
│   ├── knowledge/           # Curated 41-disease treatment database
│   ├── processed/           # Cleaned & augmented datasets
│   └── medismart.db         # Seeded SQLite database
├── docs/                    # Complete phase documentation (Phases 1-7)
├── medismart/
│   ├── api/                 # Flask REST blueprints (auth, inventory, predict, expiry, chat)
│   ├── chatbot/             # Conversational AI assistant engine
│   ├── db/                  # SQLAlchemy ORM models (User, Medicine, Batch, Sale, etc.)
│   ├── models/              # Neural network definitions & common training helpers
│   ├── services/            # Core business logic (FEFO Inventory & AI Engine)
│   ├── utils/               # Path configuration & environment checks
│   └── web/
│       ├── static/          # CSS design system & client JS
│       └── templates/       # Jinja2 HTML templates (Dashboard, POS, Predict, Expiry)
├── scripts/                 # Seed, train, verify, and integration demo scripts
├── requirements.txt         # Pinned open-source dependencies
└── wsgi.py                  # Server entry point
```

---

## 📜 Academic Attribution & License

Developed for academic presentation and major student project demonstration.
- **Repository**: [AI-Powered-Smart-Medical-Inventory-Management-System](https://github.com/GJayamohan/AI-Powered-Smart-Medical-Inventory-Management-System.git)
- **License**: MIT Open Source License. Free to use, adapt, and distribute.
