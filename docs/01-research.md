# Phase 1 — Research & System Design

**Project:** MediSmart — AI-Powered Smart Medical Inventory Management System with Deep Learning-Based Disease Prediction and Medication Recommendation

**Phase status:** Complete
**Date:** 2026-09-04

---

## 1. Objectives

The system is a single web application serving a pharmacy, built from three integrated subsystems:

| # | Subsystem | What it does |
|---|-----------|--------------|
| 1 | **Medical Inventory Management** | Full CRUD over medicines, batches, suppliers, purchases, sales, stock levels and users. This is the backbone application. |
| 2 | **Disease Prediction & Medication Recommendation** | A pharmacist enters a walk-in customer's age, vitals and symptoms. Deep learning models predict the likely condition with a confidence score, explain what the disease is, recommend a medication class, and name the specialist to consult. |
| 3 | **Expiry Risk Prediction** | A model forecasts each medicine's usage rate and flags batches likely to sit unsold until expiry, surfaced as a risk column in the inventory table. |

A conversational assistant sits across all three, answering questions about stock, diseases and medicines in natural language.

**Hard constraint:** every component must be free of cost — no paid APIs, no paid hosting, no licensed datasets.

---

## 2. Problem Statement

Independent pharmacies face two recurring losses:

1. **Expiry wastage.** Stock is ordered on intuition. Slow-moving medicines are discovered only once they have expired. Industry estimates put pharmacy expiry write-off at roughly 3-5% of inventory value — money that is simply thrown away.
2. **Unstructured triage.** Customers describe symptoms verbally at the counter. Advice quality depends entirely on which staff member is present, and nothing is recorded.

Conventional inventory software is **reactive** — it reports what already happened. This project makes it **predictive**: it forecasts what will expire before it does, and it structures the counter conversation into a recorded, model-assisted recommendation.

### 2.1 Scope boundary (stated deliberately)

This system is a **decision-support tool for a pharmacist**, not a diagnostic medical device and not a replacement for a doctor. Every prediction is a *suggestion with a confidence score*, every output carries a disclaimer, and every recommendation ends by naming a specialist to consult. Section 9 covers the safety design in full. This boundary must be stated in the final report and in the live demo — examiners will ask about it.

---

## 3. Review of Existing Approaches

| Approach | Typical accuracy reported | Limitation for our use case |
|---|---|---|
| Decision Tree / Random Forest on symptom vectors | 95-100% | Trained on clean synthetic data; ignores numeric vitals entirely |
| Naive Bayes symptom checkers | 90-95% | Assumes symptom independence, which is clinically false |
| CNN on symptom-disease relations | ~96% | Heavy for a tabular binary input; no vitals fusion |
| ARIMA / Prophet for pharmacy demand | Varies | Statistical only; no per-batch expiry reasoning |
| Commercial pharmacy software | — | Reactive expiry alerts (date-based only), no usage-rate intelligence |

**Gap we are filling:** existing symptom checkers use *either* symptoms *or* lab vitals. The worked example in the project brief needs **both** — Age 45, Glucose 180, BMI 29, BP 140/90 **plus** "increased thirst, frequent urination". Our fusion design (Section 5.1) handles exactly that, and no single public dataset does.

---

## 4. Dataset Survey — all four verified downloadable, no login

Every URL below was fetched and inspected during this phase. All return HTTP 200 without a Kaggle account, which keeps the project fully reproducible.

### 4.1 Disease-Symptom dataset *(primary, for Model 1a)*

- **Source:** `github.com/anujdutt9/Disease-Prediction-from-Symptoms` (originally Columbia DBMI)
- **Verified shape:** **4,920 rows x 133 columns** — 132 binary symptom flags + `prognosis` label
- **Classes:** **41 diseases**, perfectly balanced at **120 samples each**
- **Note:** the file ships with a trailing all-empty column that must be dropped

> ### Honest finding you must know about now, not at the viva
>
> This dataset is **synthetic and perfectly separable**. Every disease has exactly 120 identical-pattern rows and no label noise. A plain decision tree scores **100% accuracy** on it. If we report "our deep learning model achieved 100%", any examiner will correctly call it meaningless.
>
> **Our mitigation, applied in Phase 2-3:**
> 1. Add realistic symptom noise (random flips) and missing-symptom dropout, simulating a patient who forgets to mention things.
> 2. Evaluate with stratified k-fold cross-validation, not a single lucky split.
> 3. Report the honest ceiling and explain *why* it is high — this becomes a strength of the report, not a weakness.
> 4. Anchor real, hard performance numbers on the two genuine clinical datasets below, where 100% is impossible.

### 4.2 Pima Indians Diabetes *(for Model 1b — the vitals branch)*

- **Source:** `github.com/jbrownlee/Datasets` — real clinical data, NIDDK
- **Verified shape:** **768 rows x 9 columns**; target balance 500 negative / 268 positive
- **Columns:** Pregnancies, **Glucose**, **BloodPressure**, SkinThickness, Insulin, **BMI**, DiabetesPedigreeFunction, **Age**, Outcome
- **Why it matters:** it contains **exactly** the four fields from the project's worked example — Age, Glucose, BMI, BloodPressure. This dataset is what makes the brief's example reproducible.
- **Known data quality issue:** zeros are used as missing values in Glucose, BloodPressure, SkinThickness, Insulin and BMI. Phase 2 must impute these — a biological BMI of 0 is impossible.

### 4.3 UCI Cleveland Heart Disease *(for Model 1c)*

- **Source:** `github.com/kb22/Heart-Disease-Prediction`
- **Verified shape:** **303 rows x 14 columns**; target balance 165 / 138 — genuinely balanced and genuinely hard
- **Columns:** age, sex, cp (chest pain type), trestbps (resting BP), chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
- **Realistic accuracy ceiling:** ~85%. This is where we demonstrate honest model performance.

### 4.4 Pharmacy Point-of-Sale Sales *(for Model 2 — expiry risk)*

- **Source:** `github.com/mcallara/pharma-sales-data` (mirror of the Milan Zdravkovic dataset)
- **Verified shape:** **2,106 daily rows x 13 columns**, spanning **2014-01 to 2019-09**
- **Content:** real POS data from a single pharmacy, ~600,000 transactions aggregated into **8 ATC drug groups**: `M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06`
- **Also available:** hourly, weekly and monthly aggregations
- **Why it matters:** this is **real** consumption data with real seasonality (e.g. R03 respiratory medicines spike in winter). Expiry risk modelled on real demand is far more defensible than modelling on invented numbers.

### 4.5 Knowledge files for the Treatment Database *(Phase 4)*

- `symptom_Description.csv` (11 KB) — plain-language description of each of the 41 diseases
- `symptom_precaution.csv` (3.5 KB) — four recommended precautions per disease
- `Symptom-severity.csv` (2.3 KB) — severity weight per symptom, useful for triage urgency

### 4.6 What we must build ourselves

No free dataset maps **disease -> specific medication -> specialist doctor**. Phase 4 therefore builds a **curated Treatment Database** covering all 41 diseases, with each entry citing a public clinical source. This is original project work and a genuine contribution, not a gap.

---

## 5. AI/ML Design

### 5.1 Model 1 — Disease Prediction (fusion architecture)

The core design decision of this project. Because no dataset carries both symptoms and lab vitals, we train **three specialist networks** and fuse them.

```
   SYMPTOMS (132 binary)          VITALS (age, glucose, BMI, BP, ...)
          |                              |            |
          v                              v            v
   +--------------+          +------------------+  +------------------+
   | Model 1a     |          | Model 1b         |  | Model 1c         |
   | Symptom DNN  |          | Diabetes Risk    |  | Heart Risk       |
   | 41-class     |          | DNN (Pima)       |  | DNN (Cleveland)  |
   | softmax      |          | sigmoid          |  | sigmoid          |
   +--------------+          +------------------+  +------------------+
          |                              |            |
          |  posterior over 41           | P(diabetes)| P(heart disease)
          v                              v            v
        +-------------------------------------------------+
        |          FUSION / DECISION LAYER                 |
        |  weighted evidence combination + calibration     |
        +-------------------------------------------------+
                              |
                              v
             Ranked diagnosis + calibrated confidence
                              |
                              v
        Treatment DB lookup -> medication + doctor + precautions
```

**Model 1a — Symptom classifier (Keras)**
`Input(132) -> Dense(256, ReLU) -> BatchNorm -> Dropout(0.4) -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(41, softmax)`, trained with categorical cross-entropy, Adam, early stopping. Trained on **noise-augmented** data per Section 4.1.

**Models 1b / 1c — Vitals risk networks (Keras)**
`Input(n) -> Dense(64) -> Dropout -> Dense(32) -> Dense(1, sigmoid)`, on standard-scaled features after median imputation of the zero-coded missing values.

**Fusion layer.** Combines the symptom posterior with the vitals risk scores. Where a vitals model and the symptom model agree on a condition, confidence is reinforced; where they disagree, the system reports both candidates rather than hiding the conflict. Confidence is calibrated so that "91%" means something close to 91% of such predictions being correct — an uncalibrated softmax score is not a probability, and claiming otherwise in a medical context is misleading.

**Worked example traced through the design** (the brief's own example):
Glucose 180 (well above the 140 threshold) + BMI 29 + BP 140/90 + Age 45 drives Model 1b high; "increased thirst, frequent urination" are classic polydipsia and polyuria symptoms driving Model 1a toward Diabetes; the two agree, fusion reinforces, output is **Diabetes with high confidence** -> Treatment DB returns medication class (e.g. Metformin as first-line, subject to prescriber approval), specialist **Endocrinologist**, and lifestyle precautions.

**Mandatory classical-ML comparison (scikit-learn).**
Per the technology stack, each deep model is benchmarked against Logistic Regression, Naive Bayes, SVM, Random Forest and Gradient Boosting, reporting accuracy, precision, recall, F1 and confusion matrices, with charts. We will report results honestly — **if a Random Forest beats the neural network on 303 rows of heart data, we say so and explain why** (deep learning needs volume; small tabular datasets favour tree ensembles). That analysis is worth more marks than a fabricated win.

### 5.2 Model 2 — Expiry Risk Prediction

Two stages:

**Stage A — Demand forecasting.** An **LSTM** over daily sales history per drug group predicts consumption for the coming N days, learning weekly and seasonal patterns. Benchmarked against moving average, linear regression and Random Forest baselines.

**Stage B — Per-batch risk scoring.** For each batch, using forecast demand:

- `days_to_expiry`
- `avg_daily_usage` (30/60/90-day windows) and usage trend
- `days_of_cover = quantity_on_hand / avg_daily_usage`
- `projected_units_remaining_at_expiry` from the LSTM forecast
- `value_at_risk = projected_remaining x unit_cost`

producing a **risk score 0-100** bucketed into `SAFE / WATCH / HIGH / CRITICAL`, plus a recommended action (discount, return to supplier, transfer, prioritise dispensing). This becomes the extra column in the inventory table.

**Training data note:** we have real *sales* history but no real *batch/expiry* records — no pharmacy publishes those. Phase 2 therefore generates a realistic batch ledger driven by the **real** consumption rates from the POS dataset, so the demand signal is genuine even though batch metadata is simulated. This limitation will be stated plainly in the report rather than glossed over.

### 5.3 Model 3 — Conversational Assistant

Per your decision: a **free-tier LLM API** (Groq or Google Gemini — both offer a genuinely free tier), given tools to query live inventory, the Treatment DB and prediction results.

**Risk acknowledged:** free tiers are rate-limited, require an API key, need internet, and can change terms. Because a graded demo must never fail live, the assistant will be built with a **provider-agnostic wrapper and an offline retrieval fallback** — if the key is missing, the quota is hit, or the network is down, it degrades to local intent matching over the same data instead of erroring out. You get natural conversation when online, and a working demo always.

---

## 6. System Architecture

```
                          BROWSER
              HTML + CSS + JavaScript (Jinja2 templates)
   Dashboard | Inventory | Expiry Risk | Diagnose | Chatbot | Reports
                              |
                        HTTP / JSON
                              |
                   FLASK APPLICATION (single server)
   +---------------------------------------------------------------+
   |  Auth (Flask-Login, role-based: Admin / Pharmacist / Staff)    |
   +---------------------------------------------------------------+
   |  Blueprints:  /inventory   /predict   /expiry   /chat   /api   |
   +---------------------------------------------------------------+
   |  Service layer: inventory, sales, prediction, expiry, chatbot  |
   +---------------------------------------------------------------+
   |  ML inference layer: loads .keras + .pkl from artifacts/models |
   +---------------------------------------------------------------+
   |  Data layer: SQLAlchemy ORM  ->  SQLite (MySQL-swappable)      |
   +---------------------------------------------------------------+
```

**Why one Flask server:** it serves both the HTML pages and the JSON API, so there is a single process to run, a single thing to deploy, and a single thing to explain in the report. Models load **once at startup**, not per request — loading a Keras model per request would make the page take seconds.

---

## 7. Database Design (outline — built in Phase 6)

| Table | Purpose |
|---|---|
| `users` | Login, role (admin/pharmacist/staff), password hash |
| `suppliers` | Supplier master with contact and lead time |
| `categories` | Drug category / ATC group |
| `medicines` | Medicine master: name, generic name, form, strength, price, reorder level, prescription-required flag |
| `batches` | **Per-batch** stock: batch number, quantity, cost, manufacture date, **expiry date** |
| `stock_transactions` | Immutable audit ledger of every stock movement |
| `sales` / `sale_items` | Counter sales, line items linked to the batch dispensed |
| `purchases` / `purchase_items` | Supplier purchase orders and receipts |
| `customers` | Optional customer record for repeat-purchase history |
| `consultations` | Every disease prediction: inputs, predicted disease, confidence, recommendation |
| `expiry_risk_scores` | Model output per batch: score, bucket, value at risk, computed timestamp |
| `chat_messages` | Assistant conversation log |
| `alerts` | Low stock, expiry, and risk notifications |

**Design decisions with reasons:**

- Expiry lives on **`batches`**, not on `medicines` — the same paracetamol arrives in multiple batches with different expiry dates, which is the entire basis of expiry prediction.
- Dispensing follows **FEFO** (First-Expiry-First-Out), not FIFO — the correct strategy for pharmacy and itself a wastage reducer.
- `stock_transactions` is append-only, so stock is always reconstructible and auditable.

---

## 8. Technology Decisions

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12.6 | Verified installed |
| Deep learning | TensorFlow 2.20 / Keras | Verified installed; simpler than PyTorch for this model shape |
| Classical ML | scikit-learn 1.8 | Required comparison study |
| Data | pandas 2.3.3, numpy 2.4 | Verified installed |
| Backend | **Flask 3.1.2** | Your choice. Serves pages + API in one process |
| Database | **SQLite via SQLAlchemy** | Your choice. Zero install, one file, deploys free; one config line switches to MySQL |
| Frontend | HTML, CSS, vanilla JS + Jinja2 | Per your stack; no build step, no npm |
| Charts | Plotly 7.0 + Matplotlib 3.10.8 | Plotly for interactive dashboard, Matplotlib for report figures |
| Chatbot | **Free-tier LLM API + offline fallback** | Your choice, with a safety net |
| Testing | pytest 9.1.1 | Phase 9 |
| Hosting | Render / PythonAnywhere / HF Spaces free tier | Confirmed free options |

**Total cost: zero.** All 16 dependencies are open-source and already installed and version-verified in your environment.

---

## 9. Safety, Ethics and Medical Disclaimer

Non-negotiable for a health-related project, and a section examiners look for.

1. **Decision support, never diagnosis.** Every output is phrased as a possibility with a confidence score.
2. **Persistent disclaimer** on every prediction screen and in every exported report.
3. **Always route to a human.** Every recommendation names the specialist to consult.
4. **Red-flag escalation.** Symptom combinations suggesting emergency (e.g. chest pain with breathlessness) bypass the normal flow and display an urgent "seek immediate medical attention" banner instead of a medication suggestion.
5. **Medication classes, not prescriptions.** The system suggests a class with standard adult context; it does not compute patient-specific doses, and it never recommends controlled substances.
6. **Prescription-required flag** enforced in inventory, so Rx-only items cannot be dispensed as OTC suggestions.
7. **Known bias, stated.** Pima is drawn from a single population (Pima Indian women) and Cleveland skews male; these models do not generalise equally to all patients. We will state this rather than imply universal validity.
8. **Full audit trail.** Every consultation is stored with inputs, output and timestamp.

---

## 10. Phase Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| **1. Research** | This document, verified datasets, project scaffold, pinned dependencies | **Done** |
| 2. Dataset | Download + cleaning + augmentation scripts; EDA notebook; processed datasets | Next |
| 3. ML | Trained Keras models + sklearn comparison + metrics and figures | |
| 4. Treatment DB | Curated 41-disease -> medication -> doctor knowledge base | |
| 5. Backend | Flask app, blueprints, auth, service layer | |
| 6. Database | SQLAlchemy models, migrations, seed data, FEFO logic | |
| 7. Frontend | HTML/CSS/JS pages, dashboard, inventory grid, diagnose form | |
| 8. Integration | Wire models into API; expiry risk column; chatbot | |
| 9. Testing | pytest suite, validation, error handling | |
| 10. Deployment | Free hosting, environment config | |
| 11. Documentation | Report, diagrams, screenshots, demo script | |

---

## 11. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Symptom dataset is trivially separable, giving a meaningless 100% accuracy | **High** | Noise augmentation + cross-validation + honest reporting (Section 4.1) |
| Free LLM tier rate-limited or unavailable during demo | **High** | Offline retrieval fallback built in from the start (Section 5.3) |
| No real batch/expiry data exists publicly | Medium | Simulate batches driven by *real* POS consumption rates; state the limitation |
| Heart dataset only 303 rows, so deep model may underperform trees | Medium | Report honestly; this is a legitimate finding, not a failure |
| Medical liability / over-trust by users | **High** | Disclaimers, red-flag escalation, doctor referral (Section 9) |
| Model load time slowing the web app | Low | Load models once at startup, cache in app context |
| Scope creep across 11 phases | Medium | Each phase ends in something runnable and demonstrable |

---

## 12. Success Criteria

**Functional**

- Full inventory CRUD with batch-level expiry tracking and FEFO dispensing
- Disease prediction reproducing the brief's worked example end to end
- Medication + disease explanation + specialist recommendation returned together
- Expiry risk column live in the inventory table with sortable risk buckets
- Working chatbot with graceful offline degradation
- Role-based login

**Model quality (targets, to be reported honestly whatever the outcome)**

- Symptom model: >95% cross-validated accuracy *under noise augmentation*
- Diabetes model: at least 75% accuracy and 0.80 ROC-AUC (competitive with published Pima results)
- Heart model: at least 82% accuracy
- Expiry/demand model: beat the moving-average baseline on MAE
- A complete DNN-vs-classical comparison table for every task

**Engineering**

- One command to install, one command to run
- Reproducible: scripts regenerate every dataset and model from scratch
- pytest suite passing
- Zero cost

---

## Sources

- Disease-symptom dataset — https://github.com/anujdutt9/Disease-Prediction-from-Symptoms
- Symptom description / precaution / severity — https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms
- Pima Indians Diabetes — https://github.com/jbrownlee/Datasets
- UCI Cleveland Heart Disease — https://github.com/kb22/Heart-Disease-Prediction
- Pharmacy POS sales data — https://github.com/mcallara/pharma-sales-data
- Original pharma sales source — https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data
