# MediSmart

**AI-Powered Smart Medical Inventory Management System with Deep Learning-Based Disease Prediction and Medication Recommendation**

A pharmacy management web application with three integrated AI subsystems:

1. **Inventory management** — medicines, batches, suppliers, purchases, sales, stock, users
2. **Disease prediction & medication recommendation** — enter a customer's age, vitals and symptoms; deep learning models predict the likely condition with a confidence score, explain the disease, recommend a medication class, and name the specialist to consult
3. **Expiry risk prediction** — forecasts usage rates and flags batches likely to expire unsold, shown as a risk column in the inventory table

A conversational assistant answers questions about stock, diseases and medicines across all three.

> **Medical disclaimer.** This is a decision-support tool for pharmacy staff, not a diagnostic medical device. Predictions are suggestions with confidence scores, never diagnoses. Always consult a qualified doctor.

---

## Technology

| Component | Technology |
|---|---|
| Programming | Python 3.12 |
| Deep Learning | TensorFlow / Keras |
| Data processing | Pandas, NumPy |
| ML comparison | Scikit-learn |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript (Jinja2) |
| Database | SQLite via SQLAlchemy (MySQL-swappable) |
| Visualization | Matplotlib, Plotly |

Every dependency is free and open-source. The project has **zero running cost**.

---

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify your machine can build and run the project:

```bash
python scripts/check_environment.py
```

This checks the Python version, every pinned dependency, live reachability of all four datasets, and that Keras can build and run a model.

Build every dataset from scratch (downloads ~1.5 MB, takes a few minutes):

```bash
python scripts/download_datasets.py && python scripts/prepare_datasets.py && python scripts/build_expiry_dataset.py && python scripts/eda.py
```

---

## Project structure

```
MMS/
├── data/
│   ├── raw/          # datasets downloaded by script (git-ignored)
│   ├── processed/    # cleaned / engineered datasets (git-ignored)
│   └── knowledge/    # curated treatment database (committed)
├── medismart/
│   ├── models/       # model definitions and training code
│   ├── api/          # Flask blueprints
│   ├── services/     # business logic
│   ├── db/           # SQLAlchemy models
│   ├── chatbot/      # assistant + offline fallback
│   ├── utils/
│   └── web/          # templates/ and static/
├── artifacts/
│   ├── models/       # trained .keras / .pkl files (git-ignored)
│   ├── figures/      # charts for the report
│   └── reports/      # metrics tables
├── notebooks/        # EDA
├── scripts/          # download, train, seed, verify
├── tests/            # pytest suite
└── docs/             # phase documentation
```

---

## Development phases

| Phase | Deliverable | Status |
|---|---|---|
| 1. Research | Dataset survey, system design, scaffold, pinned deps | **Done** |
| 2. Dataset | Download, clean, augment; expiry dataset; EDA | **Done** |
| 3. ML | Train Keras models + scikit-learn comparison | Next |
| 4. Treatment DB | 41 diseases to medication and specialist | |
| 5. Backend | Flask app, blueprints, auth | |
| 6. Database | SQLAlchemy models, seed data, FEFO | |
| 7. Frontend | Pages, dashboard, inventory grid | |
| 8. Integration | Wire models into the app; chatbot | |
| 9. Testing | pytest suite | |
| 10. Deployment | Free hosting | |
| 11. Documentation | Report, diagrams, demo script | |

Phase documentation lives in [`docs/`](docs/). Start with [Phase 1 — Research & System Design](docs/01-research.md).

---

## Datasets

All four are public and downloadable **without a Kaggle account**, which keeps the project fully reproducible.

| Dataset | Shape | Used for |
|---|---|---|
| Disease-Symptom | 4,920 x 133, 41 diseases | Symptom-based disease classifier |
| Pima Indians Diabetes | 768 x 9 | Vitals-based diabetes risk model |
| UCI Cleveland Heart Disease | 303 x 14 | Vitals-based heart risk model |
| Pharmacy POS sales | 2,106 daily rows, 8 drug groups | Demand forecasting and expiry risk |

See [Section 4 of the research doc](docs/01-research.md) for sources, verified shapes and known data-quality caveats.
