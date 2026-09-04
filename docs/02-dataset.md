# Phase 2 — Dataset Acquisition, Cleaning and Preparation

**Phase status:** Complete
**Date:** 2026-09-04

Reproduce this entire phase from a clean checkout with four commands:

```bash
python scripts/download_datasets.py
python scripts/prepare_datasets.py
python scripts/build_expiry_dataset.py
python scripts/eda.py
```

---

## 1. Datasets used

All sources are public GitHub raw URLs requiring **no Kaggle account and no API key**, so the pipeline is fully reproducible on any machine.

| # | Dataset | Downloaded shape | Used for | Direct link | Canonical source |
|---|---|---|---|---|---|
| 1 | Disease–Symptom (train) | 4,920 × 133 | Symptom → disease classifier | [GitHub](https://github.com/anujdutt9/Disease-Prediction-from-Symptoms) | [Columbia DBMI](https://impact.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html) · [Kaggle](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) |
| 2 | Disease–Symptom (test) | 42 × 133 | Sanity check only — see §2.3 | same | same |
| 3 | Symptom descriptions | 41 × 2 | Disease explanations | [GitHub](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms) | [Kaggle](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) |
| 4 | Symptom precautions | 41 × 5 | Precaution advice | same | same |
| 5 | Symptom severity | 133 × 2 | Triage urgency weighting | same | same |
| 6 | Pima Indians Diabetes | 768 × 9 | Vitals → diabetes risk | [GitHub](https://github.com/jbrownlee/Datasets) | NIDDK · [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| 7 | UCI Cleveland Heart Disease | 303 × 14 | Vitals → heart risk | [GitHub](https://github.com/kb22/Heart-Disease-Prediction) | [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| 8 | Pharmacy POS sales (daily) | 2,106 × 13 | Demand forecasting, expiry risk | [GitHub](https://github.com/mcallara/pharma-sales-data) | [Kaggle](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data) |
| 9 | Pharmacy POS sales (monthly) | 70 × 9 | Seasonality analysis | same | same |

`scripts/download_datasets.py` validates every file against its expected shape and rejects HTML error pages saved as `.csv`, so a moved mirror fails loudly here instead of silently corrupting model training.

---

## 2. The central finding: the symptom dataset leaks

### 2.1 What we found

The dataset is universally described as "4,920 samples". It is not.

| Measurement | Value |
|---|---|
| Rows in the file | 4,920 |
| **Unique (symptom pattern, disease) pairs** | **304** |
| Duplicate rows | 4,616 (**93.8%**) |
| Unique patterns per disease | min 5, max 10, mean 7.4 |
| Symptoms never used by any disease | 1 (`fluid_overload`) |

Each disease is one of roughly 7 fixed symptom templates, copied out to exactly 120 rows.

### 2.2 Why that destroys a naive evaluation

With 93.8% duplicates, a random row-level train/test split puts **identical rows on both sides**:

| Split strategy | Test rows found verbatim in train | Decision tree accuracy |
|---|---|---|
| Naive random row split | **984 / 984 (100.0%)** | **100.0%** |
| Our pattern split + augmentation | 10 / 810 (1.2%) | **70.7%** |

The model is not classifying; it is recalling rows it has already seen. **Every published "100% accuracy" result on this dataset is a data-leakage artifact.** See `artifacts/figures/01_leakage_demonstration.png`.

That 70.7% honest baseline is good news for the project: it leaves genuine room for the deep learning model in Phase 3 to demonstrate value, which a 100% ceiling would not.

### 2.3 A note on the supplied test file

The dataset ships a `test_data.csv` of only **42 rows** — roughly one per disease. That is far too small to estimate 41-class accuracy with any confidence, so we do not use it for evaluation. It is kept only as a smoke test. Our own test split, built from held-out patterns, is used instead.

### 2.4 How we fixed it

1. **Deduplicate** 4,920 rows → 304 unique patterns.
2. **Drop** the always-zero `fluid_overload` column (132 → 131 symptoms), since it carries no information.
3. **Split on patterns, not rows**, stratified by disease, so no pattern can appear in two splits:

   | Split | Unique patterns | After augmentation |
   |---|---|---|
   | Train | 197 | 7,880 |
   | Validation | 53 | 795 |
   | Test | 54 | 810 |

4. **Augment with realistic noise**, modelling how patients actually describe symptoms:
   - **Symptom dropout** (`p = 0.30`) — the patient forgets to mention something
   - **False positives** (`p = 0.02`) — an unrelated symptom is reported
   - A floor of 2 symptoms per sample; the clean original pattern is always kept

   40 variants per training pattern, 15 per evaluation pattern.

5. **Verify**: residual overlap between train and test fell to **1.2%**, and that remainder is coincidental collision between independently generated noisy variants, not copied rows.

---

## 3. Pima Indians Diabetes

**Data quality problem.** The dataset encodes missing measurements as `0`, which is biologically impossible for these fields:

| Field | Impossible zeros | % of rows |
|---|---|---|
| Insulin | 374 | 48.7% |
| SkinThickness | 227 | 29.6% |
| BloodPressure | 35 | 4.6% |
| BMI | 11 | 1.4% |
| Glucose | 5 | 0.7% |

Left untreated, the model learns that a BMI of 0 predicts something.

**What we did.** Converted these zeros to `NaN` and saved them that way in `diabetes_clean.csv`.

**What we deliberately did *not* do:** impute them here. Computing a median over the whole dataset and then cross-validating leaks information from the test folds into training. Imputation is instead fitted **inside** the CV pipeline in Phase 3, on training folds only. This is a small detail that quietly inflates results in a great many student projects.

Class balance is 500 negative / 268 positive (34.9% positive) — imbalanced enough that accuracy alone is a misleading metric, so Phase 3 reports ROC-AUC, precision and recall as well.

---

## 4. UCI Cleveland Heart Disease

| Issue | Rows affected | Action |
|---|---|---|
| Exact duplicate row | 1 | Dropped |
| `ca = 4` (a placeholder, not a real vessel count) | 4 | Set to `NaN` |
| `thal = 0` (placeholder) | 2 | Set to `NaN` |

Result: 302 clean rows, target balance 165 / 138. Left in place, those placeholders become phantom categories that the model treats as meaningful.

At 302 rows this is a genuinely small dataset. Phase 3 will very likely show tree ensembles matching or beating the neural network here, and we will report that honestly — deep learning needs data volume, and saying so demonstrates understanding rather than weakness.

---

## 5. Pharmacy point-of-sale sales

Quality is excellent: **2,106 rows, 0 missing days, 0 null values**, spanning 2014-01-02 to 2019-10-08.

| ATC group | Meaning | Units/day | Zero-sale days | Seasonality (peak ÷ trough) |
|---|---|---|---|---|
| N02BE | Paracetamol / anilides | 29.9 | 26 | 2.15× |
| N05B | Anxiolytics | 8.9 | 43 | 1.47× |
| R03 | Asthma / COPD | 5.5 | 484 | 2.68× |
| M01AB | Diclofenac-type NSAIDs | 5.0 | 40 | 1.15× |
| M01AE | Ibuprofen-type NSAIDs | 3.9 | 36 | 1.31× |
| N02BA | Aspirin | 3.9 | 78 | 1.39× |
| R06 | Antihistamines | 2.9 | 256 | **3.69×** |
| N05C | Hypnotics / sedatives | 0.6 | 1,430 | 1.65× |

**Why this matters for Phase 3.** R06 antihistamines swing **3.69×** between their busiest and quietest month, and R03 respiratory drugs **2.68×**. That is a strong, real seasonal signal — exactly the structure an LSTM can exploit and a flat moving-average baseline cannot. It justifies the deep learning approach on evidence rather than assertion.

N05C sells nothing on 1,430 of 2,106 days. It is a very sparse series and will be reported separately rather than being allowed to distort aggregate forecasting metrics.

**Engineered features** (`sales_long_clean.csv`): calendar fields, rolling usage rates over 7/30/90 days, and `usage_trend = usage_30d / usage_90d`.

---

## 6. Building the expiry-risk dataset

### 6.1 The problem

No pharmacy publishes its batch and expiry ledger, so no public dataset exists for this task. But we do have six years of **real consumption**.

### 6.2 Approach, stated plainly

We simulate the batch ledger while driving it entirely with the **real demand signal** — real volumes, real seasonality, real weekday effects.

> **Limitation, stated openly:** consumption is real; procurement and expiry metadata are simulated. The model learns the relationship between *real* usage rates and expiry outcomes. This should be declared in the report; it is a reasonable approach to an unavailable-data problem, not a hidden shortcut.

### 6.3 The simulation

A genuine pharmacy inventory loop over 2,106 days for 22 medicines across 8 ATC groups:

```
weekly stock review  ->  reorder if cover < 25 days
                     ->  receive batch with an expiry date
                     ->  dispense daily demand FEFO (First-Expiry-First-Out)
                     ->  units left in a batch on its expiry date are wasted
```

**A first attempt produced exactly zero waste**, because a policy that orders 75 days of cover against a 400+ day shelf life can never expire anything. That is not how pharmacies actually lose money. The three real causes were then modelled explicitly:

| Cause | Model |
|---|---|
| **Pack sizes** | Stock is sold in whole packs (100 tablets, 500 tablets, 5 inhalers). For a slow mover the smallest orderable pack can exceed a year of demand. |
| **Bulk-discount over-ordering** | 15% of orders are inflated 2.5–5× to chase a supplier discount. |
| **Short-dated deliveries** | 18% of batches arrive with only 8–30% of shelf life left. |

To generate enough labelled data, a small **chain of 8 stores** is simulated, each with its own demand noise and ordering luck, driven by the same real sales signal.

### 6.4 Result

| Metric | Value |
|---|---|
| Batches created | 2,921 |
| Batches reaching expiry | 2,275 |
| Batches with waste | 210 |
| Units wasted | 35,664 of 1.1M received |
| **Waste rate** | **3.23%** |
| Value wasted | ₹928,860 |

**3.23% sits inside the 3–5% industry benchmark** for pharmacy expiry write-off cited in Phase 1 — the simulation was not tuned to hit that; it emerged from the pack-size and short-dating mechanics.

A further sanity check: **N05C (sedatives), the slowest mover at 0.6 units/day, has the highest expiry rate at 61%**, while fast-moving N02BE paracetamol has the lowest at 33%. That ordering is what pack-size economics predicts, and it was not imposed by hand.

### 6.5 The labelled dataset

Each batch is observed at 11 points before expiry (240, 180, 150, 120, 90, 60, 45, 30, 21, 14, 7 days), producing **3,824 labelled rows across 877 unique batches** — 43.9% positive, a healthy balance.

**Features**

| Feature | Meaning |
|---|---|
| `days_to_expiry` | Days remaining |
| `qty_remaining`, `pct_batch_remaining` | How much of this batch is left |
| `total_stock_all_batches` | Total stock of this medicine |
| `usage_rate_7d / 30d / 90d` | Rolling consumption from **real** sales |
| `usage_trend` | 30-day rate ÷ 90-day rate — is demand accelerating? |
| `days_of_cover` | Days of stock left at current speed |
| **`cover_ratio`** | **(usage rate × days left) ÷ qty remaining. Below 1 means demand cannot clear the batch before it expires.** |
| `shelf_life_days`, `pct_life_remaining` | Batch age context |
| `unit_cost`, `month`, `prescription_required` | Economic and seasonal context |

**Labels:** `will_expire_unused` (binary), `units_wasted` (regression), `value_at_risk` (rupees).

`cover_ratio` separates the two classes almost perfectly at the value 1 (see `artifacts/figures/06_expiry_risk.png`) — a strong sign the feature engineering captures the real mechanism rather than a statistical accident.

> **Important for Phase 3:** the 3,824 rows come from only 877 batches, so rows sharing a `batch_id` are **not independent**. Cross-validation must group by `batch_id`, or results will be optimistically biased — the same mistake as §2.2, in a different disguise.

---

## 7. Outputs

**`data/processed/`**

| File | Contents |
|---|---|
| `symptoms_train / val / test.csv` | 7,880 / 795 / 810 augmented rows, 131 symptoms |
| `symptom_columns.json`, `disease_classes.json` | Feature and label ordering, fixed for inference |
| `diabetes_clean.csv` | 768 rows, missing values marked `NaN` |
| `heart_clean.csv` | 302 rows, placeholders marked `NaN` |
| `sales_daily_clean.csv`, `sales_long_clean.csv` | Cleaned sales with calendar and rolling features |
| `drug_group_stats.csv` | Per-ATC-group demand and seasonality summary |
| `medicine_catalog.csv` | 22 medicines: cost, price, pack size, Rx flag — seeds the inventory DB in Phase 6 |
| `expiry_risk_dataset.csv` | 3,824 labelled rows |
| `batch_ledger.csv`, `inventory_simulation_log.csv` | Full simulation trace |

**`artifacts/figures/`** — six report-ready figures:

1. `01_leakage_demonstration.png` — the headline finding
2. `02_symptom_distribution.png` — 304 patterns, top 20 symptoms
3. `03_diabetes_analysis.png` — missing values, Glucose/BMI separation
4. `04_heart_analysis.png` — feature correlations
5. `05_sales_demand.png` — demand, seasonality, fast vs slow movers
6. `06_expiry_risk.png` — label balance, `cover_ratio` separation, risk by group

**`artifacts/reports/`** — `phase2_data_report.json`, `phase2_expiry_report.json`.

Raw and processed data are git-ignored by design; the scripts regenerate everything.

---

## 8. Carried into Phase 3

1. Group CV by `batch_id` for expiry; never split augmented symptom variants of the same pattern across folds.
2. Fit imputation and scaling **inside** the CV pipeline, never before it.
3. Report ROC-AUC, precision and recall alongside accuracy — Pima and the expiry set are imbalanced.
4. Honest baselines to beat: **70.7%** (symptom decision tree), and a moving-average forecast for demand.
5. Expect tree ensembles to be competitive on the 302-row heart dataset, and report it plainly if so.
