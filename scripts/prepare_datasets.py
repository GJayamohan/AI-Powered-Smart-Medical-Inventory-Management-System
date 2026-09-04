"""
Phase 2, step 2 - clean, de-duplicate, split and augment the raw datasets.

The important work happens on the disease-symptom dataset. Its 4,920 rows are
only 304 unique symptom patterns repeated over and over, so a naive random
split puts identical rows in both train and test. That leaks 100% of the test
set and is the reason published results on this dataset report 100% accuracy.

This script fixes that by splitting on *unique patterns*, then augmenting with
realistic noise so the model learns to tolerate a patient who forgets a symptom
or reports an unrelated one.

Outputs land in data/processed/. Run with:
    python scripts/prepare_datasets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medismart.utils.paths import PROCESSED, RAW, REPORTS, ensure_dirs  # noqa: E402

SEED = 42

# Augmentation settings, chosen to mimic a real pharmacy counter conversation.
P_DROP = 0.30      # chance a genuinely present symptom goes unmentioned
P_ADD = 0.02       # chance an unrelated symptom is reported by mistake
MIN_SYMPTOMS = 2   # never produce a sample with fewer than this many symptoms
N_AUG_TRAIN = 40   # variants generated per training pattern
N_AUG_EVAL = 15    # variants per validation / test pattern

# In Pima, a value of 0 is impossible for these fields; it encodes "missing".
PIMA_ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
PIMA_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

DRUG_GROUPS = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]

# What each ATC code actually means, needed later for the inventory seed data.
ATC_MEANING = {
    "M01AB": "Anti-inflammatory, acetic acid derivatives (e.g. diclofenac)",
    "M01AE": "Anti-inflammatory, propionic acid derivatives (e.g. ibuprofen)",
    "N02BA": "Salicylic acid derivatives (e.g. aspirin)",
    "N02BE": "Pyrazolones and anilides (e.g. paracetamol)",
    "N05B": "Anxiolytics",
    "N05C": "Hypnotics and sedatives",
    "R03": "Drugs for obstructive airway disease (asthma, COPD)",
    "R06": "Antihistamines for systemic use",
}

report: dict = {}


def banner(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


# ---------------------------------------------------------------------------
# 1. Disease-symptom dataset
# ---------------------------------------------------------------------------

def augment(
    X: np.ndarray, y: np.ndarray, n_variants: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy variants of each symptom pattern.

    Two kinds of realistic noise:
      - dropout: a present symptom the patient did not mention
      - false positive: an absent symptom reported in error

    The original clean pattern is always kept as the first variant.
    """
    out_X, out_y = [], []
    for row, label in zip(X, y):
        out_X.append(row.copy())
        out_y.append(label)
        present = np.flatnonzero(row == 1)
        absent = np.flatnonzero(row == 0)

        for _ in range(n_variants - 1):
            variant = row.copy()

            # Drop some present symptoms, but never below MIN_SYMPTOMS.
            if len(present) > MIN_SYMPTOMS:
                drop_mask = rng.random(len(present)) < P_DROP
                max_droppable = len(present) - MIN_SYMPTOMS
                droppable = present[drop_mask][:max_droppable]
                variant[droppable] = 0

            # Add a few spurious symptoms.
            add_mask = rng.random(len(absent)) < P_ADD
            variant[absent[add_mask]] = 1

            out_X.append(variant)
            out_y.append(label)

    return np.array(out_X, dtype="int8"), np.array(out_y)


def prepare_symptoms() -> None:
    banner("1. Disease-symptom dataset")
    from sklearn.model_selection import train_test_split

    raw = pd.read_csv(RAW / "disease_symptom_train.csv").dropna(axis=1, how="all")
    symptom_cols = [c for c in raw.columns if c != "prognosis"]
    print(f"  raw rows: {len(raw):,}  symptoms: {len(symptom_cols)}  diseases: {raw.prognosis.nunique()}")

    # THE key step: collapse to unique (pattern, disease) pairs.
    unique = raw.drop_duplicates().reset_index(drop=True)
    dup_pct = (1 - len(unique) / len(raw)) * 100
    print(f"  unique patterns: {len(unique)}  -> {dup_pct:.1f}% of the raw file was duplicated")

    # One symptom is never used by any disease; it carries zero information.
    never_used = [c for c in symptom_cols if unique[c].sum() == 0]
    if never_used:
        print(f"  dropping {len(never_used)} always-zero symptom(s): {never_used}")
        symptom_cols = [c for c in symptom_cols if c not in never_used]
        unique = unique[symptom_cols + ["prognosis"]]

    X = unique[symptom_cols].to_numpy(dtype="int8")
    y = unique["prognosis"].to_numpy()

    # Split by PATTERN, stratified by disease. No pattern can appear in two splits.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.35, random_state=SEED, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
    )
    print(f"  pattern split -> train {len(X_tr)}  val {len(X_val)}  test {len(X_te)}")

    rng = np.random.default_rng(SEED)
    Xa_tr, ya_tr = augment(X_tr, y_tr, N_AUG_TRAIN, rng)
    Xa_val, ya_val = augment(X_val, y_val, N_AUG_EVAL, rng)
    Xa_te, ya_te = augment(X_te, y_te, N_AUG_EVAL, rng)
    print(f"  after augmentation -> train {len(Xa_tr):,}  val {len(Xa_val):,}  test {len(Xa_te):,}")

    # Prove the leak is gone.
    train_set = {r.tobytes() for r in Xa_tr}
    overlap = sum(1 for r in Xa_te if r.tobytes() in train_set)
    print(f"  leakage check: {overlap}/{len(Xa_te)} test rows also appear in train ({overlap/len(Xa_te)*100:.2f}%)")

    for name, Xs, ys in [
        ("train", Xa_tr, ya_tr), ("val", Xa_val, ya_val), ("test", Xa_te, ya_te)
    ]:
        frame = pd.DataFrame(Xs, columns=symptom_cols)
        frame["prognosis"] = ys
        frame.to_csv(PROCESSED / f"symptoms_{name}.csv", index=False)

    (PROCESSED / "symptom_columns.json").write_text(
        json.dumps(symptom_cols, indent=2), encoding="utf-8"
    )
    (PROCESSED / "disease_classes.json").write_text(
        json.dumps(sorted(pd.unique(y).tolist()), indent=2), encoding="utf-8"
    )

    report["symptoms"] = {
        "raw_rows": len(raw),
        "unique_patterns": len(unique),
        "duplicate_pct": round(dup_pct, 2),
        "n_symptoms": len(symptom_cols),
        "n_diseases": int(pd.unique(y).size),
        "dropped_symptoms": never_used,
        "patterns": {"train": len(X_tr), "val": len(X_val), "test": len(X_te)},
        "augmented": {"train": len(Xa_tr), "val": len(Xa_val), "test": len(Xa_te)},
        "test_rows_leaked_into_train": overlap,
    }


# ---------------------------------------------------------------------------
# 2. Pima diabetes
# ---------------------------------------------------------------------------

def prepare_diabetes() -> None:
    banner("2. Pima diabetes dataset")
    raw = pd.read_csv(RAW / "pima_diabetes.csv", header=None)
    raw.columns = PIMA_COLUMNS
    print(f"  rows: {len(raw)}  outcome balance: {raw.Outcome.value_counts().to_dict()}")

    cleaned = raw.copy()
    missing_counts = {}
    for col in PIMA_ZERO_IS_MISSING:
        n_zero = int((cleaned[col] == 0).sum())
        missing_counts[col] = n_zero
        cleaned.loc[cleaned[col] == 0, col] = np.nan
        print(f"  {col:<16} {n_zero:>3} impossible zeros marked as missing ({n_zero/len(raw)*100:5.1f}%)")

    # Deliberately NOT imputing here. Imputation must be fitted inside the
    # cross-validation loop in Phase 3, otherwise statistics from the test fold
    # leak into training. We save NaNs and let the sklearn Pipeline handle it.
    cleaned.to_csv(PROCESSED / "diabetes_clean.csv", index=False)
    print("  saved with NaNs intact - imputation happens inside the CV pipeline (Phase 3)")

    report["diabetes"] = {
        "rows": len(raw),
        "outcome_balance": {str(k): int(v) for k, v in raw.Outcome.value_counts().items()},
        "zeros_marked_missing": missing_counts,
    }


# ---------------------------------------------------------------------------
# 3. UCI Cleveland heart disease
# ---------------------------------------------------------------------------

def prepare_heart() -> None:
    banner("3. UCI Cleveland heart disease dataset")
    raw = pd.read_csv(RAW / "heart_disease.csv")
    print(f"  rows: {len(raw)}  target balance: {raw.target.value_counts().to_dict()}")

    cleaned = raw.copy()

    n_dupes = int(cleaned.duplicated().sum())
    if n_dupes:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        print(f"  dropped {n_dupes} exact duplicate row(s)")

    # In the original UCI coding, ca=4 and thal=0 are placeholders, not real
    # categories. Left in place they become phantom classes for the model.
    n_ca = int((cleaned.ca == 4).sum())
    n_thal = int((cleaned.thal == 0).sum())
    cleaned.loc[cleaned.ca == 4, "ca"] = np.nan
    cleaned.loc[cleaned.thal == 0, "thal"] = np.nan
    print(f"  ca=4   -> missing: {n_ca} row(s)")
    print(f"  thal=0 -> missing: {n_thal} row(s)")

    cleaned.to_csv(PROCESSED / "heart_clean.csv", index=False)
    print(f"  saved {len(cleaned)} rows")

    report["heart"] = {
        "raw_rows": len(raw),
        "clean_rows": len(cleaned),
        "duplicates_dropped": n_dupes,
        "ca_placeholder_4": n_ca,
        "thal_placeholder_0": n_thal,
        "target_balance": {str(k): int(v) for k, v in raw.target.value_counts().items()},
    }


# ---------------------------------------------------------------------------
# 4. Pharmacy point-of-sale sales
# ---------------------------------------------------------------------------

def prepare_sales() -> None:
    banner("4. Pharmacy POS sales")
    raw = pd.read_csv(RAW / "pharma_sales_daily.csv")
    raw["date"] = pd.to_datetime(raw["datum"], format="mixed")
    raw = raw.sort_values("date").reset_index(drop=True)

    span = (raw.date.max() - raw.date.min()).days + 1
    print(f"  {len(raw):,} rows, {raw.date.min().date()} to {raw.date.max().date()}")
    print(f"  calendar days {span}, distinct dates {raw.date.nunique()}, missing {span - raw.date.nunique()}")

    wide = raw[["date"] + DRUG_GROUPS].copy()

    # Calendar features the LSTM and the baselines both benefit from.
    wide["year"] = wide.date.dt.year
    wide["month"] = wide.date.dt.month
    wide["day_of_week"] = wide.date.dt.dayofweek
    wide["is_weekend"] = (wide.day_of_week >= 5).astype(int)
    wide.to_csv(PROCESSED / "sales_daily_clean.csv", index=False)

    # Long format is easier to model per drug group and to join to inventory.
    long = wide.melt(
        id_vars=["date", "year", "month", "day_of_week", "is_weekend"],
        value_vars=DRUG_GROUPS, var_name="atc_group", value_name="units_sold",
    ).sort_values(["atc_group", "date"]).reset_index(drop=True)

    # Rolling usage rates - these become the core expiry-risk features.
    grouped = long.groupby("atc_group")["units_sold"]
    for window in (7, 30, 90):
        long[f"usage_rate_{window}d"] = (
            grouped.transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        )
    long["usage_trend"] = long["usage_rate_30d"] / long["usage_rate_90d"].replace(0, np.nan)
    long.to_csv(PROCESSED / "sales_long_clean.csv", index=False)

    stats = []
    print(f"\n  {'ATC':<7}{'mean/day':>10}{'std':>8}{'zero days':>11}{'seasonality':>13}")
    for atc in DRUG_GROUPS:
        series = wide[atc]
        monthly = wide.groupby("month")[atc].mean()
        # Ratio of the busiest month to the quietest: how seasonal is this drug?
        seasonality = float(monthly.max() / monthly.min()) if monthly.min() > 0 else float("inf")
        n_zero = int((series == 0).sum())
        print(f"  {atc:<7}{series.mean():>10.1f}{series.std():>8.1f}{n_zero:>11}{seasonality:>12.2f}x")
        stats.append({
            "atc_group": atc, "meaning": ATC_MEANING[atc],
            "mean_units_per_day": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "zero_sale_days": n_zero,
            "seasonality_ratio": round(seasonality, 2),
            "peak_month": int(monthly.idxmax()), "trough_month": int(monthly.idxmin()),
        })

    pd.DataFrame(stats).to_csv(PROCESSED / "drug_group_stats.csv", index=False)
    report["sales"] = {
        "rows": len(raw),
        "date_start": str(raw.date.min().date()),
        "date_end": str(raw.date.max().date()),
        "missing_days": int(span - raw.date.nunique()),
        "drug_groups": stats,
    }


def main() -> int:
    ensure_dirs()
    print("=" * 74)
    print(" MediSmart - Phase 2, step 2: cleaning and preparing datasets")
    print("=" * 74)

    prepare_symptoms()
    prepare_diabetes()
    prepare_heart()
    prepare_sales()

    out = REPORTS / "phase2_data_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    banner("Done")
    for path in sorted(PROCESSED.glob("*")):
        print(f"  {path.name:<28} {path.stat().st_size/1024:>8,.0f} KB")
    print(f"\n  Summary written to {out.relative_to(out.parents[2])}")
    print("  Next: python scripts/build_expiry_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
