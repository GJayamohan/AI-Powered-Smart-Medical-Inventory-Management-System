"""
Phase 2, step 1 - download every raw dataset.

All sources are public GitHub raw URLs that need no login and no API key, so
this script is fully reproducible on any machine with internet access.

Each download is validated against an expected shape. A file that downloads
successfully but has the wrong contents (a mirror changed, a repo was
restructured, or GitHub returned an HTML error page saved as .csv) is caught
here rather than producing a confusing failure during model training.

Run with:  python scripts/download_datasets.py
Options:   --force   re-download files that already exist
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medismart.utils.paths import RAW, ensure_dirs  # noqa: E402

TIMEOUT = 60


@dataclass
class Dataset:
    """One downloadable file plus what we expect it to contain."""

    key: str
    filename: str
    url: str
    description: str
    source_page: str
    # Validation: expected shape after loading. None means "do not check".
    expect_rows: int | None = None
    expect_cols: int | None = None
    read_kwargs: dict = field(default_factory=dict)
    # Some files carry a trailing all-empty column that must be dropped first.
    drop_empty_cols: bool = False


DATASETS: list[Dataset] = [
    # ---- 1. Disease-symptom, the primary classification dataset ----
    Dataset(
        key="symptom_train",
        filename="disease_symptom_train.csv",
        url="https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/training_data.csv",
        description="Disease-symptom training set: 132 binary symptoms, 41 diseases",
        source_page="https://github.com/anujdutt9/Disease-Prediction-from-Symptoms",
        expect_rows=4920,
        expect_cols=133,
        drop_empty_cols=True,
    ),
    Dataset(
        key="symptom_test",
        filename="disease_symptom_test.csv",
        url="https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/test_data.csv",
        description="Disease-symptom held-out test set from the same source",
        source_page="https://github.com/anujdutt9/Disease-Prediction-from-Symptoms",
        expect_cols=133,
        drop_empty_cols=True,
    ),
    # ---- 2. Knowledge files that become the Treatment Database ----
    Dataset(
        key="symptom_description",
        filename="symptom_description.csv",
        url="https://raw.githubusercontent.com/amMistic/Diseases-Prediction-based-on-Symptoms/main/Dataset/symptom_Description.csv",
        description="Plain-language description of each of the 41 diseases",
        source_page="https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms",
        expect_cols=2,
    ),
    Dataset(
        key="symptom_precaution",
        filename="symptom_precaution.csv",
        url="https://raw.githubusercontent.com/amMistic/Diseases-Prediction-based-on-Symptoms/main/Dataset/symptom_precaution.csv",
        description="Four recommended precautions per disease",
        source_page="https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms",
        expect_cols=5,
    ),
    Dataset(
        key="symptom_severity",
        filename="symptom_severity.csv",
        url="https://raw.githubusercontent.com/amMistic/Diseases-Prediction-based-on-Symptoms/main/Dataset/Symptom-severity.csv",
        description="Severity weight per symptom, used for triage urgency",
        source_page="https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms",
        expect_cols=2,
    ),
    # ---- 3. Pima diabetes: the vitals branch of the fusion model ----
    Dataset(
        key="diabetes",
        filename="pima_diabetes.csv",
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        description="Pima Indians Diabetes (NIDDK): Age, Glucose, BMI, BloodPressure, Outcome",
        source_page="https://github.com/jbrownlee/Datasets",
        expect_rows=768,
        expect_cols=9,
        read_kwargs={"header": None},
    ),
    # ---- 4. UCI Cleveland heart disease ----
    Dataset(
        key="heart",
        filename="heart_disease.csv",
        url="https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
        description="UCI Cleveland Heart Disease, 13 clinical features + target",
        source_page="https://archive.ics.uci.edu/dataset/45/heart+disease",
        expect_rows=303,
        expect_cols=14,
    ),
    # ---- 5. Real pharmacy point-of-sale data for expiry risk ----
    Dataset(
        key="sales_daily",
        filename="pharma_sales_daily.csv",
        url="https://raw.githubusercontent.com/mcallara/pharma-sales-data/master/salesdaily.csv",
        description="Real pharmacy POS daily sales 2014-2019, 8 ATC drug groups",
        source_page="https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data",
        expect_rows=2106,
        expect_cols=13,
    ),
    Dataset(
        key="sales_monthly",
        filename="pharma_sales_monthly.csv",
        url="https://raw.githubusercontent.com/mcallara/pharma-sales-data/master/salesmonthly.csv",
        description="Same POS data aggregated monthly, used for seasonality analysis",
        source_page="https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data",
        expect_cols=9,
    ),
]


def download_one(ds: Dataset, force: bool) -> tuple[bool, str]:
    """Download and validate a single dataset. Returns (ok, message)."""
    target = RAW / ds.filename

    if target.exists() and not force:
        try:
            frame = _load(target, ds)
            return True, f"cached  {frame.shape[0]:>5} x {frame.shape[1]:<3} {ds.filename}"
        except Exception:
            # Cached file is corrupt; fall through and re-download it.
            pass

    try:
        resp = requests.get(ds.url, timeout=TIMEOUT)
    except Exception as exc:
        return False, f"NETWORK {ds.filename}: {type(exc).__name__}: {exc}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} {ds.filename}: {ds.url}"

    # Guard against a mirror that 200s with an HTML error page.
    head = resp.content[:200].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        return False, f"HTML    {ds.filename}: got a web page, not a CSV"

    target.write_bytes(resp.content)

    try:
        frame = _load(target, ds)
    except Exception as exc:
        return False, f"PARSE   {ds.filename}: {type(exc).__name__}: {exc}"

    rows, cols = frame.shape
    if ds.expect_rows is not None and rows != ds.expect_rows:
        return False, f"SHAPE   {ds.filename}: expected {ds.expect_rows} rows, got {rows}"
    if ds.expect_cols is not None and cols != ds.expect_cols:
        return False, f"SHAPE   {ds.filename}: expected {ds.expect_cols} cols, got {cols}"

    size_kb = len(resp.content) / 1024
    return True, f"ok      {rows:>5} x {cols:<3} {ds.filename}  ({size_kb:,.0f} KB)"


def _load(path: Path, ds: Dataset) -> pd.DataFrame:
    frame = pd.read_csv(path, **ds.read_kwargs)
    if ds.drop_empty_cols:
        frame = frame.dropna(axis=1, how="all")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MediSmart raw datasets")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    ensure_dirs()

    print("=" * 74)
    print(" MediSmart - Phase 2, step 1: downloading raw datasets")
    print(f" Target: {RAW}")
    print("=" * 74)

    failures = []
    for ds in DATASETS:
        ok, message = download_one(ds, args.force)
        print(f"  {message}")
        if not ok:
            failures.append(ds.filename)

    print("-" * 74)
    if failures:
        print(f"  {len(failures)} of {len(DATASETS)} downloads FAILED: {', '.join(failures)}")
        print("  Check your internet connection, or the mirror may have moved.")
        return 1

    total_mb = sum(f.stat().st_size for f in RAW.glob("*.csv")) / 1024 / 1024
    print(f"  All {len(DATASETS)} datasets downloaded and shape-validated ({total_mb:.1f} MB).")
    print("  Next: python scripts/prepare_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
