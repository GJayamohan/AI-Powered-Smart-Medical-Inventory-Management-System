"""
Phase 1 verification script.

Confirms that this machine can actually build and run the project:
  1. Python version is new enough
  2. Every pinned dependency imports at the expected version
  3. All four datasets are reachable without a login

Run with:  python scripts/check_environment.py
"""

from __future__ import annotations

import importlib.metadata as metadata
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)

# Datasets the project depends on. Checked live so a dead mirror is caught
# here, in Phase 1, rather than halfway through Phase 2.
DATASETS = {
    "Disease-symptom (4920x133, 41 diseases)": "https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/training_data.csv",
    "Symptom descriptions": "https://raw.githubusercontent.com/amMistic/Diseases-Prediction-based-on-Symptoms/main/Dataset/symptom_Description.csv",
    "Symptom precautions": "https://raw.githubusercontent.com/amMistic/Diseases-Prediction-based-on-Symptoms/main/Dataset/symptom_precaution.csv",
    "Symptom severity": "https://raw.githubusercontent.com/amMistic/Diseases-Prediction-based-on-Symptoms/main/Dataset/Symptom-severity.csv",
    "Pima diabetes (768x9)": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "UCI heart disease (303x14)": "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
    "Pharmacy POS daily sales (2106x13)": "https://raw.githubusercontent.com/mcallara/pharma-sales-data/master/salesdaily.csv",
}

PASS = "  [ OK ]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def header(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check_python() -> bool:
    header("1. Python version")
    ok = sys.version_info >= MIN_PYTHON
    got = ".".join(str(p) for p in sys.version_info[:3])
    need = ".".join(str(p) for p in MIN_PYTHON)
    print(f"{PASS if ok else FAIL} Python {got} (need >= {need})")
    return ok


def check_packages() -> bool:
    header("2. Dependencies (from requirements.txt)")
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    if not req.exists():
        print(f"{FAIL} requirements.txt not found at {req}")
        return False

    all_ok = True
    for raw in req.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, pinned = line.split("==", 1)
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            print(f"{FAIL} {name}: not installed (pinned {pinned})")
            all_ok = False
            continue
        if installed == pinned:
            print(f"{PASS} {name}=={installed}")
        else:
            # A drift is not fatal, but it can change model results, so say so.
            print(f"{WARN} {name}: installed {installed}, pinned {pinned}")
    return all_ok


def check_datasets() -> bool:
    header("3. Dataset availability (no login required)")
    try:
        import requests
    except ImportError:
        print(f"{FAIL} requests not installed, cannot check dataset URLs")
        return False

    all_ok = True
    for label, url in DATASETS.items():
        try:
            resp = requests.get(url, timeout=25, stream=True)
            size = len(resp.content)
            if resp.status_code == 200 and size > 1000:
                print(f"{PASS} {label} ({size:,} bytes)")
            else:
                print(f"{FAIL} {label}: HTTP {resp.status_code}, {size} bytes")
                all_ok = False
        except Exception as exc:  # network down, DNS, TLS, timeout
            print(f"{FAIL} {label}: {type(exc).__name__}: {exc}")
            all_ok = False
    return all_ok


def check_tensorflow() -> bool:
    header("4. TensorFlow smoke test")
    try:
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import numpy as np
        import tensorflow as tf

        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2, activation="softmax")]
        )
        out = model.predict(np.zeros((1, 4), dtype="float32"), verbose=0)
        ok = out.shape == (1, 2)
        print(f"{PASS if ok else FAIL} Keras built and ran a model, output shape {out.shape}")
        print(f"{PASS} TensorFlow {tf.__version__} on {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
        return ok
    except Exception as exc:
        print(f"{FAIL} TensorFlow smoke test failed: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print("=" * 62)
    print(" MediSmart - Phase 1 environment verification")
    print("=" * 62)

    results = {
        "Python version": check_python(),
        "Dependencies": check_packages(),
        "Datasets reachable": check_datasets(),
        "TensorFlow works": check_tensorflow(),
    }

    header("SUMMARY")
    for name, ok in results.items():
        print(f"{PASS if ok else FAIL} {name}")

    if all(results.values()):
        print("\nEnvironment is ready. Phase 2 (Dataset) can begin.\n")
        return 0
    print("\nSome checks failed. Fix the FAIL lines above before Phase 2.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
