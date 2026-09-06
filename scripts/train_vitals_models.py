"""
Phase 3, models 1b and 1c - vitals-based risk models.

  Model 1b: Pima Indians Diabetes  -> P(diabetes) from Age, Glucose, BMI, BP...
  Model 1c: UCI Cleveland Heart    -> P(heart disease) from 13 clinical features

Model 1b is what makes the project brief's worked example work: Age 45,
Glucose 180, BMI 29, BP 140/90 goes into this network.

Methodology note that matters: imputation and scaling are fitted INSIDE each
cross-validation fold, never on the whole dataset first. Pima has 48.7%
missing Insulin values, so a median computed over everything and then
cross-validated would leak test-fold information into training and quietly
inflate every score.

Run with:  python scripts/train_vitals_models.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medismart.models.common import (  # noqa: E402
    CORAL, SEED, TEAL, classification_metrics, print_table, save_results,
    set_seeds, style_matplotlib,
)
from medismart.utils.paths import FIGURES, MODELS, PROCESSED, REPORTS, ensure_dirs  # noqa: E402

N_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 16

TASKS = {
    "diabetes": {
        "file": "diabetes_clean.csv",
        "target": "Outcome",
        "title": "Model 1b - Diabetes risk (Pima Indians)",
        "slug": "diabetes",
    },
    "heart": {
        "file": "heart_clean.csv",
        "target": "target",
        "title": "Model 1c - Heart disease risk (UCI Cleveland)",
        "slug": "heart",
    },
}


def make_preprocessor():
    """Median imputation then standardisation, as a fittable unit."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def classical_models():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


def build_network(n_features: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name="vitals_dnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_dnn(X_tr, y_tr, X_val, y_val):
    import tensorflow as tf

    set_seeds()
    model = build_network(X_tr.shape[1])
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0
        ),
    ]
    model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0,
    )
    return model


def cross_validate(X: pd.DataFrame, y: np.ndarray) -> list[dict]:
    """5-fold stratified CV. Every fold refits its own imputer and scaler."""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores: dict[str, list[dict]] = {name: [] for name in classical_models()}
    scores["Deep Neural Network"] = []
    timings: dict[str, float] = {}

    for train_idx, test_idx in folds.split(X, y):
        X_tr_raw, X_te_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Fitted on the training fold ONLY.
        pre = make_preprocessor().fit(X_tr_raw)
        X_tr, X_te = pre.transform(X_tr_raw), pre.transform(X_te_raw)

        for name, clf in classical_models().items():
            start = time.time()
            clf.fit(X_tr, y_tr)
            timings[name] = timings.get(name, 0) + time.time() - start
            proba = clf.predict_proba(X_te)[:, 1]
            scores[name].append(classification_metrics(y_te, clf.predict(X_te), proba, "binary"))

        # The network needs its own validation slice, carved out of the
        # training fold so the test fold stays completely untouched.
        Xa, Xb, ya, yb = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr
        )
        start = time.time()
        model = fit_dnn(Xa, ya, Xb, yb)
        timings["Deep Neural Network"] = timings.get("Deep Neural Network", 0) + time.time() - start
        proba = model.predict(X_te, verbose=0).ravel()
        scores["Deep Neural Network"].append(
            classification_metrics(y_te, (proba >= 0.5).astype(int), proba, "binary")
        )

    rows = []
    for name, fold_scores in scores.items():
        row = {"model": name}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            values = [f[metric] for f in fold_scores if metric in f]
            if values:
                row[metric] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values))
        row["train_seconds"] = round(timings[name], 1)
        rows.append(row)
    return rows


def fit_final_model(X: pd.DataFrame, y: np.ndarray, slug: str) -> None:
    """Refit on all data and save, for use by the web application."""
    import joblib
    from sklearn.model_selection import train_test_split

    pre = make_preprocessor().fit(X)
    X_all = pre.transform(X)
    Xa, Xb, ya, yb = train_test_split(X_all, y, test_size=0.2, random_state=SEED, stratify=y)
    model = fit_dnn(Xa, ya, Xb, yb)

    model.save(MODELS / f"{slug}_model.keras")
    joblib.dump(pre, MODELS / f"{slug}_preprocessor.pkl")
    (MODELS / f"{slug}_features.json").write_text(
        json.dumps(list(X.columns), indent=2), encoding="utf-8"
    )
    print(f"  saved {slug}_model.keras, {slug}_preprocessor.pkl, {slug}_features.json")


def plot_task(rows: list[dict], title: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    style_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    order = sorted(rows, key=lambda r: r["accuracy"])
    names = [r["model"] for r in order]

    for ax, metric, label in [(ax1, "accuracy", "Accuracy"), (ax2, "roc_auc", "ROC-AUC")]:
        values = [r.get(metric, 0) for r in order]
        errors = [r.get(f"{metric}_std", 0) for r in order]
        colours = [TEAL if n == "Deep Neural Network" else "#94a3b8" for n in names]
        ax.barh(range(len(names)), values, xerr=errors, color=colours, height=0.6,
                error_kw={"ecolor": CORAL, "capsize": 3, "lw": 1})
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(f"{label} (5-fold CV mean, bars show std)")
        ax.set_xlim(0, 1.05)
        ax.set_title(label, fontweight="bold")
        for i, v in enumerate(values):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8)

    fig.suptitle(title, fontweight="bold", y=1.04)
    fig.savefig(FIGURES / filename)
    plt.close(fig)
    print(f"  figure saved: {filename}")


def main() -> int:
    ensure_dirs()
    set_seeds()
    print("=" * 74)
    print(" MediSmart - Phase 3, models 1b and 1c: vitals-based risk models")
    print("=" * 74)

    all_results = {}
    for key, cfg in TASKS.items():
        print(f"\n{cfg['title']}")
        print("=" * len(cfg["title"]))

        df = pd.read_csv(PROCESSED / cfg["file"])
        X = df.drop(columns=[cfg["target"]])
        y = df[cfg["target"]].to_numpy()
        missing = int(X.isna().sum().sum())
        print(f"  rows {len(df)}  features {X.shape[1]}  positives {y.sum()} "
              f"({y.mean()*100:.1f}%)  missing cells {missing}")
        print(f"  {N_FOLDS}-fold stratified CV, imputation and scaling refitted per fold")

        rows = cross_validate(X, y)
        print_table(
            sorted(rows, key=lambda r: -r.get("roc_auc", 0)),
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds"],
            title="RESULTS (5-fold CV means, sorted by ROC-AUC)",
        )

        best = max(rows, key=lambda r: r.get("roc_auc", 0))
        dnn = next(r for r in rows if r["model"] == "Deep Neural Network")
        print(f"\n  Best by ROC-AUC: {best['model']} ({best['roc_auc']:.4f})")
        if best["model"] != "Deep Neural Network":
            gap = best["roc_auc"] - dnn["roc_auc"]
            print(f"  The DNN is {gap:.4f} ROC-AUC behind. With {len(df)} rows this is expected:")
            print("  tree ensembles typically win on small tabular datasets.")

        print()
        fit_final_model(X, y, cfg["slug"])
        plot_task(rows, cfg["title"],
                  f"08_{cfg['slug']}_model_comparison.png" if key == "diabetes"
                  else f"09_{cfg['slug']}_model_comparison.png")

        all_results[key] = {
            "rows": len(df), "features": list(X.columns),
            "positive_rate": float(y.mean()), "results": rows,
        }

    save_results(all_results, REPORTS / "phase3_vitals_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
