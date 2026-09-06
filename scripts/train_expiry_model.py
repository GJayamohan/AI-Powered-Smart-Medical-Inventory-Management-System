"""
Phase 3, model 2b - expiry risk prediction.

Dual-head model that:
  - Classifies whether a pharmacy batch will expire with unsold stock
  - Estimates the financial value-at-risk (Rs)

The split is done by batch_id (not by row) so that a single batch's multiple
time-horizon observations never leak between train and test.

Run with:  python scripts/train_expiry_model.py
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
    CORAL, SEED, TEAL, AMBER, SLATE,
    classification_metrics, print_table, save_results, set_seeds, style_matplotlib,
)
from medismart.utils.paths import FIGURES, MODELS, PROCESSED, REPORTS, ensure_dirs  # noqa: E402

EPOCHS = 150
BATCH_SIZE = 64

# Features the model will use (no ID columns, no targets, no date strings)
FEATURES = [
    "days_to_expiry", "qty_remaining", "pct_batch_remaining",
    "total_stock_all_batches", "usage_rate_7d", "usage_rate_30d",
    "usage_rate_90d", "usage_trend", "days_of_cover", "cover_ratio",
    "shelf_life_days", "pct_life_remaining", "unit_cost", "month",
    "prescription_required",
]

CLF_TARGET = "will_expire_unused"
REG_TARGET = "value_at_risk"


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def load_and_split():
    """Load expiry dataset and split by batch_id (no observation leakage)."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(PROCESSED / "expiry_risk_dataset.csv")
    # Convert boolean prescription_required to int if needed
    if df["prescription_required"].dtype == bool:
        df["prescription_required"] = df["prescription_required"].astype(int)

    batch_ids = df["batch_id"].unique()
    # Stratify by dominant class of each batch
    batch_labels = df.groupby("batch_id")[CLF_TARGET].max().loc[batch_ids]

    tr_ids, rest_ids = train_test_split(
        batch_ids, test_size=0.30, random_state=SEED,
        stratify=batch_labels.values,
    )
    val_ids, te_ids = train_test_split(
        rest_ids, test_size=0.50, random_state=SEED,
        stratify=batch_labels.loc[rest_ids].values,
    )

    train = df[df["batch_id"].isin(tr_ids)]
    val = df[df["batch_id"].isin(val_ids)]
    test = df[df["batch_id"].isin(te_ids)]
    return train, val, test


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def build_dual_head(n_features: int):
    """Dual-head model: classification + regression from shared trunk."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(n_features,), name="features")

    # Shared trunk
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    shared = layers.Dense(32, activation="relu")(x)

    # Classification head
    clf_out = layers.Dense(1, activation="sigmoid", name="clf")(shared)

    # Regression head
    reg = layers.Dense(16, activation="relu")(shared)
    reg_out = layers.Dense(1, activation="linear", name="reg")(reg)

    model = Model(inputs=inp, outputs=[clf_out, reg_out], name="expiry_risk")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"clf": "binary_crossentropy", "reg": "mse"},
        loss_weights={"clf": 1.0, "reg": 0.01},   # scale regression loss down
        metrics={"clf": "accuracy", "reg": "mae"},
    )
    return model


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the threshold that maximises F1 on the validation set."""
    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2)


# ------------------------------------------------------------------
# Classical baselines
# ------------------------------------------------------------------

def classical_models():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_results(rows, y_te, y_prob, y_pred, feature_importances, features):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    )

    style_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. ROC curve
    ax = axes[0, 0]
    RocCurveDisplay.from_predictions(y_te, y_prob, ax=ax, color=TEAL, lw=1.5)
    ax.plot([0, 1], [0, 1], ls="--", color=SLATE, lw=0.8)
    ax.set_title("ROC Curve", fontweight="bold")

    # 2. Precision-Recall curve
    ax = axes[0, 1]
    PrecisionRecallDisplay.from_predictions(y_te, y_prob, ax=ax, color=CORAL, lw=1.5)
    ax.set_title("Precision-Recall Curve", fontweight="bold")

    # 3. Confusion matrix
    ax = axes[1, 0]
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred, display_labels=["SAFE", "WILL EXPIRE"],
        cmap="Greens", ax=ax, colorbar=False,
    )
    ax.set_title("Confusion Matrix", fontweight="bold")

    # 4. Feature importance (from Gradient Boosting)
    ax = axes[1, 1]
    if feature_importances is not None:
        idx = np.argsort(feature_importances)
        ax.barh(range(len(features)), feature_importances[idx], color=TEAL, height=0.6)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([features[i] for i in idx], fontsize=7)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance (Gradient Boosting)", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Feature Importance", fontweight="bold")

    fig.suptitle("Model 2b - Expiry Risk Prediction", fontweight="bold", y=1.02)
    path = FIGURES / "11_expiry_risk_model.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure saved: {path.name}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    import joblib
    import tensorflow as tf

    ensure_dirs()
    set_seeds()

    print("=" * 74)
    print(" MediSmart - Phase 3, model 2b: expiry risk prediction")
    print("=" * 74)

    train, val, test = load_and_split()
    print(f"\n  train {len(train):,} rows ({train['batch_id'].nunique()} batches)")
    print(f"  val   {len(val):,} rows ({val['batch_id'].nunique()} batches)")
    print(f"  test  {len(test):,} rows ({test['batch_id'].nunique()} batches)")
    print(f"  positive rate - train: {train[CLF_TARGET].mean()*100:.1f}%  "
          f"test: {test[CLF_TARGET].mean()*100:.1f}%")
    print(f"  features: {len(FEATURES)}")

    # Preprocess
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[FEATURES].values).astype("float32")
    X_val = scaler.transform(val[FEATURES].values).astype("float32")
    X_te = scaler.transform(test[FEATURES].values).astype("float32")

    y_tr_clf = train[CLF_TARGET].values.astype("float32")
    y_val_clf = val[CLF_TARGET].values.astype("float32")
    y_te_clf = test[CLF_TARGET].values.astype("float32")

    y_tr_reg = train[REG_TARGET].values.astype("float32")
    y_val_reg = val[REG_TARGET].values.astype("float32")
    y_te_reg = test[REG_TARGET].values.astype("float32")

    # ---- Deep model (dual head) ----
    print("\n  Training dual-head DNN...")
    model = build_dual_head(len(FEATURES))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_clf_accuracy", patience=20,
            restore_best_weights=True, verbose=0, mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0,
        ),
    ]
    start = time.time()
    history = model.fit(
        X_tr, {"clf": y_tr_clf, "reg": y_tr_reg},
        validation_data=(X_val, {"clf": y_val_clf, "reg": y_val_reg}),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0,
    )
    dnn_time = time.time() - start
    epochs_run = len(history.history["loss"])
    print(f"  DNN trained in {dnn_time:.1f}s ({epochs_run} epochs)")

    # Predict
    clf_prob, reg_pred = model.predict(X_te, verbose=0)
    clf_prob = clf_prob.ravel()
    reg_pred = reg_pred.ravel()

    # Find optimal threshold on validation set
    val_clf_prob = model.predict(X_val, verbose=0)[0].ravel()
    threshold = find_optimal_threshold(y_val_clf, val_clf_prob)
    print(f"  optimal threshold (on val): {threshold}")

    clf_pred = (clf_prob >= threshold).astype(int)
    dnn_metrics = classification_metrics(y_te_clf.astype(int), clf_pred, clf_prob, "binary")
    dnn_metrics["model"] = "Deep Neural Network"
    dnn_metrics["train_seconds"] = round(dnn_time, 1)
    dnn_metrics["threshold"] = threshold

    # Regression metrics for value_at_risk
    reg_mae = float(np.mean(np.abs(y_te_reg - reg_pred)))
    reg_rmse = float(np.sqrt(np.mean((y_te_reg - reg_pred) ** 2)))
    dnn_metrics["reg_mae"] = round(reg_mae, 2)
    dnn_metrics["reg_rmse"] = round(reg_rmse, 2)

    print(f"  DNN: accuracy={dnn_metrics['accuracy']*100:.2f}%  "
          f"F1={dnn_metrics['f1']:.4f}  AUC={dnn_metrics.get('roc_auc', 0):.4f}  "
          f"reg MAE=Rs{reg_mae:.2f}")

    rows = [dnn_metrics]

    # ---- Classical baselines ----
    print("\n  Classical baselines...")
    gb_importance = None
    for name, clf in classical_models().items():
        s = time.time()
        clf.fit(X_tr, y_tr_clf)
        took = time.time() - s
        pred = clf.predict(X_te)
        try:
            prob = clf.predict_proba(X_te)[:, 1]
        except Exception:
            prob = None
        m = classification_metrics(y_te_clf.astype(int), pred, prob, "binary")
        m["model"] = name
        m["train_seconds"] = round(took, 1)
        rows.append(m)
        print(f"  {name:<22} accuracy={m['accuracy']*100:6.2f}%  "
              f"F1={m['f1']:.4f}  AUC={m.get('roc_auc', 0):.4f}")
        if name == "Gradient Boosting" and hasattr(clf, "feature_importances_"):
            gb_importance = clf.feature_importances_

    # ---- Comparison table ----
    print_table(
        sorted(rows, key=lambda r: -r.get("roc_auc", 0)),
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds"],
        title="COMPARISON (sorted by ROC-AUC)",
    )

    # ---- Risk buckets ----
    risk_config = {
        "threshold": threshold,
        "buckets": {
            "SAFE": [0, 25],
            "WATCH": [26, 50],
            "HIGH": [51, 75],
            "CRITICAL": [76, 100],
        },
    }

    # ---- Save artifacts ----
    model.save(MODELS / "expiry_model.keras")
    joblib.dump(scaler, MODELS / "expiry_preprocessor.pkl")
    (MODELS / "expiry_features.json").write_text(
        json.dumps(FEATURES, indent=2), encoding="utf-8"
    )
    (MODELS / "expiry_threshold.json").write_text(
        json.dumps(risk_config, indent=2), encoding="utf-8"
    )
    print(f"\n  saved expiry_model.keras, expiry_preprocessor.pkl, "
          f"expiry_features.json, expiry_threshold.json")

    # ---- Plot ----
    plot_results(rows, y_te_clf.astype(int), clf_prob, clf_pred,
                 gb_importance, FEATURES)

    # ---- Report ----
    save_results(
        {
            "task": "expiry_risk_prediction",
            "features": FEATURES,
            "splits": {
                "train": len(train), "val": len(val), "test": len(test),
                "train_batches": int(train["batch_id"].nunique()),
                "test_batches": int(test["batch_id"].nunique()),
            },
            "positive_rate": {
                "train": round(float(train[CLF_TARGET].mean()), 4),
                "test": round(float(test[CLF_TARGET].mean()), 4),
            },
            "risk_config": risk_config,
            "results": rows,
        },
        REPORTS / "phase3_expiry_results.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
