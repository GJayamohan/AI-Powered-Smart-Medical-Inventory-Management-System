"""
Phase 3, model 1a - symptom-based disease classifier.

Trains a Keras deep neural network on the noise-augmented symptom data from
Phase 2, and benchmarks it against five classical scikit-learn models on
exactly the same splits.

The honest baseline to beat is 70.7% (decision tree on the pattern split).
Anything close to 100% here would mean the leakage from Phase 2 came back.

Run with:  python scripts/train_symptom_model.py
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
    set_seeds, style_matplotlib, top_k_accuracy,
)
from medismart.utils.paths import FIGURES, MODELS, PROCESSED, REPORTS, ensure_dirs  # noqa: E402

EPOCHS = 120
BATCH_SIZE = 64


def load_splits():
    train = pd.read_csv(PROCESSED / "symptoms_train.csv")
    val = pd.read_csv(PROCESSED / "symptoms_val.csv")
    test = pd.read_csv(PROCESSED / "symptoms_test.csv")
    features = [c for c in train.columns if c != "prognosis"]
    return train, val, test, features


# Candidate architectures. Symptom -> disease is close to a linear problem, so
# heavy regularisation can hurt; we let validation accuracy decide rather than
# assuming a deeper or more heavily regularised network is better.
ARCHITECTURES = [
    {"name": "256-128 drop 0.4/0.3", "units": (256, 128), "dropout": (0.4, 0.3), "bn": True},
    {"name": "256-128 drop 0.2/0.1", "units": (256, 128), "dropout": (0.2, 0.1), "bn": True},
    {"name": "512-256-128 drop 0.3", "units": (512, 256, 128), "dropout": (0.3, 0.3, 0.2), "bn": True},
    {"name": "128-64 drop 0.2", "units": (128, 64), "dropout": (0.2, 0.1), "bn": False},
]


def build_network(n_features: int, n_classes: int, cfg: dict):
    import tensorflow as tf
    from tensorflow.keras import layers

    seq = [layers.Input(shape=(n_features,))]
    for units, drop in zip(cfg["units"], cfg["dropout"]):
        seq.append(layers.Dense(units, activation="relu"))
        if cfg["bn"]:
            seq.append(layers.BatchNormalization())
        seq.append(layers.Dropout(drop))
    seq.append(layers.Dense(n_classes, activation="softmax"))

    model = tf.keras.Sequential(seq, name="symptom_dnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _fit_one(X_tr, y_tr, X_val, y_val, n_classes, cfg):
    import tensorflow as tf

    set_seeds()  # same starting point for every candidate: a fair comparison
    model = build_network(X_tr.shape[1], n_classes, cfg)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5, verbose=0
        ),
    ]
    start = time.time()
    history = model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0,
    )
    return model, history, time.time() - start


def train_dnn(X_tr, y_tr, X_val, y_val, n_classes):
    """Search a few architectures, selecting purely on validation accuracy.

    The test set is never consulted during selection. Choosing an architecture
    by test score is a subtle form of overfitting that inflates the reported
    number, and it is a very common mistake in student projects.
    """
    best = None
    print(f"  {'architecture':<24}{'params':>10}{'epochs':>8}{'val acc':>10}{'time':>8}")
    for cfg in ARCHITECTURES:
        model, history, elapsed = _fit_one(X_tr, y_tr, X_val, y_val, n_classes, cfg)
        val_acc = max(history.history["val_accuracy"])
        epochs_run = len(history.history["loss"])
        print(f"  {cfg['name']:<24}{model.count_params():>10,}{epochs_run:>8}"
              f"{val_acc*100:>9.2f}%{elapsed:>7.0f}s")
        if best is None or val_acc > best[3]:
            best = (model, history, elapsed, val_acc, cfg)

    model, history, elapsed, val_acc, cfg = best
    print(f"\n  selected on validation: {cfg['name']}  ({val_acc*100:.2f}% val accuracy)")
    return model, history, elapsed


def classical_models():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        # BernoulliNB, not GaussianNB: the features are binary presence flags.
        "Naive Bayes": BernoulliNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "SVM (RBF)": SVC(probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=SEED
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=60, max_depth=3, random_state=SEED
        ),
    }


def plot_results(rows, history, y_test, y_pred, classes):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    style_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # --- model comparison ---
    ax = axes[0]
    order = sorted(rows, key=lambda r: r["accuracy"])
    names = [r["model"] for r in order]
    accs = [r["accuracy"] * 100 for r in order]
    colours = [TEAL if n == "Deep Neural Network" else "#94a3b8" for n in names]
    ax.barh(range(len(names)), accs, color=colours, height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Deep learning vs classical ML", fontweight="bold")
    for i, v in enumerate(accs):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=8)

    # --- learning curves ---
    ax = axes[1]
    ax.plot(history.history["accuracy"], label="train", color=TEAL, lw=1.4)
    ax.plot(history.history["val_accuracy"], label="validation", color=CORAL, lw=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("DNN learning curve", fontweight="bold")
    ax.legend(fontsize=8)

    # --- confusion matrix ---
    ax = axes[2]
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    ax.set_title(f"Confusion matrix ({len(classes)} diseases)", fontweight="bold")
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Model 1a - symptom-based disease classification", fontweight="bold", y=1.03)
    path = FIGURES / "07_symptom_model_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure saved: {path.name}")


def main() -> int:
    ensure_dirs()
    set_seeds()

    print("=" * 74)
    print(" MediSmart - Phase 3, model 1a: symptom -> disease classifier")
    print("=" * 74)

    train, val, test, features = load_splits()
    print(f"\n  train {len(train):,}  val {len(val):,}  test {len(test):,}  "
          f"features {len(features)}")

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder().fit(pd.concat([train, val, test]).prognosis)
    classes = list(encoder.classes_)
    print(f"  diseases: {len(classes)}")

    X_tr = train[features].to_numpy("float32")
    X_val = val[features].to_numpy("float32")
    X_te = test[features].to_numpy("float32")
    y_tr = encoder.transform(train.prognosis)
    y_val = encoder.transform(val.prognosis)
    y_te = encoder.transform(test.prognosis)

    rows = []

    # ------------------------------------------------------------------
    print("\n  Deep Neural Network (Keras)")
    print("  " + "-" * 40)
    model, history, elapsed = train_dnn(X_tr, y_tr, X_val, y_val, len(classes))

    proba = model.predict(X_te, verbose=0)
    y_pred = proba.argmax(axis=1)
    dnn = classification_metrics(y_te, y_pred, proba)
    dnn["top3_accuracy"] = top_k_accuracy(y_te, proba, 3)
    dnn["model"] = "Deep Neural Network"
    dnn["train_seconds"] = round(elapsed, 1)
    rows.append(dnn)
    print(f"  test accuracy {dnn['accuracy']*100:.2f}%   "
          f"macro F1 {dnn['f1']:.4f}   top-3 {dnn['top3_accuracy']*100:.2f}%")

    # ------------------------------------------------------------------
    print("\n  Classical models (scikit-learn), same splits")
    print("  " + "-" * 40)
    for name, clf in classical_models().items():
        start = time.time()
        clf.fit(X_tr, y_tr)
        took = time.time() - start
        pred = clf.predict(X_te)
        try:
            prob = clf.predict_proba(X_te)
        except Exception:
            prob = None
        m = classification_metrics(y_te, pred, prob)
        if prob is not None:
            m["top3_accuracy"] = top_k_accuracy(y_te, prob, 3)
        m["model"] = name
        m["train_seconds"] = round(took, 1)
        rows.append(m)
        print(f"  {name:<22} accuracy {m['accuracy']*100:6.2f}%   "
              f"F1 {m['f1']:.4f}   ({took:.1f}s)")

    # ------------------------------------------------------------------
    print_table(
        sorted(rows, key=lambda r: -r["accuracy"]),
        ["model", "accuracy", "precision", "recall", "f1", "top3_accuracy", "train_seconds"],
        title="COMPARISON (sorted by test accuracy)",
    )

    best = max(rows, key=lambda r: r["accuracy"])
    print(f"\n  Best model: {best['model']} at {best['accuracy']*100:.2f}%")
    print(f"  Phase 2 honest baseline was 70.7% (decision tree on the pattern split).")
    if best["accuracy"] > 0.995:
        print("  WARNING: accuracy above 99.5% suggests leakage has returned. Investigate.")

    # ------------------------------------------------------------------
    model.save(MODELS / "symptom_model.keras")
    import joblib

    joblib.dump(encoder, MODELS / "symptom_label_encoder.pkl")
    (MODELS / "symptom_features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    print(f"\n  saved symptom_model.keras, symptom_label_encoder.pkl, symptom_features.json")

    plot_results(rows, history, y_te, y_pred, classes)
    save_results(
        {"task": "symptom_disease_classification",
         "n_classes": len(classes),
         "splits": {"train": len(train), "val": len(val), "test": len(test)},
         "results": rows},
        REPORTS / "phase3_symptom_results.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
