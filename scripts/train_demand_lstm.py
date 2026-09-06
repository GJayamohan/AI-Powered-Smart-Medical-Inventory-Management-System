"""
Phase 3, model 2a - LSTM demand forecasting per ATC drug group.

Predicts daily sales for each of the 8 ATC drug groups using a sliding-window
LSTM, then benchmarks against moving-average and classical ML baselines.

The temporal split ensures no future leakage:
    Train: 2014-01-02 - 2017-12-31  (~70%)
    Val:   2018-01-01 - 2018-12-31  (~15%)
    Test:  2019-01-01 - 2019-10-08  (~15%)

Run with:  python scripts/train_demand_lstm.py
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
    CORAL, SEED, TEAL, AMBER, PURPLE, SLATE,
    save_results, set_seeds, style_matplotlib,
)
from medismart.utils.paths import FIGURES, MODELS, PROCESSED, REPORTS, ensure_dirs  # noqa: E402

WINDOW = 60          # days of history the LSTM sees
HORIZON = 1          # predict 1 day ahead
EPOCHS = 80
BATCH_SIZE = 32
TRAIN_END = "2017-12-31"
VAL_END = "2018-12-31"


def flush_print(*args, **kwargs):
    """Print with immediate flush so output appears in logs."""
    print(*args, **kwargs, flush=True)


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load the long-format cleaned daily sales and return groups."""
    df = pd.read_csv(PROCESSED / "sales_long_clean.csv", parse_dates=["date"])
    groups = sorted(df["atc_group"].unique())
    return df, groups


def build_features(series: pd.Series) -> pd.DataFrame:
    """From a raw daily sales series, build lag/calendar features."""
    df = pd.DataFrame({"sales": series.values}, index=series.index)
    df["roll_7"] = df["sales"].rolling(7, min_periods=1).mean()
    df["roll_30"] = df["sales"].rolling(30, min_periods=1).mean()
    df["dow"] = df.index.dayofweek / 6.0
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    return df.dropna()


def make_windows(data: np.ndarray, window: int = WINDOW):
    """Slice a 2-D feature array into (X, y) sliding windows."""
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window, 0])  # target = sales column (index 0)
    if len(X) == 0:
        return np.empty((0, window, data.shape[1]), dtype="float32"), np.empty(0, dtype="float32")
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

def build_lstm(n_features: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Input(shape=(WINDOW, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="demand_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def moving_average_baseline(train_sales: np.ndarray, test_sales: np.ndarray,
                            window: int) -> dict:
    """Simple moving average baseline."""
    preds = []
    history = list(train_sales[-window:])
    for actual in test_sales:
        pred = np.mean(history[-window:])
        preds.append(pred)
        history.append(actual)
    preds = np.array(preds)
    return _regression_metrics(test_sales, preds)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, and MAPE."""
    ae = np.abs(y_true - y_pred)
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true > 0
    mape = float(np.mean(ae[mask] / y_true[mask]) * 100) if mask.any() else float("nan")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 2)}


def _inv_scale_target(values: np.ndarray, scaler, col_idx: int = 0) -> np.ndarray:
    """Inverse-transform just the target column from MinMaxScaler."""
    mn = scaler.data_min_[col_idx]
    mx = scaler.data_max_[col_idx]
    rng = mx - mn
    if rng == 0:
        return values * 0.0
    return values * rng + mn


# ------------------------------------------------------------------
# Per-group training loop
# ------------------------------------------------------------------

def train_group(df: pd.DataFrame, group: str) -> dict:
    """Train LSTM + baselines for one ATC group."""
    import tensorflow as tf
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler

    set_seeds()

    grp = df[df["atc_group"] == group].set_index("date").sort_index()
    series = grp["units_sold"]
    feat = build_features(series)

    # Temporal split
    train_df = feat.loc[:TRAIN_END]
    val_df = feat.loc[TRAIN_END:VAL_END].iloc[1:]
    test_df = feat.loc[VAL_END:].iloc[1:]

    if len(test_df) < WINDOW + 1:
        flush_print(f"    {group}: too few test points ({len(test_df)}), skipping")
        return {"group": group, "skipped": True}

    # Scale on training data only
    scaler = MinMaxScaler()
    scaler.fit(train_df.values)

    train_sc = scaler.transform(train_df.values)
    val_sc = scaler.transform(val_df.values)
    test_sc = scaler.transform(test_df.values)

    X_tr, y_tr = make_windows(train_sc)
    X_val, y_val = make_windows(val_sc)

    # For test, prepend context from end of val period
    context_needed = min(WINDOW, len(val_sc))
    test_with_context = np.vstack([val_sc[-context_needed:], test_sc])
    X_te, y_te = make_windows(test_with_context)

    n_features = X_tr.shape[2]

    flush_print(f"    {group}: train {len(X_tr)} | val {len(X_val)} | test {len(X_te)} windows")

    # --- LSTM ---
    model = build_lstm(n_features)
    use_val = len(X_val) > 0
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if use_val else "loss",
            patience=12, restore_best_weights=True, verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if use_val else "loss",
            factor=0.5, patience=6, min_lr=1e-5, verbose=0,
        ),
    ]
    val_data = (X_val, y_val) if use_val else None
    start = time.time()
    history = model.fit(
        X_tr, y_tr, validation_data=val_data,
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0,
    )
    lstm_time = time.time() - start
    epochs_run = len(history.history["loss"])

    # Inverse-scale predictions for real-unit metrics
    y_te_real = _inv_scale_target(y_te, scaler)
    lstm_pred_sc = model.predict(X_te, verbose=0).ravel()
    lstm_pred_real = _inv_scale_target(lstm_pred_sc, scaler)

    lstm_metrics = _regression_metrics(y_te_real, lstm_pred_real)
    lstm_metrics["model"] = "LSTM"
    lstm_metrics["train_seconds"] = round(lstm_time, 1)
    lstm_metrics["epochs_run"] = epochs_run

    # --- Baselines on raw test sales ---
    raw_train_sales = train_df["sales"].values
    raw_test_sales = _inv_scale_target(y_te, scaler)  # aligned to same length as LSTM test

    rows = [lstm_metrics]

    for w_name, w in [("MA-7d", 7), ("MA-30d", 30)]:
        m = moving_average_baseline(raw_train_sales, raw_test_sales, w)
        m["model"] = w_name
        m["train_seconds"] = 0.0
        rows.append(m)

    # Classical ML on flattened windows
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
    X_te_flat = X_te.reshape(X_te.shape[0], -1)

    for name, clf in [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)),
    ]:
        start = time.time()
        clf.fit(X_tr_flat, y_tr)
        took = time.time() - start
        clf_pred_sc = clf.predict(X_te_flat)
        clf_pred_real = _inv_scale_target(clf_pred_sc, scaler)
        m = _regression_metrics(y_te_real, clf_pred_real)
        m["model"] = name
        m["train_seconds"] = round(took, 1)
        rows.append(m)

    # Save per-group artifacts
    model.save(MODELS / f"demand_lstm_{group}.keras")
    joblib.dump(scaler, MODELS / f"demand_scaler_{group}.pkl")

    flush_print(f"    LSTM MAE={lstm_metrics['mae']:.2f}  "
                f"MA-7d={rows[1]['mae']:.2f}  MA-30d={rows[2]['mae']:.2f}  "
                f"({epochs_run} epochs, {lstm_time:.0f}s)")

    return {
        "group": group,
        "train_size": int(len(X_tr)),
        "test_size": int(len(X_te)),
        "results": rows,
        "y_te_real": y_te_real.tolist(),
        "lstm_pred_real": lstm_pred_real.tolist(),
    }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_results(all_results: dict) -> None:
    import matplotlib.pyplot as plt

    style_matplotlib()
    groups = [g for g, r in all_results.items() if not r.get("skipped")]
    n = len(groups)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: MAE comparison across groups
    ax = axes[0, 0]
    bar_w = 0.18
    x = np.arange(n)
    model_names = ["LSTM", "MA-7d", "MA-30d", "Linear Regression", "Random Forest"]
    colours = [TEAL, SLATE, CORAL, AMBER, PURPLE]
    for i, (mname, col) in enumerate(zip(model_names, colours)):
        maes = []
        for g in groups:
            res = all_results[g]
            row = next((r for r in res["results"] if r["model"] == mname), None)
            maes.append(row["mae"] if row else 0)
        ax.bar(x + i * bar_w, maes, bar_w, label=mname, color=col, alpha=0.85)
    ax.set_xticks(x + bar_w * 2)
    ax.set_xticklabels(groups, fontsize=7, rotation=30)
    ax.set_ylabel("MAE (units/day)")
    ax.set_title("MAE by model and drug group", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    # Panel 2: LSTM vs best baseline improvement
    ax = axes[0, 1]
    improvements = []
    for g in groups:
        res = all_results[g]
        lstm_mae = next(r["mae"] for r in res["results"] if r["model"] == "LSTM")
        baseline_mae = min(r["mae"] for r in res["results"] if r["model"] != "LSTM")
        imp = ((baseline_mae - lstm_mae) / baseline_mae * 100) if baseline_mae > 0 else 0
        improvements.append(imp)
    colours_bar = [TEAL if v >= 0 else CORAL for v in improvements]
    ax.barh(range(n), improvements, color=colours_bar, height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlabel("LSTM improvement over best baseline (%)")
    ax.set_title("LSTM vs best classical baseline", fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    for i, v in enumerate(improvements):
        ax.text(v + (1 if v >= 0 else -1), i, f"{v:.1f}%", va="center", fontsize=7,
                ha="left" if v >= 0 else "right")

    # Panel 3: Sample forecast (pick group with most test data)
    best_g = max(groups, key=lambda g: len(all_results[g].get("y_te_real", [])))
    res = all_results[best_g]
    ax = axes[1, 0]
    actual = res["y_te_real"][:90]
    predicted = res["lstm_pred_real"][:90]
    ax.plot(actual, label="Actual", color=SLATE, lw=1.2, alpha=0.8)
    ax.plot(predicted, label="LSTM prediction", color=TEAL, lw=1.2)
    ax.set_xlabel("Day (test period)")
    ax.set_ylabel("Units sold")
    ax.set_title(f"LSTM forecast vs actual - {best_g}", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 4: Error distribution
    ax = axes[1, 1]
    errors = np.array(res["y_te_real"]) - np.array(res["lstm_pred_real"])
    ax.hist(errors, bins=30, color=TEAL, alpha=0.7, edgecolor="white")
    ax.axvline(0, color=CORAL, lw=1.5, ls="--")
    ax.set_xlabel("Prediction error (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"LSTM error distribution - {best_g}", fontweight="bold")
    ax.text(0.95, 0.95, f"Mean: {np.mean(errors):.2f}\nStd: {np.std(errors):.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Model 2a - LSTM demand forecasting", fontweight="bold", y=1.02)
    path = FIGURES / "10_demand_forecast.png"
    fig.savefig(path)
    plt.close(fig)
    flush_print(f"\n  figure saved: {path.name}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    ensure_dirs()
    set_seeds()

    flush_print("=" * 74)
    flush_print(" MediSmart - Phase 3, model 2a: LSTM demand forecasting")
    flush_print("=" * 74)

    df, groups = load_data()
    flush_print(f"\n  {len(df):,} daily records across {len(groups)} ATC groups")
    flush_print(f"  date range: {df['date'].min().date()} to {df['date'].max().date()}")
    flush_print(f"  train <= {TRAIN_END}  |  val <= {VAL_END}  |  test = remainder\n")

    all_results = {}
    for group in groups:
        flush_print(f"  Training {group}...")
        result = train_group(df, group)
        all_results[group] = result

    # Summary table
    flush_print(f"\n  {'group':<10}{'LSTM MAE':>10}{'MA-7d MAE':>12}{'MA-30d MAE':>12}"
                f"{'LR MAE':>10}{'RF MAE':>10}")
    flush_print("  " + "-" * 64)
    for g in groups:
        res = all_results[g]
        if res.get("skipped"):
            flush_print(f"  {g:<10}{'(skipped)':>10}")
            continue
        vals = {r["model"]: r["mae"] for r in res["results"]}
        flush_print(f"  {g:<10}{vals.get('LSTM', 0):>10.2f}{vals.get('MA-7d', 0):>12.2f}"
                    f"{vals.get('MA-30d', 0):>12.2f}{vals.get('Linear Regression', 0):>10.2f}"
                    f"{vals.get('Random Forest', 0):>10.2f}")

    # Save config
    config = {
        "window": WINDOW,
        "horizon": HORIZON,
        "groups": groups,
        "feature_columns": ["sales", "roll_7", "roll_30", "dow", "month_sin", "month_cos"],
        "train_end": TRAIN_END,
        "val_end": VAL_END,
    }
    (MODELS / "demand_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    flush_print(f"\n  saved demand_config.json")

    # Clean results for JSON (remove large arrays)
    json_results = {}
    for g, res in all_results.items():
        r = {k: v for k, v in res.items()
             if k not in ("y_te_real", "lstm_pred_real")}
        json_results[g] = r

    plot_results(all_results)
    save_results(
        {"task": "demand_forecasting_lstm", "config": config, "results": json_results},
        REPORTS / "phase3_demand_results.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
