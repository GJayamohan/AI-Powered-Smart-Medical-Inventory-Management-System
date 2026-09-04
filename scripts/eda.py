"""
Phase 2, step 4 - exploratory data analysis.

Produces the figures that go into the project report, saved as PNG files in
artifacts/figures/. Each figure is built to make one specific point, not just
to decorate the report.

Run with:  python scripts/eda.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we only write files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medismart.utils.paths import FIGURES, PROCESSED, RAW, ensure_dirs  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TEAL, CORAL, SLATE, AMBER = "#0f766e", "#e05252", "#475569", "#d97706"
saved: list[str] = []


def save(fig, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    saved.append(name)
    print(f"  saved {name}")


# ---------------------------------------------------------------------------

def fig_leakage() -> None:
    """The single most important figure: why published 100% results are wrong."""
    raw = pd.read_csv(RAW / "disease_symptom_train.csv").dropna(axis=1, how="all")
    X = raw.drop(columns=["prognosis"])
    y = raw["prognosis"]

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    # Naive approach: split rows at random.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_rows = set(map(tuple, Xtr.values))
    leaked = sum(1 for r in map(tuple, Xte.values) if r in train_rows)
    naive_acc = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)

    # Our approach: split unique patterns, then evaluate on augmented data.
    tr = pd.read_csv(PROCESSED / "symptoms_train.csv")
    te = pd.read_csv(PROCESSED / "symptoms_test.csv")
    cols = [c for c in tr.columns if c != "prognosis"]
    honest_acc = (
        DecisionTreeClassifier(random_state=0)
        .fit(tr[cols], tr.prognosis)
        .score(te[cols], te.prognosis)
    )
    train_rows2 = {r.tobytes() for r in tr[cols].to_numpy(dtype="int8")}
    leaked2 = sum(1 for r in te[cols].to_numpy(dtype="int8") if r.tobytes() in train_rows2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    bars = ax1.bar(["Naive\nrow split", "Pattern split\n+ augmentation"],
                   [leaked / len(Xte) * 100, leaked2 / len(te) * 100],
                   color=[CORAL, TEAL], width=0.55)
    ax1.set_ylabel("% of test rows found verbatim in training")
    ax1.set_title("Data leakage", fontweight="bold")
    ax1.set_ylim(0, 108)
    for b, v in zip(bars, [leaked / len(Xte) * 100, leaked2 / len(te) * 100]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.1f}%", ha="center", fontweight="bold")

    bars = ax2.bar(["Naive\nrow split", "Pattern split\n+ augmentation"],
                   [naive_acc * 100, honest_acc * 100], color=[CORAL, TEAL], width=0.55)
    ax2.set_ylabel("Decision tree accuracy (%)")
    ax2.set_title("Resulting accuracy", fontweight="bold")
    ax2.set_ylim(0, 108)
    for b, v in zip(bars, [naive_acc * 100, honest_acc * 100]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.1f}%", ha="center", fontweight="bold")

    fig.suptitle("Why published 100% accuracy on this dataset is a leakage artifact",
                 fontweight="bold", y=1.04)
    save(fig, "01_leakage_demonstration.png")


def fig_symptoms() -> None:
    raw = pd.read_csv(RAW / "disease_symptom_train.csv").dropna(axis=1, how="all")
    unique = raw.drop_duplicates()
    symptom_cols = [c for c in raw.columns if c != "prognosis"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    per_disease = unique.groupby("prognosis").size().sort_values()
    ax1.barh(range(len(per_disease)), per_disease.values, color=TEAL, height=0.7)
    ax1.set_yticks(range(len(per_disease)))
    ax1.set_yticklabels(per_disease.index, fontsize=6)
    ax1.set_xlabel("Unique symptom patterns")
    ax1.set_title(f"Only {len(unique)} distinct patterns across {len(raw):,} rows", fontweight="bold")

    freq = unique[symptom_cols].sum().sort_values(ascending=False).head(20)
    ax2.barh(range(len(freq)), freq.values, color=SLATE, height=0.7)
    ax2.set_yticks(range(len(freq)))
    ax2.set_yticklabels([s.replace("_", " ") for s in freq.index], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Number of disease patterns featuring this symptom")
    ax2.set_title("20 most common symptoms", fontweight="bold")

    save(fig, "02_symptom_distribution.png")


def fig_diabetes() -> None:
    df = pd.read_csv(PROCESSED / "diabetes_clean.csv")
    raw = pd.read_csv(RAW / "pima_diabetes.csv", header=None)
    raw.columns = df.columns

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    pct = [(raw[c] == 0).sum() / len(raw) * 100 for c in cols]
    axes[0].barh(cols, pct, color=[CORAL if p > 20 else AMBER for p in pct], height=0.6)
    axes[0].set_xlabel("% of rows with an impossible zero")
    axes[0].set_title("Zero-coded missing values", fontweight="bold")
    for i, p in enumerate(pct):
        axes[0].text(p + 1, i, f"{p:.1f}%", va="center", fontsize=8)

    # The two features that drive the worked example in the project brief.
    for ax, col in zip(axes[1:], ["Glucose", "BMI"]):
        for outcome, colour, label in [(0, TEAL, "No diabetes"), (1, CORAL, "Diabetes")]:
            ax.hist(df[df.Outcome == outcome][col].dropna(), bins=25, alpha=0.6,
                    color=colour, label=label)
        ax.set_xlabel(col)
        ax.set_ylabel("Patients")
        ax.set_title(f"{col} by outcome", fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Pima Indians Diabetes - data quality and class separation",
                 fontweight="bold", y=1.05)
    save(fig, "03_diabetes_analysis.png")


def fig_heart() -> None:
    df = pd.read_csv(PROCESSED / "heart_clean.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    corr = df.corr(numeric_only=True)["target"].drop("target").sort_values()
    colours = [CORAL if v < 0 else TEAL for v in corr.values]
    ax1.barh(range(len(corr)), corr.values, color=colours, height=0.7)
    ax1.set_yticks(range(len(corr)))
    ax1.set_yticklabels(corr.index, fontsize=8)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_xlabel("Correlation with heart disease")
    ax1.set_title("Feature correlation with target", fontweight="bold")

    im = ax2.imshow(df.corr(numeric_only=True), cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(df.columns)))
    ax2.set_xticklabels(df.columns, rotation=90, fontsize=6)
    ax2.set_yticks(range(len(df.columns)))
    ax2.set_yticklabels(df.columns, fontsize=6)
    ax2.grid(False)
    ax2.set_title("Correlation matrix", fontweight="bold")
    fig.colorbar(im, ax=ax2, shrink=0.8)

    fig.suptitle("UCI Cleveland Heart Disease", fontweight="bold", y=1.02)
    save(fig, "04_heart_analysis.png")


def fig_sales() -> None:
    daily = pd.read_csv(PROCESSED / "sales_daily_clean.csv", parse_dates=["date"])
    stats = pd.read_csv(PROCESSED / "drug_group_stats.csv")
    groups = stats.atc_group.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))

    ax = axes[0, 0]
    for atc in ["N02BE", "R03", "R06"]:
        ax.plot(daily.date, daily[atc].rolling(30).mean(), label=atc, lw=1.2)
    ax.set_title("30-day rolling demand (real POS data)", fontweight="bold")
    ax.set_ylabel("Units per day")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    monthly = daily.groupby("month")[groups].mean()
    for atc in ["R06", "R03", "N02BE"]:
        ax.plot(monthly.index, monthly[atc] / monthly[atc].mean(), marker="o", label=atc, lw=1.4)
    ax.axhline(1.0, color=SLATE, ls="--", lw=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Demand relative to annual mean")
    ax.set_title("Seasonality: winter peaks, summer troughs", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    order = stats.sort_values("seasonality_ratio")
    ax.barh(order.atc_group, order.seasonality_ratio, color=TEAL, height=0.65)
    ax.axvline(1.0, color=SLATE, ls="--", lw=0.8)
    ax.set_xlabel("Peak month / trough month")
    ax.set_title("How seasonal is each drug group?", fontweight="bold")
    for i, (_, r) in enumerate(order.iterrows()):
        ax.text(r.seasonality_ratio + 0.03, i, f"{r.seasonality_ratio:.2f}x", va="center", fontsize=7)

    ax = axes[1, 1]
    order = stats.sort_values("mean_units_per_day")
    ax.barh(order.atc_group, order.mean_units_per_day, color=SLATE, height=0.65)
    ax.set_xlabel("Mean units sold per day")
    ax.set_title("Fast vs slow movers", fontweight="bold")

    fig.suptitle("Pharmacy point-of-sale demand, 2014-2019", fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "05_sales_demand.png")


def fig_expiry() -> None:
    df = pd.read_csv(PROCESSED / "expiry_risk_dataset.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))

    ax = axes[0, 0]
    counts = df.will_expire_unused.value_counts().sort_index()
    ax.bar(["Sold in time", "Expired unused"], counts.values, color=[TEAL, CORAL], width=0.5)
    ax.set_ylabel("Observations")
    ax.set_title("Label balance", fontweight="bold")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=8)

    # cover_ratio < 1 means demand cannot clear the batch before expiry.
    ax = axes[0, 1]
    for label, colour, name in [(0, TEAL, "Sold in time"), (1, CORAL, "Expired unused")]:
        vals = df[df.will_expire_unused == label]["cover_ratio"].clip(0, 6)
        ax.hist(vals, bins=40, alpha=0.6, color=colour, label=name)
    ax.axvline(1.0, color="black", ls="--", lw=1.2)
    ax.text(1.05, ax.get_ylim()[1] * 0.85, "ratio = 1", fontsize=8)
    ax.set_xlabel("cover_ratio  =  (usage rate x days left) / qty remaining")
    ax.set_ylabel("Observations")
    ax.set_title("The strongest single predictor", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    grouped = df.groupby("atc_group").will_expire_unused.mean().sort_values() * 100
    ax.barh(grouped.index, grouped.values, color=AMBER, height=0.65)
    ax.set_xlabel("% of observations that expired unused")
    ax.set_title("Expiry risk by drug group", fontweight="bold")

    ax = axes[1, 1]
    numeric = ["days_to_expiry", "qty_remaining", "pct_batch_remaining", "usage_rate_30d",
               "usage_trend", "days_of_cover", "cover_ratio", "pct_life_remaining"]
    corr = df[numeric + ["will_expire_unused"]].corr()["will_expire_unused"].drop(
        "will_expire_unused").sort_values()
    ax.barh(range(len(corr)), corr.values,
            color=[CORAL if v < 0 else TEAL for v in corr.values], height=0.65)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Correlation with expiring unused")
    ax.set_title("Feature signal strength", fontweight="bold")

    fig.suptitle("Expiry-risk dataset (real demand, simulated batch ledger)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "06_expiry_risk.png")


def main() -> int:
    ensure_dirs()
    print("=" * 74)
    print(" MediSmart - Phase 2, step 4: exploratory data analysis")
    print("=" * 74 + "\n")

    for name, fn in [
        ("leakage demonstration", fig_leakage),
        ("symptom distribution", fig_symptoms),
        ("diabetes analysis", fig_diabetes),
        ("heart analysis", fig_heart),
        ("sales demand", fig_sales),
        ("expiry risk", fig_expiry),
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED {name}: {type(exc).__name__}: {exc}")
            return 1

    print(f"\n  {len(saved)} figures written to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
