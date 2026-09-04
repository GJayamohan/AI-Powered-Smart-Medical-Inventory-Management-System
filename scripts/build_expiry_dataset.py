"""
Phase 2, step 3 - build the expiry-risk training dataset.

No pharmacy publishes its batch and expiry ledger, so no public dataset exists
for expiry prediction. What we DO have is six years of real point-of-sale
consumption from an actual pharmacy.

So we simulate the batch ledger, but drive it entirely with the real demand
signal: real daily volumes, real seasonality, real weekday effects. The
procurement and expiry metadata is synthetic; the consumption the model learns
from is not. This limitation is stated openly in docs/02-dataset.md.

The simulation runs a genuine pharmacy inventory loop:
    reorder when stock is low  ->  receive a batch with an expiry date
    ->  dispense daily demand FEFO (First-Expiry-First-Out)
    ->  whatever is left in a batch on its expiry date is wasted

Every batch is then observed at several points in its life, producing labelled
training rows: given what the shelf looked like on day X, how many units of
this batch actually went to waste?

Run with:  python scripts/build_expiry_dataset.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medismart.utils.paths import PROCESSED, REPORTS, ensure_dirs  # noqa: E402

SEED = 42
rng = np.random.default_rng(SEED)

# Medicines per ATC group, with realistic Indian-pharmacy pricing in rupees.
# (brand name, share of the group's demand, unit cost, shelf life months, pack size)
#
# Pack size matters enormously. A pharmacy cannot order 3 tablets; it orders a
# strip, a box or a jar. For a slow-moving medicine the smallest orderable pack
# can be a year of stock, and that is a primary real-world cause of expiry.
CATALOG_SPEC = {
    "M01AB": [("Diclofenac 50mg Tab", 0.45, 2.5, 24, 100), ("Diclofenac Gel 30g", 0.30, 85.0, 24, 10),
              ("Aceclofenac 100mg Tab", 0.25, 4.0, 36, 100)],
    "M01AE": [("Ibuprofen 400mg Tab", 0.50, 1.8, 36, 200), ("Ibuprofen Suspension 100ml", 0.25, 62.0, 24, 10),
              ("Naproxen 250mg Tab", 0.25, 6.5, 24, 100)],
    "N02BA": [("Aspirin 75mg Tab", 0.60, 0.8, 24, 500), ("Aspirin 325mg Tab", 0.40, 1.5, 24, 200)],
    "N02BE": [("Paracetamol 500mg Tab", 0.55, 1.0, 36, 500), ("Paracetamol 650mg Tab", 0.25, 1.4, 36, 200),
              ("Paracetamol Syrup 60ml", 0.20, 45.0, 24, 20)],
    "N05B": [("Alprazolam 0.25mg Tab", 0.40, 3.0, 24, 100), ("Clonazepam 0.5mg Tab", 0.35, 4.2, 24, 100),
             ("Buspirone 10mg Tab", 0.25, 8.0, 24, 60)],
    "N05C": [("Zolpidem 5mg Tab", 0.55, 9.0, 24, 100), ("Melatonin 3mg Tab", 0.45, 12.0, 18, 60)],
    "R03": [("Salbutamol Inhaler 100mcg", 0.45, 165.0, 24, 5), ("Budesonide Inhaler", 0.30, 320.0, 24, 5),
            ("Montelukast 10mg Tab", 0.25, 11.0, 24, 60)],
    "R06": [("Cetirizine 10mg Tab", 0.45, 1.2, 36, 200), ("Loratadine 10mg Tab", 0.30, 2.8, 36, 100),
            ("Levocetirizine 5mg Tab", 0.25, 2.0, 36, 100)],
}

# Prescription-only groups. Needed by the recommendation engine later so it
# never suggests a controlled medicine as an over-the-counter option.
RX_ONLY_GROUPS = {"N05B", "N05C", "R03"}

MARKUP = 1.35            # retail price over cost
REVIEW_EVERY_DAYS = 7    # pharmacy reviews stock weekly
TARGET_COVER_DAYS = 75   # order enough to cover this many days of demand
REORDER_COVER_DAYS = 25  # reorder once stock falls below this many days of cover

# The three real-world causes of pharmacy expiry waste, modelled explicitly.
P_BULK_BUY = 0.15        # pharmacist over-orders to chase a bulk discount
BULK_MULTIPLIER = (2.5, 5.0)
P_SHORT_DATED = 0.18     # supplier ships stock already well through its life
SHORT_DATED_LIFE = (0.08, 0.30)
NORMAL_LIFE = (0.50, 0.90)

# Observation points, expressed as days before expiry. This mirrors how the
# live application will actually query the model ("what is at risk in 90 days?")
# far better than arbitrary fractions of shelf life would.
OBSERVE_DAYS_BEFORE = [240, 180, 150, 120, 90, 60, 45, 30, 21, 14, 7]

# One store yields only a few hundred labelled batches. We simulate a small
# chain, each store with its own demand noise and ordering luck, driven by the
# same real sales signal. Rows from one batch share an id so Phase 3 can group
# them and avoid splitting the same batch across train and test.
N_STORES = 8


@dataclass
class Batch:
    batch_id: str
    medicine_id: int
    received_on: pd.Timestamp
    expires_on: pd.Timestamp
    qty_received: int
    qty_remaining: int
    unit_cost: float
    shelf_life_days: int
    # Daily history of how much of THIS batch was still on the shelf.
    history: dict = field(default_factory=dict)
    wasted: int = 0
    expired: bool = False


def build_catalog() -> pd.DataFrame:
    """Expand the spec into a medicine master table."""
    rows, mid = [], 1
    for atc, items in CATALOG_SPEC.items():
        for name, share, cost, shelf_months, pack in items:
            rows.append({
                "medicine_id": mid, "name": name, "atc_group": atc,
                "demand_share": share, "unit_cost": cost,
                "unit_price": round(cost * MARKUP, 2),
                "shelf_life_months": shelf_months,
                "pack_size": pack,
                "prescription_required": atc in RX_ONLY_GROUPS,
            })
            mid += 1
    return pd.DataFrame(rows)


def split_demand(sales: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Turn group-level daily sales into per-medicine daily demand.

    The group total is real. We split it across that group's medicines by their
    demand share, then add small Poisson noise so each medicine has its own
    believable day-to-day variation instead of a perfectly scaled copy.
    """
    frames = []
    for _, med in catalog.iterrows():
        group_units = sales[med.atc_group].to_numpy(dtype=float)
        expected = group_units * med.demand_share
        noisy = rng.poisson(np.clip(expected, 0, None))
        frames.append(pd.DataFrame({
            "date": sales["date"].to_numpy(),
            "medicine_id": med.medicine_id,
            "demand": noisy.astype(int),
        }))
    return pd.concat(frames, ignore_index=True)


def simulate(catalog: pd.DataFrame, demand: pd.DataFrame) -> tuple[list[Batch], pd.DataFrame]:
    """Run the day-by-day inventory simulation for every medicine."""
    all_batches: list[Batch] = []
    stock_log = []

    for _, med in catalog.iterrows():
        mid = int(med.medicine_id)
        series = demand[demand.medicine_id == mid].sort_values("date").reset_index(drop=True)
        dates = series["date"].to_numpy()
        demands = series["demand"].to_numpy()
        shelf_days = int(med.shelf_life_months * 30)

        batches: list[Batch] = []
        batch_no = 0

        for i, (today, want) in enumerate(zip(dates, demands)):
            today = pd.Timestamp(today)

            # --- expire anything that reached its date ---
            for b in batches:
                if not b.expired and today >= b.expires_on:
                    b.expired = True
                    b.wasted = b.qty_remaining
                    b.qty_remaining = 0

            live = [b for b in batches if not b.expired and b.qty_remaining > 0]
            on_hand = sum(b.qty_remaining for b in live)

            # --- weekly reorder review ---
            if i % REVIEW_EVERY_DAYS == 0:
                lookback = demands[max(0, i - 90):i + 1]
                avg_daily = float(lookback.mean()) if len(lookback) else 1.0
                avg_daily = max(avg_daily, 0.05)
                if on_hand < avg_daily * REORDER_COVER_DAYS:
                    wanted = avg_daily * TARGET_COVER_DAYS

                    # Bulk-discount over-ordering: a classic source of waste.
                    if rng.random() < P_BULK_BUY:
                        wanted *= rng.uniform(*BULK_MULTIPLIER)

                    # Stock is only sold in whole packs, rounded up. For a slow
                    # mover the minimum pack can be more than a year of demand.
                    pack = int(med.pack_size)
                    order_qty = int(max(1, np.ceil(wanted / pack)) * pack)

                    # Suppliers ship stock already part-way through its life,
                    # and sometimes clear short-dated stock onto the pharmacy.
                    if rng.random() < P_SHORT_DATED:
                        life_frac = rng.uniform(*SHORT_DATED_LIFE)
                    else:
                        life_frac = rng.uniform(*NORMAL_LIFE)
                    remaining_life = max(20, int(shelf_days * life_frac))
                    batch_no += 1
                    b = Batch(
                        batch_id=f"B{mid:03d}-{batch_no:03d}",
                        medicine_id=mid,
                        received_on=today,
                        expires_on=today + pd.Timedelta(days=remaining_life),
                        qty_received=order_qty,
                        qty_remaining=order_qty,
                        unit_cost=float(med.unit_cost),
                        shelf_life_days=remaining_life,
                    )
                    batches.append(b)
                    live.append(b)
                    on_hand += order_qty

            # --- dispense FEFO: consume the batch that expires soonest first ---
            remaining_want = int(want)
            for b in sorted(live, key=lambda x: x.expires_on):
                if remaining_want <= 0:
                    break
                take = min(b.qty_remaining, remaining_want)
                b.qty_remaining -= take
                remaining_want -= take

            # --- record each live batch's level for later feature extraction ---
            for b in batches:
                if not b.expired:
                    b.history[today] = b.qty_remaining

            stock_log.append({
                "date": today, "medicine_id": mid,
                "demand": int(want), "fulfilled": int(want) - remaining_want,
                "stock_after": sum(x.qty_remaining for x in batches if not x.expired),
            })

        all_batches.extend(batches)

    return all_batches, pd.DataFrame(stock_log)


def build_features(
    batches: list[Batch], stock_log: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    """Observe each batch at several points in its life and label the outcome."""
    cat = catalog.set_index("medicine_id")
    # Per-medicine daily demand series, for rolling usage-rate features.
    usage = (
        stock_log.pivot_table(index="date", columns="medicine_id", values="demand", aggfunc="sum")
        .sort_index()
    )
    roll = {w: usage.rolling(w, min_periods=1).mean() for w in (7, 30, 90)}

    rows = []
    for b in batches:
        # Only batches that actually reached their expiry date have a known
        # outcome. Batches still live at the end of the simulation are censored
        # and would teach the model a false "no waste" label.
        if not b.expired:
            continue

        med = cat.loc[b.medicine_id]
        life = (b.expires_on - b.received_on).days
        if life < 30:
            continue

        for days_before in OBSERVE_DAYS_BEFORE:
            obs = b.expires_on - pd.Timedelta(days=days_before)
            if obs < b.received_on:
                continue  # batch had not arrived yet at this point
            if obs not in b.history or obs not in usage.index:
                continue

            qty_left = b.history[obs]
            if qty_left <= 0:
                continue  # batch already sold out; nothing left to be at risk

            u7 = float(roll[7].at[obs, b.medicine_id])
            u30 = float(roll[30].at[obs, b.medicine_id])
            u90 = float(roll[90].at[obs, b.medicine_id])
            days_left = (b.expires_on - obs).days
            total_stock = int(
                stock_log[(stock_log.date == obs) & (stock_log.medicine_id == b.medicine_id)]
                .stock_after.iloc[0]
            )

            rows.append({
                "batch_id": b.batch_id,
                "medicine_id": b.medicine_id,
                "atc_group": med.atc_group,
                "observed_on": obs,
                "expires_on": b.expires_on,
                # --- features ---
                "days_to_expiry": days_left,
                "qty_remaining": qty_left,
                "qty_received": b.qty_received,
                "pct_batch_remaining": qty_left / b.qty_received,
                "total_stock_all_batches": total_stock,
                "usage_rate_7d": round(u7, 3),
                "usage_rate_30d": round(u30, 3),
                "usage_rate_90d": round(u90, 3),
                "usage_trend": round(u30 / u90, 3) if u90 > 0 else 1.0,
                # Days of stock left at current speed. The single strongest signal.
                "days_of_cover": round(qty_left / u30, 2) if u30 > 0 else 9999.0,
                # <1 means demand cannot clear this batch before it expires.
                "cover_ratio": round((u30 * days_left) / qty_left, 3) if qty_left > 0 else 0.0,
                "shelf_life_days": b.shelf_life_days,
                "pct_life_remaining": round(days_left / b.shelf_life_days, 3),
                "unit_cost": b.unit_cost,
                "month": obs.month,
                "prescription_required": int(med.prescription_required),
                # --- labels ---
                "units_wasted": b.wasted,
                "will_expire_unused": int(b.wasted > 0),
                "value_at_risk": round(b.wasted * b.unit_cost, 2),
            })

    return pd.DataFrame(rows)


def main() -> int:
    global rng
    ensure_dirs()
    print("=" * 74)
    print(" MediSmart - Phase 2, step 3: building the expiry-risk dataset")
    print("=" * 74)

    sales_path = PROCESSED / "sales_daily_clean.csv"
    if not sales_path.exists():
        print(f"  ERROR: {sales_path} not found. Run scripts/prepare_datasets.py first.")
        return 1

    sales = pd.read_csv(sales_path, parse_dates=["date"])
    catalog = build_catalog()
    print(f"\n  catalog: {len(catalog)} medicines across {catalog.atc_group.nunique()} ATC groups")
    print(f"  driving demand from {len(sales):,} real days of pharmacy sales")
    print(f"  simulating {N_STORES} stores\n")

    all_features, all_batches_rows = [], []
    totals = {"batches": 0, "expired": 0, "wasted_batches": 0,
              "units_received": 0, "units_wasted": 0, "value": 0.0}
    first_store_log = None

    print(f"  {'store':<7}{'batches':>9}{'expired':>9}{'w/ waste':>10}{'units lost':>12}{'waste %':>9}{'rows':>7}")
    for store in range(1, N_STORES + 1):
        # Each store gets its own random stream: different demand noise,
        # different ordering luck, different short-dated deliveries.
        rng = np.random.default_rng(SEED + store)

        demand = split_demand(sales, catalog)
        batches, stock_log = simulate(catalog, demand)
        feats = build_features(batches, stock_log, catalog)

        expired = [b for b in batches if b.expired]
        wasted = [b for b in expired if b.wasted > 0]
        units_recv = sum(b.qty_received for b in batches)
        units_lost = sum(b.wasted for b in batches)
        value = sum(b.wasted * b.unit_cost for b in batches)

        if not feats.empty:
            feats = feats.copy()
            feats.insert(0, "store_id", store)
            # Make batch ids unique across stores so grouped CV stays correct.
            feats["batch_id"] = f"S{store}-" + feats["batch_id"].astype(str)
            all_features.append(feats)

        for b in batches:
            all_batches_rows.append({
                "store_id": store, "batch_id": f"S{store}-{b.batch_id}",
                "medicine_id": b.medicine_id,
                "received_on": b.received_on.date(), "expires_on": b.expires_on.date(),
                "qty_received": b.qty_received, "qty_remaining": b.qty_remaining,
                "unit_cost": b.unit_cost, "wasted": b.wasted, "expired": b.expired,
            })

        totals["batches"] += len(batches)
        totals["expired"] += len(expired)
        totals["wasted_batches"] += len(wasted)
        totals["units_received"] += units_recv
        totals["units_wasted"] += units_lost
        totals["value"] += value
        if first_store_log is None:
            first_store_log = stock_log

        pct = units_lost / units_recv * 100 if units_recv else 0
        print(f"  {store:<7}{len(batches):>9,}{len(expired):>9,}{len(wasted):>10,}"
              f"{units_lost:>12,}{pct:>8.2f}%{len(feats):>7,}")

    if not all_features:
        print("  ERROR: no observations produced.")
        return 1

    features = pd.concat(all_features, ignore_index=True)

    waste_pct = totals["units_wasted"] / totals["units_received"] * 100
    print("-" * 74)
    print(f"  TOTAL  {totals['batches']:>9,}{totals['expired']:>9,}{totals['wasted_batches']:>10,}"
          f"{totals['units_wasted']:>12,}{waste_pct:>8.2f}%{len(features):>7,}")
    print(f"\n  value wasted across the chain: Rs {totals['value']:,.0f}")
    print(f"  (industry benchmark for pharmacy expiry write-off is roughly 3-5%)")

    pos = int(features.will_expire_unused.sum())
    print(f"\n  labelled rows: {len(features):,}")
    print(f"    will_expire_unused = 1 : {pos:,} ({pos/len(features)*100:.1f}%)")
    print(f"    will_expire_unused = 0 : {len(features)-pos:,} ({(len(features)-pos)/len(features)*100:.1f}%)")
    print(f"    unique batches: {features.batch_id.nunique():,} "
          f"(group by this in CV - rows from one batch are not independent)")

    catalog.to_csv(PROCESSED / "medicine_catalog.csv", index=False)
    features.to_csv(PROCESSED / "expiry_risk_dataset.csv", index=False)
    pd.DataFrame(all_batches_rows).to_csv(PROCESSED / "batch_ledger.csv", index=False)
    first_store_log.to_csv(PROCESSED / "inventory_simulation_log.csv", index=False)

    summary = {
        "stores_simulated": N_STORES,
        "medicines": len(catalog),
        "batches_total": totals["batches"],
        "batches_expired": totals["expired"],
        "batches_with_waste": totals["wasted_batches"],
        "units_received": int(totals["units_received"]),
        "units_wasted": int(totals["units_wasted"]),
        "waste_pct": round(waste_pct, 2),
        "value_wasted_rupees": round(totals["value"], 2),
        "observations": len(features),
        "unique_batches_in_dataset": int(features.batch_id.nunique()),
        "positive_rate": round(pos / len(features), 4),
        "note": "Consumption is real pharmacy POS data; batch and expiry metadata is simulated.",
    }
    (REPORTS / "phase2_expiry_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n  wrote medicine_catalog.csv, expiry_risk_dataset.csv,")
    print("        batch_ledger.csv, inventory_simulation_log.csv")
    print("  Next: python scripts/eda.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
