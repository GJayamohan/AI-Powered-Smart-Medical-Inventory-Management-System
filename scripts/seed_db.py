"""
Database initialization and seeding script.

Creates database tables and populates:
1. Default system users (Admin, Pharmacist, Cashier)
2. Pharmacy suppliers
3. 22 standard medicines from `data/processed/medicine_catalog.csv`
4. Realistic active inventory batches with varying shelf lives (including near-expiry batches for risk testing)

Run with:  python scripts/seed_db.py
"""

from __future__ import annotations

import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medismart.db import db, get_default_db_uri
from medismart.db.models import Batch, Medicine, Supplier, User
from medismart.utils.paths import PROCESSED


def create_app_context():
    app = Flask("medismart_seeder")
    app.config["SQLALCHEMY_DATABASE_URI"] = get_default_db_uri()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    return app


def seed_database():
    app = create_app_context()

    with app.app_context():
        print("  Dropping and recreating all database tables...", flush=True)
        db.drop_all()
        db.create_all()

        # -------------------------------------------------------------
        # 1. Seed Users
        # -------------------------------------------------------------
        print("  Seeding default users...", flush=True)
        users = [
            User(
                username="admin",
                email="admin@medismart.local",
                role="admin",
                full_name="System Administrator",
            ),
            User(
                username="pharmacist",
                email="pharmacist@medismart.local",
                role="pharmacist",
                full_name="Lead Pharmacist",
            ),
            User(
                username="cashier",
                email="cashier@medismart.local",
                role="cashier",
                full_name="Pharmacy Cashier",
            ),
        ]
        users[0].set_password("admin123")
        users[1].set_password("pharma123")
        users[2].set_password("cash123")

        db.session.add_all(users)
        db.session.commit()
        print("    [+] Created 3 users (admin, pharmacist, cashier)")

        # -------------------------------------------------------------
        # 2. Seed Suppliers
        # -------------------------------------------------------------
        print("  Seeding suppliers...", flush=True)
        suppliers = [
            Supplier(
                name="Apex Healthcare Distributors",
                contact_person="Ramesh Gupta",
                email="orders@apexhealth.in",
                phone="+91-98200-11223",
                address="Industrial Area Phase 2, Mumbai, MH",
            ),
            Supplier(
                name="MedLife Pharma Logistics",
                contact_person="Priya Sharma",
                email="support@medlifelogistics.com",
                phone="+91-98111-44556",
                address="Sector 18, Gurugram, HR",
            ),
            Supplier(
                name="Global Formulations Ltd",
                contact_person="Dr. John Samuel",
                email="supply@globalformulations.org",
                phone="+91-94440-77889",
                address="Whitefield Tech Zone, Bengaluru, KA",
            ),
            Supplier(
                name="Sun Med Direct",
                contact_person="Anil Verma",
                email="orders@sunmeddirect.in",
                phone="+91-97000-33445",
                address="MIDC Kurla, Mumbai, MH",
            ),
        ]
        db.session.add_all(suppliers)
        db.session.commit()
        print(f"    [+] Created {len(suppliers)} suppliers")

        # -------------------------------------------------------------
        # 3. Seed Medicines
        # -------------------------------------------------------------
        print("  Seeding catalog medicines...", flush=True)
        catalog_path = PROCESSED / "medicine_catalog.csv"
        med_objs = []
        
        category_map = {
            "M01AB": "Anti-inflammatory (Acetic acid derivatives)",
            "M01AE": "Anti-inflammatory (Propionic acid derivatives)",
            "N02BA": "Analgesics & Antipyretics (Salicylic acid)",
            "N02BE": "Analgesics & Antipyretics (Anilides/Paracetamol)",
            "N05B": "Psycholeptics (Anxiolytics)",
            "N05C": "Psycholeptics (Hypnotics and sedatives)",
            "R03": "Respiratory (Anti-asthmatics/COPD)",
            "R06": "Respiratory (Antihistamines for systemic use)",
        }

        with open(catalog_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                atc = row["atc_group"].strip()
                med = Medicine(
                    id=int(row["medicine_id"]),
                    name=row["name"].strip(),
                    atc_group=atc,
                    category=category_map.get(atc, "General Medicine"),
                    unit_cost=float(row["unit_cost"]),
                    unit_price=float(row["unit_price"]),
                    pack_size=int(row["pack_size"]),
                    prescription_required=str(row["prescription_required"]).strip().lower() in ("true", "1"),
                    shelf_life_months=int(row["shelf_life_months"]),
                    min_stock_alert=int(float(row["pack_size"]) * 0.5) if float(row["pack_size"]) >= 50 else 30,
                    description=f"Standard pharmaceutical preparation in ATC class {atc}. Pack size: {row['pack_size']} units.",
                )
                med_objs.append(med)
                db.session.add(med)

        db.session.commit()
        print(f"    [+] Created {len(med_objs)} medicines from catalog")

        # -------------------------------------------------------------
        # 4. Seed Realistic Multi-Batch Inventory
        # -------------------------------------------------------------
        print("  Seeding realistic inventory batches with FEFO dynamics...", flush=True)
        today = date.today()
        batches = []
        batch_counter = 1001

        # For each medicine, create 2 to 4 batches across different expiry horizons
        # e.g., Batch 1: Expiring in 25 days (Near-expiry / High Risk test)
        #       Batch 2: Expiring in 180 days (Normal active stock)
        #       Batch 3: Expiring in 540 days (Fresh stock)
        for med in med_objs:
            # Batch 1: Near expiry (15 to 45 days remaining)
            b1_days = 25 if med.id % 2 == 0 else 40
            b1_qty = int(med.pack_size * 1.5)
            b1 = Batch(
                batch_number=f"BCH-2024-{batch_counter}",
                medicine_id=med.id,
                supplier_id=((batch_counter % len(suppliers)) + 1),
                mfg_date=today - timedelta(days=(med.shelf_life_months * 30) - b1_days),
                expiry_date=today + timedelta(days=b1_days),
                quantity_received=b1_qty * 2,
                quantity_remaining=b1_qty,
                unit_cost=med.unit_cost,
                status="ACTIVE",
                risk_score=78.5 if b1_days <= 30 else 55.0,
                risk_bucket="CRITICAL" if b1_days <= 30 else "HIGH",
                value_at_risk=round(b1_qty * med.unit_cost, 2),
                last_risk_assessed_at=datetime.utcnow(),
            )
            batches.append(b1)
            batch_counter += 1

            # Batch 2: Medium shelf-life (150 to 240 days)
            b2_days = 180
            b2_qty = int(med.pack_size * 3)
            b2 = Batch(
                batch_number=f"BCH-2025-{batch_counter}",
                medicine_id=med.id,
                supplier_id=((batch_counter % len(suppliers)) + 1),
                mfg_date=today - timedelta(days=120),
                expiry_date=today + timedelta(days=b2_days),
                quantity_received=b2_qty,
                quantity_remaining=b2_qty,
                unit_cost=med.unit_cost,
                status="ACTIVE",
                risk_score=15.0,
                risk_bucket="SAFE",
                value_at_risk=0.0,
                last_risk_assessed_at=datetime.utcnow(),
            )
            batches.append(b2)
            batch_counter += 1

            # Batch 3: Fresh stock (360 to 700 days)
            b3_days = int(med.shelf_life_months * 25)
            b3_qty = int(med.pack_size * 4)
            b3 = Batch(
                batch_number=f"BCH-2026-{batch_counter}",
                medicine_id=med.id,
                supplier_id=((batch_counter % len(suppliers)) + 1),
                mfg_date=today - timedelta(days=30),
                expiry_date=today + timedelta(days=b3_days),
                quantity_received=b3_qty,
                quantity_remaining=b3_qty,
                unit_cost=med.unit_cost,
                status="ACTIVE",
                risk_score=5.0,
                risk_bucket="SAFE",
                value_at_risk=0.0,
                last_risk_assessed_at=datetime.utcnow(),
            )
            batches.append(b3)
            batch_counter += 1

        db.session.add_all(batches)
        db.session.commit()
        print(f"    [+] Created {len(batches)} active batches across {len(med_objs)} medicines")

        # Summary check
        total_units = sum(b.quantity_remaining for b in batches)
        total_val = sum(b.quantity_remaining * b.unit_cost for b in batches)
        print(f"\n  [OK] Database seeded successfully!")
        print(f"       Total inventory units: {total_units:,}")
        print(f"       Total stock value:     Rs {total_val:,.2f}")
        print(f"       Database file:         {get_default_db_uri()}")


if __name__ == "__main__":
    seed_database()
