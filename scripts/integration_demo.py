"""
Phase 8: End-to-End System Integration & Clinical Demonstration Script.

Simulates the complete patient and pharmacy lifecycle across all subsystems:
1. Patient Consultation & Multimodal Disease Prediction (Project Brief Worked Example)
2. Treatment Knowledge Base lookup & Specialist Referral
3. Point of Sale & First-Expiry-First-Out (FEFO) Inventory Dispensing
4. Batch Expiry Risk AI Assessment & Value-at-Risk Calculation
5. 14-Day LSTM Therapeutic Demand Forecasting
6. Conversational AI Assistant Interaction

Run with:  python scripts/integration_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medismart.app import create_app
from medismart.db import db
from medismart.db.models import Batch, Medicine, PredictionLog, Sale


def run_integration_pipeline():
    app = create_app()
    client = app.test_client()

    print("=" * 78)
    print(" MEDISMART: AI-POWERED MEDICAL INVENTORY & CLINICAL DECISION SYSTEM")
    print(" PHASE 8: COMPLETE END-TO-END INTEGRATION DEMONSTRATION")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # WORKFLOW 1: MULTIMODAL PATIENT CONSULTATION & DISEASE PREDICTION
    # (Exact scenario specified in project brief)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(" [WORKFLOW 1] PATIENT CLINICAL CONSULTATION & MULTIMODAL AI DIAGNOSIS")
    print("-" * 78)

    patient_payload = {
        "patient": {
            "name": "Sunita Rao",
            "age": 45,
            "gender": "female",
        },
        "vitals": {
            "glucose": 180.0,            # High blood glucose (Normal: 70-125)
            "bmi": 29.0,                 # Overweight (Normal: 18.5-24.9)
            "blood_pressure": "140/90",  # Stage 1 Hypertension
        },
        "symptoms": [
            "polyuria",                  # frequent urination
            "excessive_hunger",          # increased thirst / appetite
            "fatigue",                   # persistent tiredness
            "weight_loss",               # unexplained weight loss
            "irregular_sugar_level",     # glycemic fluctuations
        ],
    }

    print(" Patient Input Received:")
    print(f"   Name:           {patient_payload['patient']['name']} (Age: {patient_payload['patient']['age']}, {patient_payload['patient']['gender']})")
    print(f"   Lab Vitals:     Glucose: {patient_payload['vitals']['glucose']} mg/dL | BMI: {patient_payload['vitals']['bmi']} kg/m2 | BP: {patient_payload['vitals']['blood_pressure']} mmHg")
    print(f"   Symptoms:       {', '.join(patient_payload['symptoms'])}")
    print("\n -> Calling AI Engine Clinical Decision Support (/api/predict/diagnose)...")

    res = client.post("/api/predict/diagnose", json=patient_payload)
    assert res.status_code == 200, f"Diagnosis endpoint failed: {res.get_json()}"
    diag = res.get_json()["result"]

    print("\n [AI System Output Generated]:")
    print(f"   ===================================================================")
    print(f"   PREDICTED CONDITION:      {diag['predicted_condition']}")
    print(f"   MODEL CONFIDENCE:         {diag['model_confidence']}%")
    print(f"   CLINICAL URGENCY:         {diag['urgency'].upper()}")
    print(f"   RECOMMENDED MEDICATIONS:  {', '.join(diag['recommended_medication_classes'])}")
    print(f"   FIRST-LINE REGIMEN:       {diag['first_line_treatment']}")
    print(f"   SPECIALIST DOCTOR:        {diag['specialist_to_consult']} ({diag['department']})")
    print(f"   KEY PRECAUTIONS:          {', '.join(diag['precautions'][:3])}")
    print(f"   AUDIT LOG ID:             Consultation #{diag.get('log_id', 'N/A')}")
    print(f"   ===================================================================")

    # -------------------------------------------------------------------------
    # WORKFLOW 2: FEFO PRESCRIPTION DISPENSING & POINT OF SALE (POS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(" [WORKFLOW 2] FIRST-EXPIRY-FIRST-OUT (FEFO) PRESCRIPTION DISPENSING")
    print("-" * 78)

    # Patient is prescribed Paracetamol (N02BE) and Diclofenac (M01AB)
    # We will sell 30 units of Diclofenac 50mg Tab (ID: 1)
    target_med_id = 1

    with app.app_context():
        med = db.session.get(Medicine, target_med_id)
        active_batches = (
            Batch.query.filter(
                Batch.medicine_id == target_med_id,
                Batch.status == "ACTIVE",
                Batch.quantity_remaining > 0
            )
            .order_by(Batch.expiry_date.asc())
            .all()
        )
        print(f" Target Medicine: {med.name} (Unit Price: Rs {med.unit_price:.2f})")
        print(f" Available Batches in Pharmacy Depot:")
        for b in active_batches:
            print(f"   - Batch {b.batch_number}: {b.quantity_remaining} units remaining | Expires: {b.expiry_date} ({b.days_to_expiry} days left)")

        earliest_batch = active_batches[0]
        initial_stock = earliest_batch.quantity_remaining

    # Step A: Request FEFO preview for 30 units
    print("\n -> Requesting FEFO Batch Allocation Plan for 30 units (/api/inventory/fefo-preview)...")
    preview_res = client.post("/api/inventory/fefo-preview", json={"medicine_id": target_med_id, "quantity": 30})
    assert preview_res.status_code == 200
    plan = preview_res.get_json()

    print(" [FEFO Allocation Plan]:")
    for alloc in plan["allocation"]:
        print(f"   * Allocate {alloc['take_quantity']} units from Batch {alloc['batch_number']} (Exp: {alloc['expiry_date']})")
    print(f"   Subtotal: Rs {plan['total_subtotal']:.2f}")

    # Step B: Process Checkout & Sale
    print("\n -> Executing Atomic Checkout Transaction (/api/inventory/sale)...")
    sale_payload = {
        "customer_name": patient_payload["patient"]["name"],
        "customer_phone": "+91-98765-43210",
        "payment_method": "UPI",
        "discount": 0.0,
        "items": [{"medicine_id": target_med_id, "quantity": 30}],
    }
    sale_res = client.post("/api/inventory/sale", json=sale_payload)
    assert sale_res.status_code == 200
    sale_data = sale_res.get_json()

    print(" [Sale Completed & Invoice Generated]:")
    print(f"   Invoice Number:  {sale_data['invoice_no']}")
    print(f"   Net Amount Paid: Rs {sale_data['net_amount']:.2f} via UPI")

    # Step C: Verify FEFO deduction
    with app.app_context():
        updated_batch = db.session.get(Batch, earliest_batch.id)
        deducted = initial_stock - updated_batch.quantity_remaining
        print(f"\n [Audit Verification]:")
        print(f"   Earliest Batch ({earliest_batch.batch_number}) Initial Stock: {initial_stock} units")
        print(f"   Earliest Batch Stock After Sale: {updated_batch.quantity_remaining} units")
        print(f"   Units Deducted: {deducted} units (Exact match with requested 30 units!)")
        assert deducted == 30, "FEFO did not deduct from oldest batch!"

    # -------------------------------------------------------------------------
    # WORKFLOW 3: REAL-TIME AI EXPIRY RISK & VALUE-AT-RISK ASSESSMENT
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(" [WORKFLOW 3] AI EXPIRY RISK SCORING & WASTAGE PREVENTION (MODEL 2b)")
    print("-" * 78)

    print(" -> Scoring high-risk batch with Model 2b (/api/expiry/assess-batch/1)...")
    exp_res = client.post("/api/expiry/assess-batch/1")
    assert exp_res.status_code == 200
    batch_risk = exp_res.get_json()["assessment"]

    print(" [Batch Expiry Risk Analysis]:")
    print(f"   Batch:                BCH-2024-1001")
    print(f"   Days to Expiry:       {batch_risk['days_to_expiry']} days")
    print(f"   Stock Cover Ratio:    {batch_risk['cover_ratio']}x")
    print(f"   AI Expiry Risk Score: {batch_risk['risk_score']}/100")
    print(f"   Risk Classification:  [{batch_risk['risk_bucket']}]")
    print(f"   Financial Value at Risk: Rs {batch_risk['value_at_risk']:.2f}")
    print(f"   AI Action Directive:  {batch_risk['recommendation']}")

    # -------------------------------------------------------------------------
    # WORKFLOW 4: THERAPEUTIC DEMAND FORECASTING (MODEL 2a LSTM)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(" [WORKFLOW 4] 14-DAY DEMAND FORECASTING (MODEL 2a LSTM)")
    print("-" * 78)

    print(" -> Forecasting daily sales for ATC group N02BE (Paracetamol) (/api/expiry/forecast/N02BE)...")
    fc_res = client.get("/api/expiry/forecast/N02BE?days=14")
    assert fc_res.status_code == 200
    fc = fc_res.get_json()

    print(f" Group: {fc['atc_group']} | Projected Daily Average: {fc['daily_average']} units/day")
    print(" Projected Daily Demand Curve (Next 7 Days):")
    for f in fc["forecast"][:7]:
        bars = "#" * int(f["projected_units"])
        print(f"   Day +{f['day_ahead']:02d}: {f['projected_units']:4.1f} units  | {bars}")

    # -------------------------------------------------------------------------
    # WORKFLOW 5: CONVERSATIONAL AI ASSISTANT INTERACTIONS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(" [WORKFLOW 5] CONVERSATIONAL AI PHARMACY ASSISTANT (CHATBOT)")
    print("-" * 78)

    chat_queries = [
        "Do we have Diclofenac in stock?",
        "Which batches are expiring soon?",
        "Which doctor should I consult for Diabetes?",
        "Explain how FEFO dispensing works",
    ]

    for q in chat_queries:
        c_res = client.post("/api/chat/message", json={"message": q})
        assert c_res.status_code == 200
        reply = c_res.get_json()["reply"]
        safe_reply = reply.replace("\n", " ").encode("ascii", "replace").decode("ascii")
        print(f"\n User Query:   '{q}'")
        print(f" AI Assistant: '{safe_reply[:120]}...'")

    # -------------------------------------------------------------------------
    # WORKFLOW 6: EXECUTIVE DASHBOARD KPI METRICS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(" [WORKFLOW 6] PHARMACY EXECUTIVE DASHBOARD KPIS")
    print("-" * 78)

    dash_res = client.get("/api/dashboard/summary")
    assert dash_res.status_code == 200
    dash = dash_res.get_json()

    print(" Live Pharmacy Metrics:")
    print(f"   Total Catalog Medicines:    {dash['inventory']['total_medicines']}")
    print(f"   Total Active Stock Units:   {dash['inventory']['total_stock_units']:,} units")
    print(f"   Total Inventory Valuation:  Rs {dash['inventory']['total_inventory_value']:,.2f}")
    print(f"   Near-Expiry Alert Batches:  {dash['expiry_risk']['near_expiry_count']}")
    print(f"   Low Stock Shortage Alerts:  {dash['inventory']['low_stock_count']}")
    print(f"   Today's Sales Revenue:      Rs {dash['sales']['today_revenue']:,.2f} ({dash['sales']['today_orders']} orders)")
    print(f"   Clinical Consultations:     {dash['diagnostics']['total_consultations']}")

    print("\n" + "=" * 78)
    print(" END-TO-END INTEGRATION DEMONSTRATION COMPLETED WITH 100% SUCCESS!")
    print("=" * 78)


if __name__ == "__main__":
    run_integration_pipeline()
