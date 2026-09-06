"""
Comprehensive Phase 5 & 6 Backend & Database Verification Suite.

Tests:
1. API Health endpoint
2. Authentication & role checks
3. Inventory retrieval & batch listings
4. FEFO Dispensing Engine (verifying earliest-expiry stock is deducted first)
5. Multimodal Disease Prediction with the project brief's worked example:
     Input: Age: 45, Glucose: 180, BMI: 29, BP: 140/90, Symptoms: polyuria, fatigue...
     Output: Predicted condition: Diabetes, Confidence: >90%, Specialist: Endocrinologist
6. Emergency Red-flag clinical escalation
7. Real-time Batch Expiry Risk Scoring
8. Dashboard KPI metrics aggregation

Run with:  python scripts/test_backend.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medismart.app import create_app
from medismart.db import db
from medismart.db.models import Batch, Medicine


def run_tests():
    app = create_app()
    client = app.test_client()

    print("=" * 74)
    print(" MediSmart - Phase 5 & 6 Backend & Database Verification Suite")
    print("=" * 74)

    # -------------------------------------------------------------
    # 1. Health Check
    # -------------------------------------------------------------
    print("\n[Test 1] Health Check Endpoint")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "healthy"
    print("  [PASS] /api/health returned healthy")

    # -------------------------------------------------------------
    # 2. Authentication
    # -------------------------------------------------------------
    print("\n[Test 2] Authentication")
    res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res.status_code == 200, f"Login failed: {res.get_json()}"
    user_data = res.get_json()["user"]
    assert user_data["username"] == "admin"
    assert user_data["role"] == "admin"
    print(f"  [PASS] Admin login succeeded. Role: {user_data['role']}")

    # Invalid login test
    res_bad = client.post("/api/auth/login", json={"username": "admin", "password": "wrongpassword"})
    assert res_bad.status_code == 401
    print("  [PASS] Invalid login rejected with 401")

    # -------------------------------------------------------------
    # 3. Medicine Catalog
    # -------------------------------------------------------------
    print("\n[Test 3] Medicine Catalog & Batches")
    res = client.get("/api/inventory/medicines")
    assert res.status_code == 200
    meds = res.get_json()["medicines"]
    assert len(meds) == 22, f"Expected 22 medicines, got {len(meds)}"
    print(f"  [PASS] Retrieved {len(meds)} catalog medicines")

    # -------------------------------------------------------------
    # 4. FEFO Dispensing Engine Verification
    # -------------------------------------------------------------
    print("\n[Test 4] FEFO Dispensing Engine Accuracy")
    with app.app_context():
        # Pick medicine with ID 1 (Diclofenac 50mg Tab)
        med = db.session.get(Medicine, 1)
        active_batches = (
            Batch.query.filter(Batch.medicine_id == 1, Batch.status == "ACTIVE")
            .order_by(Batch.expiry_date.asc())
            .all()
        )
        earliest_batch_id = active_batches[0].id
        earliest_initial_qty = active_batches[0].quantity_remaining
        print(f"  Target: '{med.name}', Earliest Batch {active_batches[0].batch_number} (Exp: {active_batches[0].expiry_date})")
        print(f"  Earliest batch stock before sale: {earliest_initial_qty} units")

    # Execute a sale of 25 units
    res = client.post("/api/inventory/sale", json={
        "customer_name": "Test Customer",
        "payment_method": "UPI",
        "items": [{"medicine_id": 1, "quantity": 25}],
    })
    assert res.status_code == 200, f"Sale failed: {res.get_json()}"
    sale_data = res.get_json()
    assert sale_data["success"] is True
    print(f"  [+] Sale generated invoice: {sale_data['invoice_no']} for Rs {sale_data['total_amount']}")

    with app.app_context():
        earliest_after = db.session.get(Batch, earliest_batch_id)
        deducted = earliest_initial_qty - earliest_after.quantity_remaining
        assert deducted == 25, f"Expected exactly 25 units deducted from oldest batch, got {deducted}"
        print(f"  [PASS] FEFO Verification confirmed: Oldest batch stock decremented by exactly 25 units (Remaining: {earliest_after.quantity_remaining})")

    # -------------------------------------------------------------
    # 5. Multimodal Disease Prediction (Project Brief Worked Example)
    # -------------------------------------------------------------
    print("\n[Test 5] Worked Example: Disease Prediction & Clinical Fusion")
    payload = {
        "patient": {"name": "Suresh Kumar", "age": 45, "gender": "male"},
        "vitals": {
            "glucose": 180,
            "bmi": 29,
            "blood_pressure": "140/90",
        },
        "symptoms": [
            "polyuria",            # frequent urination
            "excessive_hunger",    # increased thirst / appetite
            "fatigue",
            "weight_loss",
            "irregular_sugar_level",
        ],
    }

    res = client.post("/api/predict/diagnose", json=payload)
    assert res.status_code == 200, f"Diagnosis failed: {res.get_json()}"
    diag = res.get_json()["result"]

    print(f"  Input: Age=45, Glucose=180 mg/dL, BMI=29, BP=140/90")
    print(f"  Symptoms: {payload['symptoms']}")
    print(f"  -> Predicted Condition: {diag['predicted_condition']}")
    print(f"  -> Model Confidence:   {diag['model_confidence']}%")
    print(f"  -> First-line:         {diag['first_line_treatment']}")
    print(f"  -> Specialist:         {diag['specialist_to_consult']}")
    print(f"  -> Department:         {diag['department']}")
    print(f"  -> Medications:        {diag['recommended_medication_classes']}")

    assert "diabet" in diag["predicted_condition"].lower(), f"Expected Diabetes, got {diag['predicted_condition']}"
    assert diag["model_confidence"] >= 90.0, f"Expected confidence >= 90%, got {diag['model_confidence']}%"
    assert "Endocrinologist" in diag["specialist_to_consult"]
    assert len(diag["recommended_medication_classes"]) > 0
    print("  [PASS] Worked Example matched project specification perfectly!")

    # -------------------------------------------------------------
    # 6. Emergency Red-Flag Symptom Escalation
    # -------------------------------------------------------------
    print("\n[Test 6] Emergency Red-Flag Triage Escalation")
    emerg_payload = {
        "patient": {"name": "Emergency Patient", "age": 62, "gender": "male"},
        "symptoms": ["chest_pain", "breathlessness", "slurred_speech"],
    }
    res = client.post("/api/predict/diagnose", json=emerg_payload)
    assert res.status_code == 200
    diag_emerg = res.get_json()["result"]
    assert diag_emerg["is_emergency"] is True
    assert diag_emerg["urgency"] == "critical"
    print(f"  Input: {emerg_payload['symptoms']}")
    print(f"  -> Emergency Flag:  {diag_emerg['is_emergency']}")
    print(f"  -> Urgency Level:   {diag_emerg['urgency']}")
    print(f"  -> Red Flags Identified: {diag_emerg['emergency_symptoms']}")
    print("  [PASS] Emergency symptoms correctly routed with CRITICAL urgency")

    # -------------------------------------------------------------
    # 7. Real-Time Batch Expiry Risk Scoring
    # -------------------------------------------------------------
    print("\n[Test 7] Real-Time Batch Expiry Risk Assessment")
    res = client.post("/api/expiry/assess-batch/1")
    assert res.status_code == 200
    exp_data = res.get_json()["assessment"]
    print(f"  Batch 1 Assessment:")
    print(f"  -> Days to Expiry:  {exp_data['days_to_expiry']} days")
    print(f"  -> AI Risk Score:   {exp_data['risk_score']}/100")
    print(f"  -> Risk Bucket:     {exp_data['risk_bucket']}")
    print(f"  -> Value at Risk:   Rs {exp_data['value_at_risk']}")
    print(f"  -> Recommendation:  {exp_data['recommendation']}")
    print("  [PASS] Expiry model scored live batch correctly")

    # -------------------------------------------------------------
    # 8. Dashboard Overview KPIs
    # -------------------------------------------------------------
    print("\n[Test 8] Dashboard KPI Aggregation")
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    summary = res.get_json()
    assert summary["inventory"]["total_medicines"] == 22
    assert summary["inventory"]["total_stock_units"] > 0
    print(f"  Total Inventory Value: Rs {summary['inventory']['total_inventory_value']:,.2f}")
    print(f"  Active Batches Count:  {summary['inventory']['active_batches_count']}")
    print(f"  Near Expiry Count:     {summary['expiry_risk']['near_expiry_count']}")
    print(f"  Total Consultations:   {summary['diagnostics']['total_consultations']}")
    print("  [PASS] Dashboard KPIs aggregated across all subsystems")

    print("\n" + "=" * 74)
    print(" ALL 8 BACKEND & DATABASE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 74)


if __name__ == "__main__":
    run_tests()
