"""
Frontend and Web UI Verification Suite.

Tests:
1. Web Routes HTTP 200 OK:
   - GET / (Dashboard)
   - GET /inventory (Inventory & Batches)
   - GET /expiry (Expiry Risk AI & Near-Expiry Batches)
   - GET /predict (Disease Prediction & Medication Recommendation)
   - GET /pos (Point of Sale & FEFO Dispensing)
   - GET /sales (Sales History Ledger)
   - GET /login (Authentication Portal)
2. Static Assets HTTP 200 OK:
   - /static/css/app.css
   - /static/js/app.js
3. AI Pharmacy Chatbot queries:
   - Medicine stock query
   - Expiry alert query
   - Clinical disease query
   - FEFO explanation query
   - Worked example query

Run with:  python scripts/test_frontend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medismart.app import create_app


def run_frontend_tests():
    app = create_app()
    client = app.test_client()

    print("=" * 74)
    print(" MediSmart - Phase 7 Frontend & Web UI Verification Suite")
    print("=" * 74)

    # 1. Test All HTML Page Routes
    pages = [
        ("/", "Dashboard"),
        ("/inventory", "Inventory Management"),
        ("/expiry", "Expiry Risk AI"),
        ("/predict", "Disease Diagnosis AI"),
        ("/pos", "Point of Sale & FEFO"),
        ("/sales", "Sales & Invoices"),
        ("/login", "Login Portal"),
    ]

    print("\n[Suite 1] Web HTML Routes")
    for path, name in pages:
        res = client.get(path)
        assert res.status_code == 200, f"Page {path} ({name}) failed with {res.status_code}"
        assert b"MediSmart" in res.data or b"Dashboard" in res.data or b"Login" in res.data
        print(f"  [PASS] GET {path:12s} -> HTTP 200 OK ({name})")

    # 2. Test Static Assets
    print("\n[Suite 2] Static Assets")
    for asset in ["/static/css/app.css", "/static/js/app.js"]:
        res = client.get(asset)
        assert res.status_code == 200, f"Asset {asset} failed with {res.status_code}"
        assert len(res.data) > 100
        print(f"  [PASS] GET {asset} -> HTTP 200 OK ({len(res.data):,} bytes)")

    # 3. Test AI Pharmacy Chatbot Conversational Queries
    print("\n[Suite 3] AI Pharmacy Chatbot Knowledge Assistant")
    queries = [
        ("Do we have Paracetamol in stock?", "medicine_stock"),
        ("Which batches are near expiry?", "expiry_summary"),
        ("What doctor should I consult for Diabetes?", "disease_treatment"),
        ("Explain how FEFO dispensing works", "fefo_explanation"),
        ("Show me the worked diagnostic example", "worked_example"),
    ]

    for q, expected_intent in queries:
        res = client.post("/api/chat/message", json={"message": q})
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert len(data["reply"]) > 20
        assert data["intent"] == expected_intent, f"Expected intent {expected_intent}, got {data['intent']}"
        print(f"  [PASS] Query: '{q[:35]}...'")
        safe_reply = data['reply'][:65].replace(chr(10), ' ').encode('ascii', 'replace').decode('ascii')
        print(f"         Intent: {data['intent']} -> Reply: {safe_reply}...")

    print("\n" + "=" * 74)
    print(" ALL FRONTEND & WEB UI TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 74)


if __name__ == "__main__":
    run_frontend_tests()
