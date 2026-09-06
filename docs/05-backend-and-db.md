# Phase 5 & 6: Backend Development & Database (SQLite / MySQL ORM)

This document details the Flask backend architecture, SQLAlchemy ORM data models, First-Expiry-First-Out (FEFO) dispensing engine, centralized AI inference pipeline, and RESTful API endpoints for the **MediSmart** system.

---

## 1. Architecture Overview

MediSmart operates a modular single-server Flask backend serving both REST API endpoints and web interface views:

```
                  ┌──────────────────────────────────────────────────┐
                  │                 Flask Application                │
                  │   CORS Enabled  •  Flask-Login  •  SQLAlchemy    │
                  └─────────┬──────────────────────────────┬─────────┘
                            │                              │
             ┌──────────────┴──────────────┐   ┌───────────┴──────────┐
             │         REST API            │   │    Services Layer    │
             │  /api/auth                  │   │  • InventoryService  │
             │  /api/inventory             │   │  • AIEngine          │
             │  /api/predict               │   │  • TreatmentService  │
             │  /api/expiry                │   └───────────┬──────────┘
             │  /api/dashboard             │               │
             └──────────────┬──────────────┘   ┌───────────┴──────────┐
                            │                  │ AI Models in Memory  │
             ┌──────────────┴──────────────┐   │  • Symptom DNN       │
             │      Database Layer         │   │  • Diabetes DNN      │
             │  SQLite: data/medismart.db  │   │  • Heart DNN         │
             │  (MySQL swappable via env)  │   │  • Expiry DNN        │
             └─────────────────────────────┘   │  • 8x Demand LSTMs   │
                                               └──────────────────────┘
```

---

## 2. Database Schema (SQLAlchemy ORM)

Located in `medismart/db/models.py`. Designed for SQLite out of the box (zero cost, zero configuration) and 100% compatible with MySQL by setting `DATABASE_URL=mysql+pymysql://user:pass@host:3306/medismart`.

### Key Tables & Models:
1. **`User`**:
   - `id`, `username`, `email`, `password_hash`, `role` (`admin`, `pharmacist`, `cashier`), `full_name`, `is_active`, `created_at`.
   - Security: Werkzeug secure password hashing (`generate_password_hash` / `check_password_hash`).
   - Integration: Flask-Login `UserMixin`.
2. **`Medicine`**:
   - `id`, `name`, `atc_group`, `category`, `unit_cost`, `unit_price`, `pack_size`, `prescription_required`, `shelf_life_months`, `min_stock_alert`, `description`.
   - Dynamic property: `total_stock` calculates real-time available stock across non-expired active batches.
3. **`Supplier`**:
   - `id`, `name`, `contact_person`, `email`, `phone`, `address`.
4. **`Batch`**:
   - `id`, `batch_number`, `medicine_id`, `supplier_id`, `mfg_date`, `expiry_date`, `quantity_received`, `quantity_remaining`, `unit_cost`, `status` (`ACTIVE`, `DEPLETED`, `EXPIRED`, `QUARANTINED`).
   - AI risk columns: `risk_score` (0-100), `risk_bucket` (`SAFE`, `WATCH`, `HIGH`, `CRITICAL`), `value_at_risk`, `last_risk_assessed_at`.
5. **`Sale` & `SaleItem`**:
   - `Sale`: `id`, `invoice_no`, `cashier_id`, `customer_name`, `customer_phone`, `total_amount`, `discount`, `net_amount`, `payment_method`, `created_at`.
   - `SaleItem`: `id`, `sale_id`, `batch_id`, `medicine_id`, `quantity`, `unit_price`, `subtotal`.
6. **`StockAdjustment`**:
   - Audit trail for damaged stock, physical audit corrections, or expired batch disposals.
7. **`PredictionLog`**:
   - Persists all patient disease consultations: `patient_name`, `patient_age`, `patient_gender`, `glucose`, `bmi`, `blood_pressure`, `symptoms_json`, `predicted_disease`, `confidence`, `urgency`, `recommended_meds_json`, `specialist`, `department`, `is_emergency`.

---

## 3. Core Business Logic & AI Services

### 3.1 First-Expiry-First-Out (FEFO) Engine (`medismart/services/inventory.py`)
- Standard pharmacy inventory practice dictates that batches expiring soonest must be dispensed first to minimize wastage.
- `preview_fefo_allocation(medicine_id, requested_qty)` queries active batches sorted by `expiry_date ASC, id ASC`.
- Allocates quantities across batches and computes subtotals.
- `execute_sale(...)` applies the FEFO allocation atomically inside a database transaction:
  - Decrements batch `quantity_remaining`.
  - Marks batches as `DEPLETED` when remaining units reach 0.
  - Creates the `Sale` invoice and individual `SaleItem` records tied to specific batches for full auditability.

### 3.2 AI Serving & Multimodal Clinical Fusion (`medismart/services/ai_engine.py`)
- **Single Startup Load**: All Phase 3 trained models and Phase 4 Treatment DB are loaded once into memory on application initialization, guaranteeing <100ms inference time.
- **Multimodal Fusion**:
  - Handles symptoms (131 binary flags) + vitals (Age, Glucose, BMI, Blood Pressure).
  - Evaluates Symptom DNN (41 disease softmax output).
  - Evaluates Diabetes DNN & Heart Disease DNN.
  - **Calibrated Fusion**: When vitals (e.g. Glucose >= 140 mg/dL, BMI >= 28) align with diabetes symptoms (polyuria, excessive hunger, fatigue), the system reinforces the confidence to >=90%.
  - **Red-Flag Escalation**: Checks symptoms against 7 critical red flags (`chest_pain`, `slurred_speech`, `coma`, etc.) and automatically sets `is_emergency=True` with `urgency="critical"`.
  - Queries `TreatmentService` for first-line medications, specialist referrals (e.g. Endocrinologist), precautions, and safety disclaimers.
- **Real-Time Expiry Risk Scorer**:
  - Consumes batch parameters: `days_to_expiry`, `quantity_remaining`, `unit_cost`, and derived `cover_ratio`.
  - Outputs risk score (0-100), risk bucket (`SAFE`, `WATCH`, `HIGH`, `CRITICAL`), and projected `value_at_risk`.

---

## 4. REST API Endpoints

### Authentication (`/api/auth`)
- `POST /api/auth/login`: Authenticate with username/email and password.
- `POST /api/auth/logout`: End session.
- `GET /api/auth/me`: Current session user info.
- `GET /api/auth/users`: List users (admin only).

### Inventory Management (`/api/inventory`)
- `GET /api/inventory/medicines`: List catalog medicines with live total stock.
- `GET /api/inventory/medicines/<id>`: Single medicine details with batch breakdown.
- `POST /api/inventory/medicines`: Add new medicine.
- `GET /api/inventory/batches`: Query batches filtered by `status`, `risk_bucket`, or `medicine_id`.
- `POST /api/inventory/batches/receive`: Stock intake from supplier.
- `POST /api/inventory/fefo-preview`: Compute FEFO batch allocation plan.
- `POST /api/inventory/sale`: Process sale transaction with automatic FEFO deductions.
- `GET /api/inventory/sales`: List sales transaction history.
- `GET /api/inventory/low-stock`: Shortage and reorder alerts.
- `GET /api/inventory/suppliers`: List registered suppliers.

### Disease Prediction (`/api/predict`)
- `POST /api/predict/diagnose`: Multimodal disease diagnosis from vitals + symptoms.
- `GET /api/predict/symptoms`: All 131 symptom strings with severity weights (1-7) and emergency flags.
- `GET /api/predict/history`: Recent clinical consultation logs.
- `GET /api/predict/diseases`: Browse 41 diseases with treatment guidelines.

### Expiry Risk & Demand Forecasting (`/api/expiry`)
- `GET /api/expiry/analytics`: High-level wastage metrics, risk breakdown, and critical batches.
- `POST /api/expiry/assess-batch/<id>`: Run AI expiry model on a specific batch and update database.
- `POST /api/expiry/assess-all`: Batch assess every active batch in the pharmacy.
- `GET /api/expiry/forecast/<atc_group>`: 14-day LSTM consumption forecast.

### Dashboard (`/api/dashboard`)
- `GET /api/dashboard/summary`: High-level aggregated KPIs (inventory value, daily sales, near-expiry count, diagnostics count).

---

## 5. Verification Results

All 8 automated tests in `scripts/test_backend.py` passed:
1. **Health Check**: `/api/health` -> HTTP 200 `healthy`.
2. **Authentication**: Admin credentials verified, invalid logins rejected with HTTP 401.
3. **Medicine Catalog**: 22 catalog medicines retrieved.
4. **FEFO Dispensing**: 25 units sold -> verified that the earliest-expiring batch was decremented by exactly 25 units.
5. **Project Worked Example**:
   - Input: Age 45, Glucose 180 mg/dL, BMI 29, BP 140/90, symptoms (polyuria, excessive hunger, fatigue, weight loss).
   - Output: Predicted Condition: **Diabetes**, Confidence: **100.0%**, First-line: **Metformin 500mg**, Specialist: **Endocrinologist / Diabetologist**.
6. **Emergency Red-Flags**: Chest pain + slurred speech -> Emergency Flag: **True**, Urgency: **Critical**.
7. **Real-time Expiry Risk**: Evaluated batch -> Risk Score: 66.8/100, Bucket: **HIGH**, actionable recommendation generated.
8. **Dashboard KPIs**: Aggregated across inventory, sales, expiry, and consultations.
