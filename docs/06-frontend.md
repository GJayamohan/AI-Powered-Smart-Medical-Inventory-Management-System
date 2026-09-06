# Phase 7: Frontend Web Application & Conversational Assistant

This document details the frontend user interface, Jinja2 template architecture, styling system, interactive client-side logic, and AI Conversational Assistant for the **MediSmart** pharmacy management platform.

---

## 1. UI Design Principles & Technology Stack

The MediSmart frontend was built for clinical accuracy, modern fintech/healthtech aesthetics, and high usability:
- **Architecture**: Server-rendered HTML5 via Flask Jinja2 templates paired with lightweight, dependency-free vanilla JavaScript for asynchronous API interactions.
- **Styling**: Modern, responsive CSS design system in `medismart/web/static/css/app.css` using CSS custom properties (variables), high-contrast medical palettes (Teal `#0f766e`, Slate `#1e293b`, Emerald `#10b981`, Amber `#f59e0b`, Red `#ef4444`).
- **Data Visualizations**: Integrated Chart.js for real-time risk distribution donut charts, category-wise inventory bar charts, and 14-day LSTM demand forecasting line charts.
- **Zero Cost**: Zero subscription or runtime licensing fees — 100% open-source libraries.

---

## 2. Page Templates & Features

### 2.1 Master Layout (`medismart/web/templates/base.html`)
- **Navigation Sidebar**: Direct links to Core Operations (Dashboard, Inventory, POS, Sales) and AI Subsystems (Expiry Risk AI, Disease Diagnosis AI).
- **Topbar Live Counters**: Real-time polling counters for Near-Expiry batches (<=30 days) and Low Stock shortages. Clicking either navigates directly to the filtered action table.
- **Role Profile**: Displays current user session with role indicator (Admin / Pharmacist / Cashier).
- **Floating AI Chatbot**: Expandable floating assistant available on every page.

### 2.2 Pharmacy Dashboard (`medismart/web/templates/dashboard.html` / route `/`)
- **KPI Metrics Cards**: Total Stock Valuation, Near-Expiry Warning count, Low Stock count, Today's Sales Revenue, and Completed AI Consultations.
- **Interactive Visualizations**:
  - *AI Expiry Risk Distribution Donut*: Live breakdown of active batches across `SAFE`, `WATCH`, `HIGH`, and `CRITICAL` risk buckets.
  - *Stock by Therapeutic Class*: Bar chart showing units on hand across 8 ATC drug categories.
- **Action Tables**:
  - Near-expiry high-risk batch alerts with days remaining and risk scores.
  - Low-stock reorder alerts with shortages.

### 2.3 Inventory Management (`medismart/web/templates/inventory.html` / route `/inventory`)
- Real-time search filter by medicine name or ATC classification.
- Master catalog table showing unit cost, selling price, pack size, prescription requirement, and total active stock across batches.
- **Batch Explorer Modal**: Click "Batches" on any medicine to inspect all active batches, manufacturing dates, expiry dates, days to expiry, and AI risk ratings.
- **Modals**:
  - Add New Medicine Modal with ATC category selector.
  - Receive New Batch Stock Modal with supplier linkage.

### 2.4 AI Expiry Risk Prediction (`medismart/web/templates/expiry.html` / route `/expiry`)
- Directly implements the user requirement for AI identifying near-expired medicines.
- **Batch Risk Grid**:
  - Expiry date, days to expiry (color-coded red <=30d, yellow <=90d, green >90d).
  - Stock remaining and quantity received.
  - **Cover Ratio**: $(\text{Daily Usage} \times \text{Days}) / \text{Stock}$.
  - **AI Risk Score (0-100)**: Visual progress bar and score.
  - **Risk Bucket**: `CRITICAL`, `HIGH`, `WATCH`, `SAFE`.
  - **Value at Risk (Rs)**.
  - **AI Action Recommendation**: e.g., "High write-off risk! Apply 30% discount or return to supplier."
  - "🤖 Re-Score Batch" button invoking the Model 2b inference endpoint.
- **Demand Forecasting Tool**: Choose any of the 8 ATC drug categories and run the trained LSTM to generate a 14-day projected consumption curve.

### 2.5 Disease Prediction & Clinical Decision Support (`medismart/web/templates/predict.html` / route `/predict`)
- **Patient Intake Form**: Patient name, age, gender.
- **Quantitative Vitals**: Blood Glucose (mg/dL), Blood Pressure (mmHg), BMI (kg/m²).
- **Reported Symptoms**: Searchable multi-select tags across all 131 clinical symptoms with emergency markers (🚨).
- ⚡ **"Load Worked Example" Button**:
  - Pre-fills the exact worked example from the project brief: Age: 45, Glucose: 180, BMI: 29, BP: 140/90, and symptoms (frequent urination, excessive hunger, fatigue, weight loss).
- **Clinical Recommendation Output Card**:
  - Predicted Condition (e.g. Diabetes).
  - Model Confidence Score (100% via Multimodal Fusion).
  - Urgency Level (`CRITICAL`, `HIGH`, `MODERATE`, `LOW`).
  - Emergency Alert Banner if red-flag symptoms are detected.
  - Recommended Medication Class (e.g., Metformin, Sulfonylureas) & First-line dosage.
  - Recommended Specialist to Consult (e.g., Endocrinologist / Diabetologist) & Medical Department.
  - Precautions & Lifestyle Advice.
  - Warning Signs to watch for.
  - Mandatory Medical Disclaimer.

### 2.6 Point of Sale & FEFO Dispensing (`medismart/web/templates/pos.html` / route `/pos`)
- Select medicine and input desired quantity.
- **Live FEFO Allocation Preview**: Dynamically queries the backend and displays the exact batches being decremented (earliest expiring batches first).
- Cart items list, discount calculation, customer details, and payment method (Cash, UPI, Card).
- Atomic checkout and printable invoice modal.

### 2.7 Sales Ledger (`medismart/web/templates/sales.html` / route `/sales`)
- Complete transaction log with invoice numbers, timestamps, items sold, payment methods, and net revenues.

### 2.8 Authentication Portal (`medismart/web/templates/login.html` / route `/login`)
- Clean portal with 1-click demo role buttons (`Admin`, `Pharmacist`, `Cashier`) for instant testing.

---

## 3. Conversational AI Assistant (`medismart/chatbot/assistant.py`)

The floating assistant operates with **zero running cost** and zero latency using an offline clinical knowledge-graph and real-time inventory query engine:
1. **Stock Inquiries**: *"Do we have Paracetamol?"*, *"Check stock of Diclofenac"*.
2. **Shortage Alerts**: *"Show low stock medicines"*.
3. **Expiry Alerts**: *"Which batches are expiring this month?"*, *"Show near expiry batches"*.
4. **Clinical Guidance**: *"What doctor to consult for Diabetes?"*, *"Medication for Asthma"*.
5. **System Rules**: *"Explain FEFO dispensing"*, *"Show worked example"*.

---

## 4. Verification Results (`scripts/test_frontend.py`)

All tests passed with 100% success:
- **Suite 1: Web Routes** — All 7 HTML endpoints returned HTTP 200 OK.
- **Suite 2: Static Assets** — `app.css` and `app.js` served with HTTP 200 OK.
- **Suite 3: Chatbot Assistant** — Handled stock queries, expiry summaries, doctor referrals, FEFO rules, and worked examples with verified intent matching.
