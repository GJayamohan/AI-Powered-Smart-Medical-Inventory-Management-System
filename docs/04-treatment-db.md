# Phase 4: Treatment Knowledge Base

This document covers the curated Treatment Database that maps all 41 diagnosed
diseases to medication recommendations, specialist referrals, and safety
information.

## Overview

The Treatment DB is a **deterministic knowledge base** (not ML) that provides
reliable, auditable medical recommendations. It combines:

1. **Hand-curated clinical mappings** - medication classes, first-line treatments,
   specialist referrals, urgency levels, and red-flag symptoms for all 41 diseases
2. **Auxiliary CSV datasets** - disease descriptions, precautions, and symptom
   severity weights from the Columbia DBMI dataset
3. **Safety guardrails** - mandatory disclaimers, emergency escalation rules,
   and controlled-substance avoidance

## Safety Guardrails

- Recommends **medication classes**, never patient-specific dosages
- Never suggests controlled substances or narcotics
- Includes **154 red-flag rules** across all diseases for emergency escalation
- **7 emergency symptoms** that trigger immediate escalation (chest pain, coma,
  altered sensorium, slurred speech, etc.)
- Every response includes a medical disclaimer
- Critical conditions (Heart attack, Brain hemorrhage) trigger immediate
  emergency instructions

## Data Sources

| Source | Content | Records |
|---|---|---|
| Hand-curated (WHO, NHS, Mayo Clinic) | Medications, specialists, urgency, red flags | 41 diseases |
| `symptom_description.csv` | Plain-language disease descriptions | 41 |
| `symptom_precaution.csv` | 4 precautions per disease | 41 |
| `symptom_severity.csv` | Urgency weights per symptom (1-7 scale) | 132 |

## Database Structure

The treatment database (`data/knowledge/treatment_db.json`) contains:

```json
{
  "version": "1.0",
  "disclaimer": "...",
  "emergency_symptoms": ["chest_pain", "coma", ...],
  "diseases": {
    "Diabetes ": {
      "name": "Diabetes ",
      "description": "Diabetes is a disease that occurs when...",
      "medication_classes": ["Metformin", "Sulfonylureas", ...],
      "first_line": "Metformin 500mg twice daily",
      "specialist": "Endocrinologist / Diabetologist",
      "department": "Endocrinology",
      "urgency": "moderate",
      "lifestyle": ["Regular blood glucose monitoring", ...],
      "precautions": ["have balanced diet", "exercise", ...],
      "red_flags": ["Blood glucose > 400 mg/dL", ...]
    }
  },
  "symptom_severity": {"itching": 1, "chest_pain": 7, ...}
}
```

## Urgency Distribution

| Level | Count | Examples |
|---|---|---|
| **Critical** | 2 | Heart attack, Paralysis (brain hemorrhage) |
| **High** | 12 | AIDS, Dengue, Pneumonia, Tuberculosis, Malaria... |
| **Moderate** | 18 | Diabetes, Hypertension, Arthritis, Asthma... |
| **Low** | 9 | Acne, Allergy, Common Cold, GERD, Fungal infection... |

## Department Coverage

| Department | Diseases |
|---|---|
| Gastroenterology | 10 (all hepatitis variants, GERD, peptic ulcer, etc.) |
| Infectious Diseases | 4 (AIDS, Dengue, Malaria, Tuberculosis) |
| Dermatology | 4 (Acne, Fungal infection, Impetigo, Psoriasis) |
| Endocrinology | 4 (Diabetes, Hyper/Hypothyroidism, Hypoglycemia) |
| Pulmonology | 3 (Asthma, Pneumonia, Tuberculosis) |

## Service API

The `TreatmentService` class (`medismart/services/treatment.py`) provides:

```python
from medismart.services.treatment import TreatmentService

svc = TreatmentService()

# Full recommendation
rec = svc.get_recommendation("Diabetes ")

# Quick lookups
meds = svc.get_medication("Diabetes ")       # ["Metformin", ...]
doc  = svc.get_specialist("Diabetes ")       # "Endocrinologist / Diabetologist"
urg  = svc.get_urgency("Heart attack")       # "critical"

# Safety checks
svc.is_emergency("chest_pain")               # True
svc.check_emergency_symptoms(["chest_pain", "headache"])  # ["chest_pain"]
svc.get_symptom_severity("coma")             # 7

# Formatted output for display
print(svc.format_recommendation("Diabetes ", confidence=0.91))
```

## Files Created

| File | Purpose |
|---|---|
| `scripts/build_treatment_db.py` | Builder script with all 41 curated disease entries |
| `data/knowledge/treatment_db.json` | Generated treatment knowledge base |
| `medismart/services/treatment.py` | Service module for backend integration |
| `scripts/verify_treatment_db.py` | Verification and testing script |
| `docs/04-treatment-db.md` | This documentation |

## Next Phase: Backend Development (Phase 5)

Phase 5 will build the Flask web application with:
- Blueprints for inventory, prediction, expiry, chat, and API routes
- Flask-Login authentication (Admin/Pharmacist/Staff roles)
- Model serving pipeline (load all models once at startup)
- Treatment DB integration into prediction endpoints
