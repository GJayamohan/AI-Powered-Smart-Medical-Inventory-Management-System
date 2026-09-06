"""
Disease prediction and clinical decision support routes.
"""

from __future__ import annotations

import json
from flask import Blueprint, jsonify, request

from medismart.db import db
from medismart.db.models import PredictionLog
from medismart.services.ai_engine import AIEngine

predict_bp = Blueprint("predict_bp", __name__, url_prefix="/api/predict")


@predict_bp.route("/symptoms", methods=["GET"])
def get_available_symptoms():
    """Returns the vocabulary of 131 symptoms formatted with display names and severity weights."""
    ai = AIEngine.get_instance()
    features = ai.symptom_features

    symptoms_list = []
    for s in features:
        severity = ai.treatment_svc.get_symptom_severity(s)
        is_emerg = ai.treatment_svc.is_emergency(s)
        symptoms_list.append({
            "id": s,
            "name": s.replace("_", " ").title(),
            "severity": severity,
            "is_emergency": is_emerg,
        })

    # Sort alphabetically by display name
    symptoms_list.sort(key=lambda x: x["name"])
    return jsonify({
        "success": True,
        "count": len(symptoms_list),
        "symptoms": symptoms_list,
    })


@predict_bp.route("/diagnose", methods=["POST"])
def diagnose_condition():
    """
    Multimodal clinical decision support endpoint:
    Accepts:
      {
        "patient": {"name": "...", "age": 45, "gender": "male"},
        "vitals": {"glucose": 180, "bmi": 29, "blood_pressure": "140/90"},
        "symptoms": ["increased thirst", "frequent urination", "fatigue"]
      }
    Returns predicted condition, confidence score, treatment recommendations, and specialist.
    """
    data = request.get_json() or {}
    symptoms = data.get("symptoms", [])
    vitals = data.get("vitals", {})
    patient = data.get("patient", {})

    if not symptoms and not vitals:
        return jsonify({"success": False, "error": "Please provide either symptoms or clinical vitals."}), 400

    ai = AIEngine.get_instance()
    result = ai.predict_disease(symptoms=symptoms, vitals=vitals, patient_info=patient)

    # Save consultation to PredictionLog
    try:
        log = PredictionLog(
            patient_name=patient.get("name", "Walk-in Patient"),
            patient_age=patient.get("age") or vitals.get("age"),
            patient_gender=patient.get("gender"),
            glucose=vitals.get("glucose"),
            bmi=vitals.get("bmi"),
            blood_pressure=str(vitals.get("blood_pressure")) if vitals.get("blood_pressure") else None,
            symptoms_json=json.dumps(symptoms),
            predicted_disease=result["predicted_condition"],
            confidence=result["model_confidence_decimal"],
            urgency=result["urgency"],
            first_line_treatment=result["first_line_treatment"],
            recommended_meds_json=json.dumps(result["recommended_medication_classes"]),
            specialist=result["specialist_to_consult"],
            department=result["department"],
            red_flags_json=json.dumps(result["warning_signs"]),
            is_emergency=result["is_emergency"],
        )
        db.session.add(log)
        db.session.commit()
        result["log_id"] = log.id
    except Exception as e:
        db.session.rollback()
        # Non-fatal: still return prediction even if logging fails
        result["log_error"] = str(e)

    return jsonify({"success": True, "result": result})


@predict_bp.route("/history", methods=["GET"])
def get_prediction_history():
    limit = request.args.get("limit", default=25, type=int)
    logs = PredictionLog.query.order_by(PredictionLog.created_at.desc()).limit(limit).all()
    return jsonify({"success": True, "count": len(logs), "history": [l.to_dict() for l in logs]})


@predict_bp.route("/diseases", methods=["GET"])
def list_diseases():
    """List all 41 diseases in the curated Treatment Database."""
    ai = AIEngine.get_instance()
    names = ai.treatment_svc.disease_names

    diseases_data = []
    for d in names:
        rec = ai.treatment_svc.get_recommendation(d)
        if rec:
            diseases_data.append({
                "name": rec["name"].strip(),
                "urgency": rec["urgency"],
                "specialist": rec["specialist"],
                "department": rec["department"],
                "first_line": rec["first_line"],
                "medication_classes": rec["medication_classes"],
            })

    diseases_data.sort(key=lambda x: x["name"])
    return jsonify({"success": True, "count": len(diseases_data), "diseases": diseases_data})
