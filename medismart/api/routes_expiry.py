"""
Expiry risk analytics and demand forecasting routes.
"""

from __future__ import annotations

from datetime import datetime
from flask import Blueprint, jsonify, request

from medismart.db import db
from medismart.db.models import Batch
from medismart.services.ai_engine import AIEngine

expiry_bp = Blueprint("expiry_bp", __name__, url_prefix="/api/expiry")


@expiry_bp.route("/analytics", methods=["GET"])
def get_expiry_analytics():
    """Returns high-level expiry risk breakdown across all inventory batches."""
    batches = Batch.query.filter(Batch.status == "ACTIVE", Batch.quantity_remaining > 0).all()

    buckets = {"SAFE": 0, "WATCH": 0, "HIGH": 0, "CRITICAL": 0}
    total_var = 0.0
    critical_batches = []
    near_expired_count = 0

    for b in batches:
        b_dict = b.to_dict()
        bucket = b_dict.get("risk_bucket", "SAFE")
        buckets[bucket] = buckets.get(bucket, 0) + 1
        total_var += b_dict.get("value_at_risk", 0.0)

        if b_dict.get("days_to_expiry", 999) <= 30:
            near_expired_count += 1

        if bucket in ("CRITICAL", "HIGH"):
            critical_batches.append(b_dict)

    # Sort critical batches by risk score desc
    critical_batches.sort(key=lambda x: -x.get("risk_score", 0))

    return jsonify({
        "success": True,
        "total_active_batches": len(batches),
        "risk_breakdown": buckets,
        "near_expired_count": near_expired_count,
        "total_value_at_risk": round(total_var, 2),
        "critical_batches_count": len(critical_batches),
        "critical_batches": critical_batches[:15],
    })


@expiry_bp.route("/assess-batch/<int:batch_id>", methods=["POST"])
def assess_batch(batch_id: int):
    """Triggers real-time AI risk evaluation on a specific batch and updates its risk score."""
    batch = db.session.get(Batch, batch_id)
    if not batch:
        return jsonify({"success": False, "error": f"Batch ID {batch_id} not found."}), 404

    ai = AIEngine.get_instance()
    assessment = ai.evaluate_batch_expiry_risk(
        days_to_expiry=batch.days_to_expiry,
        quantity_remaining=batch.quantity_remaining,
        quantity_received=batch.quantity_received,
        unit_cost=batch.unit_cost,
        shelf_life_days=batch.medicine.shelf_life_months * 30 if batch.medicine else 540,
        prescription_required=batch.medicine.prescription_required if batch.medicine else False,
    )

    # Persist updated AI risk metrics
    batch.risk_score = assessment["risk_score"]
    batch.risk_bucket = assessment["risk_bucket"]
    batch.value_at_risk = assessment["value_at_risk"]
    batch.last_risk_assessed_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        "success": True,
        "batch_id": batch.id,
        "batch_number": batch.batch_number,
        "assessment": assessment,
    })


@expiry_bp.route("/assess-all", methods=["POST"])
def assess_all_batches():
    """Runs AI risk evaluation across every active batch in the pharmacy."""
    batches = Batch.query.filter(Batch.status == "ACTIVE", Batch.quantity_remaining > 0).all()
    ai = AIEngine.get_instance()
    assessed = 0

    for batch in batches:
        assessment = ai.evaluate_batch_expiry_risk(
            days_to_expiry=batch.days_to_expiry,
            quantity_remaining=batch.quantity_remaining,
            quantity_received=batch.quantity_received,
            unit_cost=batch.unit_cost,
            shelf_life_days=batch.medicine.shelf_life_months * 30 if batch.medicine else 540,
            prescription_required=batch.medicine.prescription_required if batch.medicine else False,
        )
        batch.risk_score = assessment["risk_score"]
        batch.risk_bucket = assessment["risk_bucket"]
        batch.value_at_risk = assessment["value_at_risk"]
        batch.last_risk_assessed_at = datetime.utcnow()
        assessed += 1

    db.session.commit()
    return jsonify({"success": True, "message": f"Successfully assessed {assessed} active batches with AI model."})


@expiry_bp.route("/forecast/<atc_group>", methods=["GET"])
def get_demand_forecast(atc_group: str):
    """Runs the trained LSTM for an ATC drug group to forecast next 14 days of consumption."""
    days = request.args.get("days", default=14, type=int)
    ai = AIEngine.get_instance()
    res = ai.forecast_demand(atc_group.upper(), forecast_days=days)
    status_code = 200 if res["success"] else 400
    return jsonify(res), status_code
