"""
Dashboard overview and pharmacy analytics metrics.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from flask import Blueprint, jsonify

from medismart.db import db
from medismart.db.models import Batch, Medicine, PredictionLog, Sale
from medismart.services.inventory import InventoryService

dashboard_bp = Blueprint("dashboard_bp", __name__, url_prefix="/api/dashboard")


@dashboard_bp.route("/summary", methods=["GET"])
def get_summary():
    """Aggregates all key KPIs across inventory, sales, expiry risk, and diagnostics."""
    today = date.today()

    # 1. Inventory stats
    active_batches = Batch.query.filter(Batch.status == "ACTIVE", Batch.quantity_remaining > 0).all()
    total_stock_units = sum(b.quantity_remaining for b in active_batches)
    total_inventory_value = sum(b.quantity_remaining * b.unit_cost for b in active_batches)
    total_medicines_count = Medicine.query.count()

    # 2. Low stock & Near expiry alerts
    low_stock_list = InventoryService.get_low_stock_alerts()
    near_expiry_batches = [b for b in active_batches if (b.expiry_date - today).days <= 30]
    critical_risk_batches = [b for b in active_batches if b.risk_bucket in ("CRITICAL", "HIGH")]

    # 3. Sales stats (Today & Last 7 Days)
    today_start = datetime.combine(today, datetime.min.time())
    today_sales = Sale.query.filter(Sale.created_at >= today_start).all()
    today_revenue = sum(s.net_amount for s in today_sales)
    today_orders = len(today_sales)

    week_start = today_start - timedelta(days=7)
    week_sales = Sale.query.filter(Sale.created_at >= week_start).all()
    week_revenue = sum(s.net_amount for s in week_sales)

    # 4. Diagnostic consultations
    total_predictions = PredictionLog.query.count()
    today_predictions = PredictionLog.query.filter(PredictionLog.created_at >= today_start).count()
    recent_predictions = (
        PredictionLog.query.order_by(PredictionLog.created_at.desc()).limit(5).all()
    )

    return jsonify({
        "success": True,
        "inventory": {
            "total_medicines": total_medicines_count,
            "total_stock_units": total_stock_units,
            "total_inventory_value": round(total_inventory_value, 2),
            "active_batches_count": len(active_batches),
            "low_stock_count": len(low_stock_list),
        },
        "expiry_risk": {
            "near_expiry_count": len(near_expiry_batches),
            "critical_risk_count": len(critical_risk_batches),
            "total_value_at_risk": round(sum(b.value_at_risk for b in active_batches), 2),
        },
        "sales": {
            "today_revenue": round(today_revenue, 2),
            "today_orders": today_orders,
            "week_revenue": round(week_revenue, 2),
        },
        "diagnostics": {
            "total_consultations": total_predictions,
            "today_consultations": today_predictions,
            "recent": [p.to_dict() for p in recent_predictions],
        },
    })
