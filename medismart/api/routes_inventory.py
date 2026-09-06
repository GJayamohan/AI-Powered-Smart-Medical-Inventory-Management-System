"""
Inventory and FEFO dispensing routes.
"""

from __future__ import annotations

from datetime import datetime
from flask import Blueprint, jsonify, request
from flask_login import current_user

from medismart.db import db
from medismart.db.models import Batch, Medicine, Sale, Supplier
from medismart.services.inventory import InventoryService

inventory_bp = Blueprint("inventory_bp", __name__, url_prefix="/api/inventory")


@inventory_bp.route("/medicines", methods=["GET"])
def list_medicines():
    search = request.args.get("q", "").strip()
    medicines = InventoryService.get_all_medicines(search=search if search else None)
    return jsonify({"success": True, "count": len(medicines), "medicines": medicines})


@inventory_bp.route("/medicines/<int:med_id>", methods=["GET"])
def get_medicine(med_id: int):
    med = InventoryService.get_medicine_by_id(med_id, include_batches=True)
    if not med:
        return jsonify({"success": False, "error": "Medicine not found"}), 404
    return jsonify({"success": True, "medicine": med})


@inventory_bp.route("/medicines", methods=["POST"])
def add_medicine():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    atc_group = data.get("atc_group", "").strip().upper()
    unit_cost = float(data.get("unit_cost", 0.0))
    unit_price = float(data.get("unit_price", 0.0))

    if not name or not atc_group:
        return jsonify({"success": False, "error": "Medicine name and ATC group are required."}), 400

    existing = Medicine.query.filter(Medicine.name.ilike(name)).first()
    if existing:
        return jsonify({"success": False, "error": f"Medicine '{name}' already exists."}), 400

    med = Medicine(
        name=name,
        atc_group=atc_group,
        category=data.get("category", "General Medicine"),
        unit_cost=unit_cost,
        unit_price=unit_price,
        pack_size=int(data.get("pack_size", 1)),
        prescription_required=bool(data.get("prescription_required", False)),
        shelf_life_months=int(data.get("shelf_life_months", 24)),
        min_stock_alert=int(data.get("min_stock_alert", 30)),
        description=data.get("description", ""),
    )
    db.session.add(med)
    db.session.commit()
    return jsonify({"success": True, "message": "Medicine added successfully.", "medicine": med.to_dict()}), 201


@inventory_bp.route("/batches", methods=["GET"])
def list_batches():
    status = request.args.get("status")
    risk_bucket = request.args.get("risk_bucket")
    medicine_id = request.args.get("medicine_id", type=int)

    batches = InventoryService.get_all_batches(
        status=status, risk_bucket=risk_bucket, medicine_id=medicine_id
    )
    return jsonify({"success": True, "count": len(batches), "batches": batches})


@inventory_bp.route("/batches/receive", methods=["POST"])
def receive_batch():
    data = request.get_json() or {}
    try:
        med_id = int(data["medicine_id"])
        batch_number = str(data["batch_number"]).strip()
        mfg_date = datetime.strptime(data["mfg_date"], "%Y-%m-%d").date()
        expiry_date = datetime.strptime(data["expiry_date"], "%Y-%m-%d").date()
        qty = int(data["quantity"])
        unit_cost = float(data["unit_cost"])
        supplier_id = data.get("supplier_id")
        if supplier_id:
            supplier_id = int(supplier_id)
    except (KeyError, ValueError) as e:
        return jsonify({"success": False, "error": f"Invalid or missing payload: {e}"}), 400

    result = InventoryService.receive_stock_batch(
        medicine_id=med_id,
        batch_number=batch_number,
        mfg_date=mfg_date,
        expiry_date=expiry_date,
        quantity=qty,
        unit_cost=unit_cost,
        supplier_id=supplier_id,
    )
    status_code = 201 if result["success"] else 400
    return jsonify(result), status_code


@inventory_bp.route("/fefo-preview", methods=["POST"])
def preview_fefo():
    data = request.get_json() or {}
    med_id = data.get("medicine_id")
    qty = data.get("quantity")

    if not med_id or not qty:
        return jsonify({"success": False, "error": "medicine_id and quantity are required."}), 400

    plan = InventoryService.preview_fefo_allocation(int(med_id), int(qty))
    status_code = 200 if plan["success"] else 400
    return jsonify(plan), status_code


@inventory_bp.route("/sale", methods=["POST"])
def process_sale():
    data = request.get_json() or {}
    items = data.get("items", [])
    if not items:
        return jsonify({"success": False, "error": "Cart items cannot be empty."}), 400

    cashier_id = current_user.id if current_user.is_authenticated else None
    result = InventoryService.execute_sale(
        items=items,
        cashier_id=cashier_id,
        customer_name=data.get("customer_name", "Walk-in Customer"),
        customer_phone=data.get("customer_phone"),
        payment_method=data.get("payment_method", "CASH"),
        discount=float(data.get("discount", 0.0)),
    )
    status_code = 200 if result["success"] else 400
    return jsonify(result), status_code


@inventory_bp.route("/sales", methods=["GET"])
def list_sales():
    limit = request.args.get("limit", default=20, type=int)
    sales = Sale.query.order_by(Sale.created_at.desc()).limit(limit).all()
    return jsonify({"success": True, "sales": [s.to_dict() for s in sales]})


@inventory_bp.route("/low-stock", methods=["GET"])
def get_low_stock():
    alerts = InventoryService.get_low_stock_alerts()
    return jsonify({"success": True, "count": len(alerts), "alerts": alerts})


@inventory_bp.route("/suppliers", methods=["GET"])
def list_suppliers():
    suppliers = Supplier.query.order_by(Supplier.name.asc()).all()
    return jsonify({"success": True, "suppliers": [s.to_dict() for s in suppliers]})
