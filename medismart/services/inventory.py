"""
Inventory Management Service.

Implements First-Expiry-First-Out (FEFO) dispensing algorithm,
batch allocation, stock replenishment, and audit logs.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from medismart.db import db
from medismart.db.models import Batch, Medicine, Sale, SaleItem, StockAdjustment, Supplier


class InventoryService:
    @staticmethod
    def get_all_medicines(search: Optional[str] = None) -> list[dict]:
        query = Medicine.query
        if search:
            query = query.filter(Medicine.name.ilike(f"%{search}%"))
        medicines = query.order_by(Medicine.name.asc()).all()
        return [m.to_dict() for m in medicines]

    @staticmethod
    def get_medicine_by_id(med_id: int, include_batches: bool = True) -> Optional[dict]:
        med = db.session.get(Medicine, med_id)
        return med.to_dict(include_batches=include_batches) if med else None

    @staticmethod
    def get_all_batches(
        status: Optional[str] = None,
        risk_bucket: Optional[str] = None,
        medicine_id: Optional[int] = None
    ) -> list[dict]:
        query = Batch.query
        if status:
            query = query.filter(Batch.status == status)
        if risk_bucket:
            query = query.filter(Batch.risk_bucket == risk_bucket)
        if medicine_id:
            query = query.filter(Batch.medicine_id == medicine_id)

        batches = query.order_by(Batch.expiry_date.asc()).all()
        return [b.to_dict() for b in batches]

    @staticmethod
    def preview_fefo_allocation(medicine_id: int, requested_qty: int) -> dict:
        """
        Calculates FEFO allocation plan for a medicine without modifying the database.
        Returns whether sufficient stock exists and the exact breakdown per batch.
        """
        today = date.today()
        med = db.session.get(Medicine, medicine_id)
        if not med:
            return {"success": False, "error": f"Medicine with id {medicine_id} not found."}

        # Query active batches ordered by expiry_date ASC
        available_batches = (
            Batch.query.filter(
                Batch.medicine_id == medicine_id,
                Batch.status == "ACTIVE",
                Batch.quantity_remaining > 0,
                Batch.expiry_date >= today
            )
            .order_by(Batch.expiry_date.asc(), Batch.id.asc())
            .all()
        )

        total_avail = sum(b.quantity_remaining for b in available_batches)
        if total_avail < requested_qty:
            return {
                "success": False,
                "error": f"Insufficient stock. Requested: {requested_qty}, Available: {total_avail}",
                "total_available": total_avail,
            }

        allocation = []
        remaining_to_allocate = requested_qty

        for b in available_batches:
            take = min(b.quantity_remaining, remaining_to_allocate)
            allocation.append({
                "batch_id": b.id,
                "batch_number": b.batch_number,
                "expiry_date": b.expiry_date.isoformat(),
                "days_to_expiry": b.days_to_expiry,
                "take_quantity": take,
                "unit_price": med.unit_price,
                "subtotal": round(take * med.unit_price, 2),
            })
            remaining_to_allocate -= take
            if remaining_to_allocate <= 0:
                break

        return {
            "success": True,
            "medicine_id": med.id,
            "medicine_name": med.name,
            "requested_qty": requested_qty,
            "total_subtotal": round(sum(item["subtotal"] for item in allocation), 2),
            "allocation": allocation,
        }

    @staticmethod
    def execute_sale(
        items: list[dict],
        cashier_id: Optional[int] = None,
        customer_name: str = "Walk-in Customer",
        customer_phone: Optional[str] = None,
        payment_method: str = "CASH",
        discount: float = 0.0,
    ) -> dict:
        """
        Executes a sale transaction across multiple cart items using FEFO batch deduction.
        Each item in `items` is: {"medicine_id": int, "quantity": int}
        """
        today = date.today()
        allocations_to_commit = []
        total_sale_amount = 0.0

        # Step 1: Validate and compute FEFO allocations for all items
        for item in items:
            med_id = item.get("medicine_id")
            qty = int(item.get("quantity", 0))
            if qty <= 0:
                return {"success": False, "error": f"Quantity must be greater than 0 for medicine ID {med_id}"}

            plan = InventoryService.preview_fefo_allocation(med_id, qty)
            if not plan["success"]:
                return plan

            allocations_to_commit.append(plan)
            total_sale_amount += plan["total_subtotal"]

        # Step 2: Create Sale Header
        now = datetime.utcnow()
        invoice_no = f"INV-{now.strftime('%Y%m%d%H%M%S')}-{int(now.timestamp() % 1000)}"
        net_amount = max(0.0, total_sale_amount - discount)

        sale = Sale(
            invoice_no=invoice_no,
            cashier_id=cashier_id,
            customer_name=customer_name,
            customer_phone=customer_phone,
            total_amount=round(total_sale_amount, 2),
            discount=round(discount, 2),
            net_amount=round(net_amount, 2),
            payment_method=payment_method,
            created_at=now,
        )
        db.session.add(sale)
        db.session.flush()

        # Step 3: Deduct batch quantities & create SaleItem records
        for plan in allocations_to_commit:
            for alloc in plan["allocation"]:
                batch = db.session.get(Batch, alloc["batch_id"])
                batch.quantity_remaining -= alloc["take_quantity"]
                if batch.quantity_remaining <= 0:
                    batch.quantity_remaining = 0
                    batch.status = "DEPLETED"

                sale_item = SaleItem(
                    sale_id=sale.id,
                    batch_id=batch.id,
                    medicine_id=plan["medicine_id"],
                    quantity=alloc["take_quantity"],
                    unit_price=alloc["unit_price"],
                    subtotal=alloc["subtotal"],
                )
                db.session.add(sale_item)

        db.session.commit()
        return {
            "success": True,
            "sale_id": sale.id,
            "invoice_no": sale.invoice_no,
            "total_amount": sale.total_amount,
            "discount": sale.discount,
            "net_amount": sale.net_amount,
            "created_at": sale.created_at.isoformat(),
            "items_count": len(sale.items),
        }

    @staticmethod
    def receive_stock_batch(
        medicine_id: int,
        batch_number: str,
        mfg_date: date,
        expiry_date: date,
        quantity: int,
        unit_cost: float,
        supplier_id: Optional[int] = None,
    ) -> dict:
        """Receive a new batch of medicines from a supplier."""
        med = db.session.get(Medicine, medicine_id)
        if not med:
            return {"success": False, "error": f"Medicine with id {medicine_id} not found."}

        existing = Batch.query.filter_by(batch_number=batch_number).first()
        if existing:
            return {"success": False, "error": f"Batch number '{batch_number}' already exists."}

        if expiry_date <= mfg_date:
            return {"success": False, "error": "Expiry date must be after manufacturing date."}

        batch = Batch(
            batch_number=batch_number,
            medicine_id=medicine_id,
            supplier_id=supplier_id,
            mfg_date=mfg_date,
            expiry_date=expiry_date,
            quantity_received=quantity,
            quantity_remaining=quantity,
            unit_cost=unit_cost,
            status="ACTIVE",
            risk_score=0.0,
            risk_bucket="SAFE",
            value_at_risk=0.0,
        )
        db.session.add(batch)
        db.session.commit()
        return {"success": True, "batch": batch.to_dict()}

    @staticmethod
    def adjust_stock(batch_id: int, change_qty: int, reason: str, user_id: Optional[int] = None, note: str = "") -> dict:
        """Adjust batch stock for damages, audit adjustments, or expired stock disposal."""
        batch = db.session.get(Batch, batch_id)
        if not batch:
            return {"success": False, "error": f"Batch with id {batch_id} not found."}

        new_qty = batch.quantity_remaining + change_qty
        if new_qty < 0:
            return {"success": False, "error": f"Cannot reduce stock below zero. Current: {batch.quantity_remaining}"}

        batch.quantity_remaining = new_qty
        if new_qty == 0:
            batch.status = "DEPLETED" if reason != "EXPIRED_DISPOSAL" else "EXPIRED"

        adj = StockAdjustment(
            batch_id=batch.id,
            user_id=user_id,
            change_qty=change_qty,
            reason=reason,
            note=note,
            created_at=datetime.utcnow(),
        )
        db.session.add(adj)
        db.session.commit()
        return {"success": True, "adjustment": adj.to_dict(), "new_quantity": batch.quantity_remaining}

    @staticmethod
    def get_low_stock_alerts() -> list[dict]:
        """Returns medicines where total active stock is below min_stock_alert threshold."""
        meds = Medicine.query.all()
        alerts = []
        for m in meds:
            if m.total_stock <= m.min_stock_alert:
                alerts.append({
                    "medicine_id": m.id,
                    "name": m.name,
                    "atc_group": m.atc_group,
                    "current_stock": m.total_stock,
                    "min_stock_alert": m.min_stock_alert,
                    "shortage": m.min_stock_alert - m.total_stock,
                })
        return sorted(alerts, key=lambda x: x["current_stock"])
