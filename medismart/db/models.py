"""
SQLAlchemy ORM models for MediSmart Pharmacy Management.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from medismart.db import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="pharmacist")  # admin, pharmacist, cashier
    full_name = db.Column(db.String(120), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    sales = db.relationship("Sale", back_populates="cashier", lazy="dynamic")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Supplier(db.Model):
    __tablename__ = "suppliers"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    contact_person = db.Column(db.String(120), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(30), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    batches = db.relationship("Batch", back_populates="supplier", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "contact_person": self.contact_person,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
        }


class Medicine(db.Model):
    __tablename__ = "medicines"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False, index=True)
    atc_group = db.Column(db.String(20), nullable=False, index=True)
    category = db.Column(db.String(100), nullable=True)
    unit_cost = db.Column(db.Float, nullable=False, default=0.0)
    unit_price = db.Column(db.Float, nullable=False, default=0.0)
    pack_size = db.Column(db.Integer, nullable=False, default=1)
    prescription_required = db.Column(db.Boolean, default=False)
    min_stock_alert = db.Column(db.Integer, default=50)
    shelf_life_months = db.Column(db.Integer, default=24)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    batches = db.relationship("Batch", back_populates="medicine", lazy="dynamic", cascade="all, delete-orphan")

    @property
    def total_stock(self) -> int:
        """Sum of quantity_remaining across all non-expired, active batches."""
        today = date.today()
        active_batches = self.batches.filter(
            Batch.status == "ACTIVE",
            Batch.quantity_remaining > 0,
            Batch.expiry_date >= today
        ).all()
        return sum(b.quantity_remaining for b in active_batches)

    def to_dict(self, include_batches: bool = False) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "atc_group": self.atc_group,
            "category": self.category,
            "unit_cost": self.unit_cost,
            "unit_price": self.unit_price,
            "pack_size": self.pack_size,
            "prescription_required": self.prescription_required,
            "min_stock_alert": self.min_stock_alert,
            "shelf_life_months": self.shelf_life_months,
            "description": self.description,
            "total_stock": self.total_stock,
            "is_low_stock": self.total_stock <= self.min_stock_alert,
        }
        if include_batches:
            data["batches"] = [b.to_dict() for b in self.batches.all()]
        return data


class Batch(db.Model):
    __tablename__ = "batches"

    id = db.Column(db.Integer, primary_key=True)
    batch_number = db.Column(db.String(64), unique=True, nullable=False, index=True)
    medicine_id = db.Column(db.Integer, db.ForeignKey("medicines.id"), nullable=False)
    supplier_id = db.Column(db.Integer, db.ForeignKey("suppliers.id"), nullable=True)
    mfg_date = db.Column(db.Date, nullable=False)
    expiry_date = db.Column(db.Date, nullable=False, index=True)
    quantity_received = db.Column(db.Integer, nullable=False)
    quantity_remaining = db.Column(db.Integer, nullable=False)
    unit_cost = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default="ACTIVE")  # ACTIVE, DEPLETED, EXPIRED, QUARANTINED
    
    # AI Expiry Risk Scoring fields (populated / updated by AI Expiry service)
    risk_score = db.Column(db.Float, default=0.0)      # 0 - 100
    risk_bucket = db.Column(db.String(20), default="SAFE")  # SAFE, WATCH, HIGH, CRITICAL
    value_at_risk = db.Column(db.Float, default=0.0)
    last_risk_assessed_at = db.Column(db.DateTime, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    medicine = db.relationship("Medicine", back_populates="batches")
    supplier = db.relationship("Supplier", back_populates="batches")
    sale_items = db.relationship("SaleItem", back_populates="batch", lazy="dynamic")

    @property
    def days_to_expiry(self) -> int:
        return (self.expiry_date - date.today()).days

    @property
    def is_expired(self) -> bool:
        return self.days_to_expiry <= 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "batch_number": self.batch_number,
            "medicine_id": self.medicine_id,
            "medicine_name": self.medicine.name if self.medicine else None,
            "atc_group": self.medicine.atc_group if self.medicine else None,
            "unit_price": self.medicine.unit_price if self.medicine else None,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier.name if self.supplier else "Standard Depot",
            "mfg_date": self.mfg_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat(),
            "days_to_expiry": self.days_to_expiry,
            "is_expired": self.is_expired,
            "quantity_received": self.quantity_received,
            "quantity_remaining": self.quantity_remaining,
            "unit_cost": self.unit_cost,
            "status": "EXPIRED" if self.is_expired and self.quantity_remaining > 0 else self.status,
            "risk_score": round(self.risk_score, 1),
            "risk_bucket": self.risk_bucket,
            "value_at_risk": round(self.value_at_risk, 2),
            "last_risk_assessed_at": self.last_risk_assessed_at.isoformat() if self.last_risk_assessed_at else None,
        }


class Sale(db.Model):
    __tablename__ = "sales"

    id = db.Column(db.Integer, primary_key=True)
    invoice_no = db.Column(db.String(64), unique=True, nullable=False, index=True)
    cashier_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    customer_name = db.Column(db.String(120), default="Walk-in Customer")
    customer_phone = db.Column(db.String(30), nullable=True)
    total_amount = db.Column(db.Float, nullable=False, default=0.0)
    discount = db.Column(db.Float, default=0.0)
    net_amount = db.Column(db.Float, nullable=False, default=0.0)
    payment_method = db.Column(db.String(30), default="CASH")  # CASH, CARD, UPI
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    cashier = db.relationship("User", back_populates="sales")
    items = db.relationship("SaleItem", back_populates="sale", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "invoice_no": self.invoice_no,
            "cashier": self.cashier.username if self.cashier else "System",
            "customer_name": self.customer_name,
            "customer_phone": self.customer_phone,
            "total_amount": round(self.total_amount, 2),
            "discount": round(self.discount, 2),
            "net_amount": round(self.net_amount, 2),
            "payment_method": self.payment_method,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "items": [item.to_dict() for item in self.items],
        }


class SaleItem(db.Model):
    __tablename__ = "sale_items"

    id = db.Column(db.Integer, primary_key=True)
    sale_id = db.Column(db.Integer, db.ForeignKey("sales.id"), nullable=False)
    batch_id = db.Column(db.Integer, db.ForeignKey("batches.id"), nullable=False)
    medicine_id = db.Column(db.Integer, db.ForeignKey("medicines.id"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    subtotal = db.Column(db.Float, nullable=False)

    sale = db.relationship("Sale", back_populates="items")
    batch = db.relationship("Batch", back_populates="sale_items")
    medicine = db.relationship("Medicine")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "medicine_id": self.medicine_id,
            "medicine_name": self.medicine.name if self.medicine else None,
            "batch_id": self.batch_id,
            "batch_number": self.batch.batch_number if self.batch else None,
            "quantity": self.quantity,
            "unit_price": round(self.unit_price, 2),
            "subtotal": round(self.subtotal, 2),
        }


class StockAdjustment(db.Model):
    __tablename__ = "stock_adjustments"

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey("batches.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    change_qty = db.Column(db.Integer, nullable=False)  # negative for loss/disposal, positive for found
    reason = db.Column(db.String(50), nullable=False)   # EXPIRED_DISPOSAL, DAMAGED, AUDIT_CORRECTION
    note = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    batch = db.relationship("Batch")
    user = db.relationship("User")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "batch_number": self.batch.batch_number if self.batch else None,
            "change_qty": self.change_qty,
            "reason": self.reason,
            "note": self.note,
            "user": self.user.username if self.user else "System",
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(120), default="Patient")
    patient_age = db.Column(db.Integer, nullable=True)
    patient_gender = db.Column(db.String(10), nullable=True)
    glucose = db.Column(db.Float, nullable=True)
    bmi = db.Column(db.Float, nullable=True)
    blood_pressure = db.Column(db.String(20), nullable=True)
    symptoms_json = db.Column(db.Text, nullable=False)  # list of strings in JSON
    
    predicted_disease = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    urgency = db.Column(db.String(20), default="moderate")
    first_line_treatment = db.Column(db.String(255), nullable=True)
    recommended_meds_json = db.Column(db.Text, nullable=True)
    specialist = db.Column(db.String(120), nullable=True)
    department = db.Column(db.String(120), nullable=True)
    red_flags_json = db.Column(db.Text, nullable=True)
    is_emergency = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        import json
        return {
            "id": self.id,
            "patient_name": self.patient_name,
            "patient_age": self.patient_age,
            "patient_gender": self.patient_gender,
            "glucose": self.glucose,
            "bmi": self.bmi,
            "blood_pressure": self.blood_pressure,
            "symptoms": json.loads(self.symptoms_json) if self.symptoms_json else [],
            "predicted_disease": self.predicted_disease,
            "confidence": round(self.confidence * 100, 1),
            "urgency": self.urgency,
            "first_line_treatment": self.first_line_treatment,
            "recommended_medications": json.loads(self.recommended_meds_json) if self.recommended_meds_json else [],
            "specialist": self.specialist,
            "department": self.department,
            "red_flags": json.loads(self.red_flags_json) if self.red_flags_json else [],
            "is_emergency": self.is_emergency,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
