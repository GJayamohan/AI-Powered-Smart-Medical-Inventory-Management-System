"""
AI Pharmacy Conversational Assistant.

Operates with dual capability:
1. Local deterministic knowledge-graph & database query engine (100% offline, 0 cost, 0 latency).
   Understands stock levels, near-expiry alerts, FEFO rules, disease treatments, and specialists.
2. Optional free-tier LLM integration (Google Gemini / Groq) if an API key is configured.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Optional

from medismart.db import db
from medismart.db.models import Batch, Medicine, PredictionLog
from medismart.services.treatment import TreatmentService


class PharmacyAssistant:
    def __init__(self):
        self.treatment_svc = TreatmentService()
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def answer_query(self, message: str) -> dict:
        text = message.strip().lower()

        # 1. Low stock queries
        if any(w in text for w in ["low stock", "shortage", "out of stock", "reorder"]):
            return self._handle_low_stock()

        # 2. Expiry / near-expiry queries
        if any(w in text for w in ["expir", "near expiry", "critical batch", "shelf life", "waste"]):
            return self._handle_expiry_queries(text)

        # 3. Specific Medicine Stock queries (e.g. "stock of paracetamol", "do we have diclofenac")
        med_match = self._find_medicine_in_query(text)
        if med_match:
            return self._handle_medicine_stock(med_match)

        # 4. Disease Treatment / Specialist queries (e.g. "treatment for diabetes", "doctor for migraine")
        disease_match = self._find_disease_in_query(text)
        if disease_match:
            return self._handle_disease_query(disease_match, text)

        # 5. Inventory summary / total value
        if any(w in text for w in ["total stock", "total inventory", "inventory value", "how many medicines"]):
            return self._handle_inventory_summary()

        # 6. Explanation queries (FEFO, AI Model, How it works)
        if "fefo" in text:
            return {
                "reply": (
                    "**FEFO (First-Expiry-First-Out)** is the clinical inventory algorithm used by MediSmart. "
                    "When any medicine is sold, the system automatically allocates and deducts stock from the "
                    "batch with the **earliest expiry date first**. This prevents older stock from sitting in the "
                    "depot and minimizes pharmaceutical wastage."
                ),
                "intent": "fefo_explanation",
            }

        if ("worked" in text and "example" in text) or ("glucose" in text and "bmi" in text):
            return {
                "reply": (
                    "**Worked Diagnostic Example:**\n"
                    "• **Input Vitals:** Age: 45, Glucose: 180 mg/dL, BMI: 29, BP: 140/90\n"
                    "• **Symptoms:** Increased thirst (polydipsia), frequent urination (polyuria), fatigue\n"
                    "• **Model Prediction:** Diabetes (Confidence: 100% via Multimodal Fusion)\n"
                    "• **First-Line Medication:** Metformin 500mg twice daily\n"
                    "• **Specialist to Consult:** Endocrinologist / Diabetologist\n\n"
                    "You can test this right now by visiting the **Disease Prediction** page and clicking 'Load Worked Example'!"
                ),
                "intent": "worked_example",
            }

        # 7. Fallback: Friendly guidance
        return {
            "reply": (
                "Hello! I am your **MediSmart AI Pharmacy Assistant**. I can help you with:\n\n"
                "• **Stock inquiries**: *'Do we have Diclofenac in stock?'* or *'Show low stock medicines'*\n"
                "• **Expiry risks**: *'Which batches are expiring soon?'* or *'What is our value at risk?'*\n"
                "• **Clinical guidance**: *'What medication is recommended for Asthma?'* or *'Which doctor to consult for Diabetes?'*\n"
                "• **System processes**: *'Explain FEFO dispensing'* or *'Show worked example'*\n\n"
                "What would you like to check?"
            ),
            "intent": "help",
        }

    def _handle_low_stock(self) -> dict:
        meds = Medicine.query.all()
        low = [m for m in meds if m.total_stock <= m.min_stock_alert]
        if not low:
            return {
                "reply": "✅ All catalog medicines currently have healthy stock levels above minimum thresholds.",
                "intent": "low_stock",
            }

        lines = [f"⚠️ **Found {len(low)} medicine(s) with low stock:**\n"]
        for m in sorted(low, key=lambda x: x.total_stock)[:8]:
            lines.append(f"• **{m.name}** (ATC: {m.atc_group}): Current stock is **{m.total_stock} units** (Alert threshold: {m.min_stock_alert})")

        lines.append("\n*Tip: Go to Inventory Management to generate supplier purchase orders.*")
        return {"reply": "\n".join(lines), "intent": "low_stock"}

    def _handle_expiry_queries(self, text: str) -> dict:
        today = date.today()
        batches = Batch.query.filter(Batch.status == "ACTIVE", Batch.quantity_remaining > 0).all()

        near_exp = [b for b in batches if (b.expiry_date - today).days <= 30]
        critical = [b for b in batches if b.risk_bucket in ("CRITICAL", "HIGH")]

        total_var = sum(b.value_at_risk for b in batches)

        lines = [
            f"📊 **AI Expiry Risk Overview:**\n",
            f"• **Near-Expiry Batches (≤30 days):** {len(near_exp)} batches",
            f"• **AI High/Critical Risk Batches:** {len(critical)} batches",
            f"• **Total Value at Risk:** Rs {total_var:,.2f}\n",
        ]

        if near_exp:
            lines.append("**Top Near-Expiry Batches to Action:**")
            for b in sorted(near_exp, key=lambda x: x.days_to_expiry)[:5]:
                med_name = b.medicine.name if b.medicine else "Medicine"
                lines.append(f"• **{med_name}** [Batch #{b.batch_number}]: {b.quantity_remaining} units left — Expiring in **{b.days_to_expiry} days** (Risk: {b.risk_bucket})")

        lines.append("\n*Recommendation: Prioritize these batches in FEFO dispensing or review supplier return policies.*")
        return {"reply": "\n".join(lines), "intent": "expiry_summary"}

    def _find_medicine_in_query(self, text: str) -> Optional[Medicine]:
        meds = Medicine.query.all()
        for m in meds:
            # Check for words in medicine name (e.g. "diclofenac", "ibuprofen", "paracetamol")
            name_words = m.name.lower().split()
            first_word = name_words[0] if name_words else ""
            if first_word and len(first_word) >= 4 and first_word in text:
                return m
            if m.name.lower() in text:
                return m
        return None

    def _handle_medicine_stock(self, med: Medicine) -> dict:
        today = date.today()
        active_batches = (
            Batch.query.filter(
                Batch.medicine_id == med.id,
                Batch.status == "ACTIVE",
                Batch.quantity_remaining > 0,
                Batch.expiry_date >= today
            )
            .order_by(Batch.expiry_date.asc())
            .all()
        )

        total_stock = sum(b.quantity_remaining for b in active_batches)
        earliest = active_batches[0] if active_batches else None

        lines = [
            f"📦 **Stock Report: {med.name}**\n",
            f"• **Category:** {med.category} (ATC: `{med.atc_group}`)",
            f"• **Unit Price:** Rs {med.unit_price:.2f} | Pack Size: {med.pack_size}",
            f"• **Total Available Stock:** **{total_stock} units** across {len(active_batches)} active batches",
            f"• **Stock Status:** {'🔴 Low Stock' if total_stock <= med.min_stock_alert else '🟢 In Stock'}",
        ]

        if earliest:
            lines.append(
                f"• **Next FEFO Batch:** Batch #{earliest.batch_number} ({earliest.quantity_remaining} units, Exp: {earliest.expiry_date.isoformat()}, {earliest.days_to_expiry} days left)"
            )

        return {"reply": "\n".join(lines), "intent": "medicine_stock"}

    def _find_disease_in_query(self, text: str) -> Optional[str]:
        for d in self.treatment_svc.disease_names:
            clean = d.lower().strip()
            # Handle variations like "diabetes", "heart attack", "asthma", "malaria"
            if clean in text:
                return d
            # Single keyword match for distinctive terms
            words = clean.split()
            for w in words:
                if len(w) >= 5 and w in text:
                    return d
        return None

    def _handle_disease_query(self, disease: str, text: str) -> dict:
        rec = self.treatment_svc.get_recommendation(disease)
        if not rec:
            return {"reply": f"No treatment entry found for '{disease}'.", "intent": "disease"}

        lines = [
            f"🩺 **Clinical Guidance: {rec['name'].strip()}**\n",
            f"• **Condition Overview:** {rec.get('description', '')}\n",
            f"• **Recommended Doctor to Consult:** **{rec['specialist']}** ({rec['department']})",
            f"• **First-Line Treatment:** {rec['first_line']}",
            f"• **Recommended Medication Classes:** {', '.join(rec['medication_classes'])}",
            f"• **Urgency Level:** {rec['urgency'].upper()}",
        ]

        if rec.get("precautions"):
            lines.append(f"• **Key Precautions:** {', '.join(rec['precautions'][:3])}")

        if rec.get("red_flags"):
            lines.append(f"\n⚠️ **Warning Signs to Watch For:**\n" + "\n".join(f"  - {rf}" for rf in rec["red_flags"][:3]))

        lines.append(f"\n*{rec['disclaimer']}*")
        return {"reply": "\n".join(lines), "intent": "disease_treatment"}
