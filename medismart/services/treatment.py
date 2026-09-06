"""
Treatment Knowledge Base service.

Provides a clean API for the backend to look up treatment recommendations,
medication classes, specialist referrals, and safety information for any
of the 41 diagnosed diseases.

Usage:
    from medismart.services.treatment import TreatmentService

    svc = TreatmentService()
    rec = svc.get_recommendation("Diabetes ")
    print(rec["first_line"])        # "Metformin 500mg twice daily..."
    print(rec["specialist"])        # "Endocrinologist / Diabetologist"
    print(svc.is_emergency("chest_pain"))  # True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from medismart.utils.paths import KNOWLEDGE


class TreatmentService:
    """Singleton-style service for treatment knowledge lookups."""

    def __init__(self, db_path: Optional[Path] = None):
        path = db_path or (KNOWLEDGE / "treatment_db.json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._diseases: dict = raw["diseases"]
        self._disclaimer: str = raw["disclaimer"]
        self._emergency_symptoms: list[str] = raw["emergency_symptoms"]
        self._severity: dict = raw.get("symptom_severity", {})

    @property
    def disclaimer(self) -> str:
        return self._disclaimer

    @property
    def disease_names(self) -> list[str]:
        return list(self._diseases.keys())

    def get_recommendation(self, disease: str) -> dict | None:
        """Full treatment recommendation for a disease.

        Returns a dict with keys: name, description, medication_classes,
        first_line, specialist, department, urgency, lifestyle, precautions,
        red_flags. Returns None if disease not found.
        """
        rec = self._diseases.get(disease)
        if rec is None:
            # Try stripping whitespace (some disease names have trailing spaces)
            for key in self._diseases:
                if key.strip() == disease.strip():
                    rec = self._diseases[key]
                    break
        if rec is None:
            return None

        # Always attach the disclaimer
        result = dict(rec)
        result["disclaimer"] = self._disclaimer
        return result

    def get_medication(self, disease: str) -> list[str]:
        """Medication classes for a disease."""
        rec = self.get_recommendation(disease)
        return rec["medication_classes"] if rec else []

    def get_specialist(self, disease: str) -> str:
        """Specialist doctor to consult."""
        rec = self.get_recommendation(disease)
        return rec["specialist"] if rec else "General Physician"

    def get_urgency(self, disease: str) -> str:
        """Urgency level: critical, high, moderate, low."""
        rec = self.get_recommendation(disease)
        return rec["urgency"] if rec else "moderate"

    def is_emergency(self, symptom: str) -> bool:
        """Check if a symptom triggers emergency escalation."""
        return symptom in self._emergency_symptoms

    def check_emergency_symptoms(self, symptoms: list[str]) -> list[str]:
        """Return which of the given symptoms are emergency red flags."""
        return [s for s in symptoms if self.is_emergency(s)]

    def get_symptom_severity(self, symptom: str) -> int:
        """Severity weight (1-7) for a symptom. Returns 0 if unknown."""
        return self._severity.get(symptom, 0)

    def format_recommendation(self, disease: str, confidence: float = 0.0) -> str:
        """Format a human-readable recommendation string for display."""
        rec = self.get_recommendation(disease)
        if rec is None:
            return f"No treatment information available for '{disease}'."

        lines = []
        lines.append(f"Predicted Condition: {rec['name']}")
        if confidence > 0:
            lines.append(f"Model Confidence: {confidence*100:.1f}%")
        lines.append("")

        if rec.get("description"):
            lines.append(f"About: {rec['description']}")
            lines.append("")

        urgency_emoji = {
            "critical": "[!!!] CRITICAL",
            "high": "[!!] HIGH",
            "moderate": "[!] MODERATE",
            "low": "[OK] LOW",
        }
        lines.append(f"Urgency: {urgency_emoji.get(rec['urgency'], rec['urgency'])}")
        lines.append("")

        lines.append(f"Recommended Medication Class:")
        for med in rec["medication_classes"]:
            lines.append(f"  - {med}")
        lines.append(f"First-line: {rec['first_line']}")
        lines.append("")

        lines.append(f"Consult: {rec['specialist']}")
        lines.append(f"Department: {rec['department']}")
        lines.append("")

        if rec.get("precautions"):
            lines.append("Precautions:")
            for p in rec["precautions"]:
                lines.append(f"  - {p}")
            lines.append("")

        if rec.get("red_flags"):
            lines.append("Warning Signs (seek immediate help if):")
            for rf in rec["red_flags"]:
                lines.append(f"  ! {rf}")
            lines.append("")

        lines.append(f"NOTE: {self._disclaimer}")

        return "\n".join(lines)
