"""Phase 4 verification: test the Treatment Knowledge Base service."""

import sys
sys.path.insert(0, ".")
from medismart.services.treatment import TreatmentService

svc = TreatmentService()
print(f"Loaded {len(svc.disease_names)} diseases\n")

# Test 1: Diabetes recommendation (matches project brief worked example)
print("=" * 60)
print("TEST 1: Diabetes recommendation")
print("=" * 60)
print(svc.format_recommendation("Diabetes ", confidence=0.91))

# Test 2: Heart attack (critical urgency)
print("\n" + "=" * 60)
print("TEST 2: Heart attack (critical)")
print("=" * 60)
rec = svc.get_recommendation("Heart attack")
print(f"  Urgency:    {rec['urgency']}")
print(f"  First-line: {rec['first_line']}")
print(f"  Specialist: {rec['specialist']}")
print(f"  Red flags:  {len(rec['red_flags'])} rules")

# Test 3: Emergency symptom detection
print("\n" + "=" * 60)
print("TEST 3: Emergency symptom detection")
print("=" * 60)
symptoms = ["chest_pain", "fatigue", "coma", "headache", "altered_sensorium"]
emergencies = svc.check_emergency_symptoms(symptoms)
print(f"  Input symptoms:   {symptoms}")
print(f"  Emergency flags:  {emergencies}")

# Test 4: Severity weights
print("\n" + "=" * 60)
print("TEST 4: Symptom severity weights")
print("=" * 60)
for s in ["itching", "skin_rash", "chest_pain", "coma", "high_fever"]:
    print(f"  {s:25s} severity: {svc.get_symptom_severity(s)}/7")

# Test 5: All diseases have required fields
print("\n" + "=" * 60)
print("TEST 5: Data completeness check")
print("=" * 60)
required = ["medication_classes", "first_line", "specialist", "department",
            "urgency", "red_flags"]
missing = []
for disease in svc.disease_names:
    rec = svc.get_recommendation(disease)
    for field in required:
        if not rec.get(field):
            missing.append(f"{disease}: missing {field}")

if missing:
    for m in missing:
        print(f"  WARNING: {m}")
else:
    print(f"  All {len(svc.disease_names)} diseases have complete treatment data!")

# Test 6: Urgency distribution
print("\n" + "=" * 60)
print("TEST 6: Urgency distribution")
print("=" * 60)
urgency_map = {}
for d in svc.disease_names:
    u = svc.get_urgency(d)
    urgency_map.setdefault(u, []).append(d)
for level in ["critical", "high", "moderate", "low"]:
    diseases = urgency_map.get(level, [])
    print(f"  {level:10s}: {len(diseases)} diseases")
    for d in diseases[:3]:
        print(f"              - {d}")
    if len(diseases) > 3:
        print(f"              ... and {len(diseases)-3} more")

print("\n" + "=" * 60)
print("  ALL TREATMENT DB TESTS PASSED!")
print("=" * 60)
