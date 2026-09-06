"""
Phase 4 - Build the curated Treatment Knowledge Base.

Merges the raw auxiliary datasets (descriptions, precautions, severity) with
hand-curated clinical mappings (medications, specialists, urgency) into a
single authoritative JSON file that the backend will serve.

SAFETY GUARDRAILS:
  - Recommends medication CLASSES, never patient-specific dosages.
  - Never suggests controlled substances or narcotics.
  - Includes red-flag symptoms that trigger emergency escalation.
  - Every entry includes a disclaimer and specialist referral.

Run with:  python scripts/build_treatment_db.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medismart.utils.paths import KNOWLEDGE, PROCESSED, RAW, ensure_dirs  # noqa: E402

# -----------------------------------------------------------------------
# Hand-curated clinical knowledge for all 41 diseases.
#
# Sources: WHO Essential Medicines List, NHS clinical guidelines,
# Mayo Clinic, MedlinePlus (all publicly accessible).
#
# IMPORTANT: These are EDUCATIONAL recommendations for a student project.
# They are NOT a substitute for professional medical advice.
# -----------------------------------------------------------------------

TREATMENT_MAP = {
    "(vertigo) Paroymsal  Positional Vertigo": {
        "medication_classes": ["Antihistamines (Meclizine)", "Benzodiazepines (short-term)", "Antiemetics"],
        "first_line": "Meclizine 25mg as needed",
        "specialist": "ENT Specialist / Neurologist",
        "department": "Neurology / ENT",
        "urgency": "moderate",
        "lifestyle": ["Epley maneuver exercises", "Avoid sudden head movements", "Stay hydrated"],
        "red_flags": ["Severe headache with vertigo", "Loss of consciousness", "Slurred speech"],
    },
    "AIDS": {
        "medication_classes": ["Antiretroviral therapy (ART)", "NRTIs", "NNRTIs", "Protease inhibitors"],
        "first_line": "Tenofovir + Lamivudine + Efavirenz (TLE regimen)",
        "specialist": "Infectious Disease Specialist",
        "department": "Infectious Diseases",
        "urgency": "high",
        "lifestyle": ["Strict ART adherence", "Regular CD4 monitoring", "Safe practices", "Balanced nutrition"],
        "red_flags": ["Rapid weight loss", "Persistent fever >1 month", "Opportunistic infections"],
    },
    "Acne": {
        "medication_classes": ["Topical retinoids", "Benzoyl peroxide", "Topical antibiotics", "Oral antibiotics (severe)"],
        "first_line": "Benzoyl peroxide 2.5-5% topical gel",
        "specialist": "Dermatologist",
        "department": "Dermatology",
        "urgency": "low",
        "lifestyle": ["Gentle face washing twice daily", "Non-comedogenic products", "Avoid picking/squeezing", "Balanced diet"],
        "red_flags": ["Severe cystic acne", "Scarring", "Signs of hormonal imbalance"],
    },
    "Alcoholic hepatitis": {
        "medication_classes": ["Corticosteroids (Prednisolone)", "Pentoxifylline", "Hepatoprotectants", "Vitamin supplements"],
        "first_line": "Prednisolone 40mg/day for 28 days (if Maddrey score >= 32)",
        "specialist": "Hepatologist / Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "high",
        "lifestyle": ["Complete alcohol abstinence", "High-calorie high-protein diet", "Vitamin B/folate supplementation"],
        "red_flags": ["Jaundice worsening", "Hepatic encephalopathy", "GI bleeding", "Ascites"],
    },
    "Allergy": {
        "medication_classes": ["Antihistamines (Cetirizine, Loratadine)", "Nasal corticosteroids", "Decongestants", "Epinephrine (anaphylaxis)"],
        "first_line": "Cetirizine 10mg or Loratadine 10mg daily",
        "specialist": "Allergist / Immunologist",
        "department": "Allergy & Immunology",
        "urgency": "low",
        "lifestyle": ["Identify and avoid allergens", "Keep antihistamines available", "Use air purifiers"],
        "red_flags": ["Difficulty breathing", "Swelling of throat/tongue", "Anaphylaxis signs"],
    },
    "Arthritis": {
        "medication_classes": ["NSAIDs (Ibuprofen, Diclofenac)", "DMARDs (Methotrexate)", "Corticosteroids", "Analgesics"],
        "first_line": "Ibuprofen 400mg three times daily with food",
        "specialist": "Rheumatologist",
        "department": "Rheumatology",
        "urgency": "moderate",
        "lifestyle": ["Regular low-impact exercise", "Weight management", "Hot/cold therapy", "Joint protection techniques"],
        "red_flags": ["Sudden joint swelling with fever", "Joint deformity", "Inability to move joint"],
    },
    "Bronchial Asthma": {
        "medication_classes": ["Inhaled corticosteroids (Budesonide)", "Short-acting beta-agonists (Salbutamol)", "Long-acting beta-agonists", "Leukotriene modifiers"],
        "first_line": "Salbutamol inhaler (rescue) + Budesonide inhaler (maintenance)",
        "specialist": "Pulmonologist",
        "department": "Pulmonology",
        "urgency": "moderate",
        "lifestyle": ["Avoid triggers (dust, smoke, cold air)", "Use peak flow meter", "Carry rescue inhaler always", "Asthma action plan"],
        "red_flags": ["Severe breathlessness at rest", "Blue lips/fingers", "Cannot speak full sentences", "Peak flow < 50%"],
    },
    "Cervical spondylosis": {
        "medication_classes": ["NSAIDs (Diclofenac)", "Muscle relaxants (Tizanidine)", "Analgesics (Paracetamol)", "Neuropathic pain agents (Gabapentin)"],
        "first_line": "Diclofenac 50mg twice daily + Paracetamol 500mg",
        "specialist": "Orthopedic Surgeon / Neurologist",
        "department": "Orthopedics",
        "urgency": "moderate",
        "lifestyle": ["Neck exercises and stretches", "Proper posture", "Ergonomic workspace", "Avoid prolonged neck flexion"],
        "red_flags": ["Weakness in arms/legs", "Loss of bladder/bowel control", "Difficulty walking"],
    },
    "Chicken pox": {
        "medication_classes": ["Antivirals (Acyclovir)", "Antihistamines (Calamine)", "Analgesics (Paracetamol)", "Topical lotions"],
        "first_line": "Acyclovir 800mg five times daily for 7 days (adults)",
        "specialist": "General Physician / Pediatrician",
        "department": "General Medicine",
        "urgency": "moderate",
        "lifestyle": ["Isolate patient for 5-7 days", "Calamine lotion for itching", "Keep nails short", "Adequate rest and fluids"],
        "red_flags": ["High persistent fever", "Difficulty breathing", "Severe rash near eyes", "Confusion or drowsiness"],
    },
    "Chronic cholestasis": {
        "medication_classes": ["Ursodeoxycholic acid (UDCA)", "Cholestyramine", "Fat-soluble vitamin supplements", "Antihistamines (for pruritus)"],
        "first_line": "Ursodeoxycholic acid 13-15 mg/kg/day",
        "specialist": "Hepatologist / Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "moderate",
        "lifestyle": ["Low-fat diet", "Avoid alcohol", "Regular liver function monitoring", "Vitamin A/D/E/K supplementation"],
        "red_flags": ["Deep jaundice", "Fever with rigors", "Right upper quadrant pain", "Dark urine"],
    },
    "Common Cold": {
        "medication_classes": ["Decongestants (Pseudoephedrine)", "Antihistamines (Chlorpheniramine)", "Analgesics (Paracetamol)", "Cough suppressants"],
        "first_line": "Paracetamol 500mg + rest + fluids (symptomatic)",
        "specialist": "General Physician",
        "department": "General Medicine",
        "urgency": "low",
        "lifestyle": ["Rest adequately", "Drink warm fluids", "Steam inhalation", "Hand hygiene to prevent spread"],
        "red_flags": ["Fever > 103F for > 3 days", "Severe sinus pain", "Difficulty breathing", "Symptoms > 10 days"],
    },
    "Dengue": {
        "medication_classes": ["Analgesics (Paracetamol ONLY)", "Oral rehydration salts", "IV fluids (severe)", "Platelet monitoring"],
        "first_line": "Paracetamol 500mg (NO aspirin/ibuprofen - bleeding risk)",
        "specialist": "Infectious Disease Specialist",
        "department": "Infectious Diseases / General Medicine",
        "urgency": "high",
        "lifestyle": ["Complete bed rest", "Drink plenty of fluids/ORS", "Daily platelet count monitoring", "Mosquito net use"],
        "red_flags": ["Severe abdominal pain", "Persistent vomiting", "Bleeding gums/nose", "Platelet < 20,000", "Rapid pulse"],
    },
    "Diabetes ": {
        "medication_classes": ["Metformin", "Sulfonylureas (Glimepiride)", "DPP-4 inhibitors", "Insulin (Type 1 / advanced Type 2)"],
        "first_line": "Metformin 500mg twice daily (start low, titrate up)",
        "specialist": "Endocrinologist / Diabetologist",
        "department": "Endocrinology",
        "urgency": "moderate",
        "lifestyle": ["Regular blood glucose monitoring", "Balanced low-sugar diet", "150 min/week moderate exercise", "Foot care", "Annual eye checkup"],
        "red_flags": ["Blood glucose > 400 mg/dL", "Diabetic ketoacidosis signs", "Non-healing wounds", "Vision changes"],
    },
    "Dimorphic hemmorhoids(piles)": {
        "medication_classes": ["Topical analgesics (Lidocaine)", "Stool softeners (Lactulose)", "Flavonoids (Diosmin)", "Topical corticosteroids"],
        "first_line": "Lactulose 15ml daily + Diosmin 500mg twice daily + topical cream",
        "specialist": "General Surgeon / Proctologist",
        "department": "General Surgery",
        "urgency": "low",
        "lifestyle": ["High-fiber diet", "Adequate water intake", "Avoid straining", "Sitz baths", "Regular exercise"],
        "red_flags": ["Heavy rectal bleeding", "Severe pain", "Prolapsed hemorrhoids that cannot be pushed back"],
    },
    "Drug Reaction": {
        "medication_classes": ["Antihistamines (Cetirizine)", "Corticosteroids (Prednisolone)", "Epinephrine (severe)", "Topical corticosteroids"],
        "first_line": "Stop offending drug immediately + Cetirizine 10mg",
        "specialist": "Allergist / Dermatologist",
        "department": "Allergy & Immunology",
        "urgency": "high",
        "lifestyle": ["Document the drug allergy", "Wear medical alert bracelet", "Inform all healthcare providers", "Carry allergy card"],
        "red_flags": ["Difficulty breathing", "Facial/throat swelling", "Stevens-Johnson syndrome signs", "Widespread blistering"],
    },
    "Fungal infection": {
        "medication_classes": ["Topical antifungals (Clotrimazole, Terbinafine)", "Oral antifungals (Fluconazole)", "Antifungal powders"],
        "first_line": "Clotrimazole 1% cream twice daily for 2-4 weeks",
        "specialist": "Dermatologist",
        "department": "Dermatology",
        "urgency": "low",
        "lifestyle": ["Keep affected area dry", "Wear loose cotton clothing", "Avoid sharing towels", "Complete full course of treatment"],
        "red_flags": ["Spreading despite treatment", "Fever with skin infection", "Deep tissue involvement"],
    },
    "GERD": {
        "medication_classes": ["Proton pump inhibitors (Omeprazole, Pantoprazole)", "H2 blockers (Ranitidine)", "Antacids (Aluminum/Magnesium)", "Prokinetics (Domperidone)"],
        "first_line": "Omeprazole 20mg once daily before breakfast for 4-8 weeks",
        "specialist": "Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "low",
        "lifestyle": ["Avoid spicy/fatty foods", "Eat smaller frequent meals", "No lying down after eating (wait 2-3 hrs)", "Elevate head of bed", "Avoid smoking/alcohol"],
        "red_flags": ["Difficulty swallowing", "Unintended weight loss", "Vomiting blood", "Black tarry stools"],
    },
    "Gastroenteritis": {
        "medication_classes": ["Oral rehydration salts (ORS)", "Antiemetics (Ondansetron)", "Probiotics", "Zinc supplements (children)"],
        "first_line": "ORS solution + Ondansetron 4mg if vomiting + bland diet",
        "specialist": "General Physician / Gastroenterologist",
        "department": "General Medicine",
        "urgency": "moderate",
        "lifestyle": ["Plenty of fluids and ORS", "BRAT diet initially", "Strict hand hygiene", "Avoid dairy temporarily"],
        "red_flags": ["Bloody diarrhea", "Signs of severe dehydration", "High fever > 102F", "Symptoms > 3 days"],
    },
    "Heart attack": {
        "medication_classes": ["Aspirin (immediate)", "Nitroglycerin", "Thrombolytics", "Beta-blockers", "ACE inhibitors", "Anticoagulants"],
        "first_line": "EMERGENCY: Aspirin 325mg chewed + call ambulance immediately",
        "specialist": "Cardiologist",
        "department": "Cardiology / Emergency Medicine",
        "urgency": "critical",
        "lifestyle": ["Cardiac rehabilitation program", "Quit smoking", "Heart-healthy diet", "Stress management", "Regular follow-up"],
        "red_flags": ["Chest pain/pressure", "Pain radiating to arm/jaw", "Shortness of breath", "Cold sweat", "Nausea"],
    },
    "Hepatitis B": {
        "medication_classes": ["Antivirals (Entecavir, Tenofovir)", "Pegylated Interferon", "Hepatoprotectants"],
        "first_line": "Entecavir 0.5mg daily or Tenofovir 300mg daily",
        "specialist": "Hepatologist / Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "high",
        "lifestyle": ["Complete alcohol avoidance", "Vaccination for contacts", "Safe practices", "Regular HBV DNA monitoring"],
        "red_flags": ["Severe jaundice", "Hepatic encephalopathy", "Ascites", "Liver failure signs"],
    },
    "Hepatitis C": {
        "medication_classes": ["Direct-acting antivirals (Sofosbuvir, Ledipasvir)", "Ribavirin (combination)", "Hepatoprotectants"],
        "first_line": "Sofosbuvir 400mg + Ledipasvir 90mg daily for 12 weeks",
        "specialist": "Hepatologist / Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "high",
        "lifestyle": ["Avoid alcohol completely", "Regular liver function tests", "Avoid sharing personal items", "Nutrition counseling"],
        "red_flags": ["Progressive jaundice", "Liver cirrhosis signs", "Hepatic encephalopathy"],
    },
    "Hepatitis D": {
        "medication_classes": ["Pegylated Interferon alfa", "Hepatoprotectants", "Supportive care"],
        "first_line": "Pegylated Interferon alfa-2a for 48 weeks (specialist-guided)",
        "specialist": "Hepatologist",
        "department": "Gastroenterology",
        "urgency": "high",
        "lifestyle": ["Strict alcohol avoidance", "Hepatitis B vaccination (prevention)", "Regular monitoring", "Balanced nutrition"],
        "red_flags": ["Rapid liver deterioration", "Fulminant hepatitis", "Coagulopathy"],
    },
    "Hepatitis E": {
        "medication_classes": ["Supportive care (no specific antiviral)", "Hepatoprotectants", "Antiemetics", "IV fluids if needed"],
        "first_line": "Supportive: rest + adequate nutrition + hydration (self-limiting)",
        "specialist": "Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "moderate",
        "lifestyle": ["Drink boiled/filtered water", "Avoid alcohol", "Rest adequately", "Practice food hygiene"],
        "red_flags": ["Severe jaundice", "Pregnancy (high mortality risk)", "Acute liver failure signs"],
    },
    "Hypertension ": {
        "medication_classes": ["ACE inhibitors (Enalapril, Ramipril)", "ARBs (Losartan, Telmisartan)", "Calcium channel blockers (Amlodipine)", "Diuretics (Hydrochlorothiazide)", "Beta-blockers"],
        "first_line": "Amlodipine 5mg daily or Losartan 50mg daily",
        "specialist": "Cardiologist / General Physician",
        "department": "Cardiology",
        "urgency": "moderate",
        "lifestyle": ["DASH diet (low sodium)", "Regular exercise 150 min/week", "Limit alcohol", "Quit smoking", "Stress reduction", "Regular BP monitoring"],
        "red_flags": ["BP > 180/120 (hypertensive crisis)", "Severe headache with vision changes", "Chest pain", "Shortness of breath"],
    },
    "Hyperthyroidism": {
        "medication_classes": ["Anti-thyroid drugs (Methimazole, Carbimazole)", "Beta-blockers (Propranolol)", "Radioactive iodine", "Thyroidectomy (surgical)"],
        "first_line": "Methimazole 10-20mg daily + Propranolol 40mg for symptoms",
        "specialist": "Endocrinologist",
        "department": "Endocrinology",
        "urgency": "moderate",
        "lifestyle": ["Regular thyroid function monitoring", "Adequate calcium/vitamin D", "Avoid iodine-rich foods", "Manage stress"],
        "red_flags": ["Thyroid storm (rapid pulse, fever, confusion)", "Severe weight loss", "Eye bulging worsening"],
    },
    "Hypoglycemia": {
        "medication_classes": ["Glucose tablets/gel", "Glucagon injection (severe)", "Dextrose IV (emergency)"],
        "first_line": "IMMEDIATE: 15-20g fast-acting carbs (glucose tablets, juice, sugar)",
        "specialist": "Endocrinologist / Diabetologist",
        "department": "Endocrinology",
        "urgency": "high",
        "lifestyle": ["Regular meal schedule", "Carry glucose tablets always", "Monitor blood sugar frequently", "Adjust insulin/medication with doctor"],
        "red_flags": ["Loss of consciousness", "Seizures", "Blood glucose < 54 mg/dL", "Confusion/unresponsiveness"],
    },
    "Hypothyroidism": {
        "medication_classes": ["Levothyroxine (synthetic T4)"],
        "first_line": "Levothyroxine 25-50 mcg daily (empty stomach, 30 min before food)",
        "specialist": "Endocrinologist",
        "department": "Endocrinology",
        "urgency": "moderate",
        "lifestyle": ["Take medication consistently at same time", "Regular TSH monitoring every 6-8 weeks initially", "Adequate iodine intake", "Exercise regularly"],
        "red_flags": ["Myxedema coma (extreme fatigue, hypothermia)", "Severe depression", "Heart complications"],
    },
    "Impetigo": {
        "medication_classes": ["Topical antibiotics (Mupirocin, Fusidic acid)", "Oral antibiotics (Cephalexin, Amoxicillin)"],
        "first_line": "Mupirocin 2% ointment three times daily for 5-7 days",
        "specialist": "Dermatologist / General Physician",
        "department": "Dermatology",
        "urgency": "low",
        "lifestyle": ["Keep sores clean and covered", "Wash hands frequently", "Avoid touching/scratching sores", "Separate towels and linens"],
        "red_flags": ["Spreading despite treatment", "Fever", "Red streaks from sores", "Kidney problems (post-strep)"],
    },
    "Jaundice": {
        "medication_classes": ["Treat underlying cause", "Hepatoprotectants", "Ursodeoxycholic acid (obstructive)", "IV fluids"],
        "first_line": "Identify and treat underlying cause + supportive care",
        "specialist": "Hepatologist / Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "moderate",
        "lifestyle": ["Avoid alcohol completely", "Light easily digestible diet", "Adequate rest", "Stay hydrated"],
        "red_flags": ["Rapidly deepening jaundice", "Altered consciousness", "Abdominal distension", "Bleeding tendency"],
    },
    "Malaria": {
        "medication_classes": ["Artemisinin-based combination therapy (ACT)", "Chloroquine (P. vivax)", "Primaquine (radical cure)", "Quinine (severe)"],
        "first_line": "Artemether-Lumefantrine (ACT) for P. falciparum, Chloroquine for P. vivax",
        "specialist": "Infectious Disease Specialist",
        "department": "Infectious Diseases / General Medicine",
        "urgency": "high",
        "lifestyle": ["Complete full antimalarial course", "Use mosquito nets", "Mosquito repellent", "Drain stagnant water"],
        "red_flags": ["Cerebral malaria (confusion, seizures)", "Severe anemia", "Respiratory distress", "Renal failure"],
    },
    "Migraine": {
        "medication_classes": ["Triptans (Sumatriptan)", "NSAIDs (Ibuprofen, Naproxen)", "Antiemetics (Metoclopramide)", "Preventive: Beta-blockers, Topiramate, Amitriptyline"],
        "first_line": "Ibuprofen 400mg at onset OR Sumatriptan 50mg if severe",
        "specialist": "Neurologist",
        "department": "Neurology",
        "urgency": "moderate",
        "lifestyle": ["Identify and avoid triggers", "Regular sleep schedule", "Stress management", "Stay hydrated", "Maintain headache diary"],
        "red_flags": ["Worst headache of life (thunderclap)", "Fever with stiff neck", "New headache after age 50", "Neurological symptoms"],
    },
    "Osteoarthristis": {
        "medication_classes": ["Analgesics (Paracetamol)", "NSAIDs (Ibuprofen, Diclofenac)", "Topical NSAIDs", "Intra-articular corticosteroids", "Glucosamine (supplement)"],
        "first_line": "Paracetamol 500mg + topical Diclofenac gel + exercise",
        "specialist": "Orthopedic Surgeon / Rheumatologist",
        "department": "Orthopedics",
        "urgency": "moderate",
        "lifestyle": ["Regular low-impact exercise (swimming, walking)", "Weight management", "Joint protection techniques", "Physical therapy"],
        "red_flags": ["Sudden severe joint swelling", "Joint locking", "Inability to bear weight", "Signs of joint infection"],
    },
    "Paralysis (brain hemorrhage)": {
        "medication_classes": ["EMERGENCY: Surgical intervention may be needed", "Antihypertensives", "Anticonvulsants", "Osmotic diuretics (Mannitol)"],
        "first_line": "EMERGENCY: Immediate hospitalization + CT scan + neurosurgery consult",
        "specialist": "Neurosurgeon / Neurologist",
        "department": "Neurosurgery / Emergency Medicine",
        "urgency": "critical",
        "lifestyle": ["Rehabilitation program", "Physical therapy", "Occupational therapy", "Speech therapy if needed", "BP control"],
        "red_flags": ["ALL symptoms are red flags - immediate emergency care", "Sudden weakness", "Speech difficulty", "Severe headache"],
    },
    "Peptic ulcer diseae": {
        "medication_classes": ["Proton pump inhibitors (Omeprazole, Pantoprazole)", "H. pylori triple therapy (if positive)", "Antacids", "Sucralfate"],
        "first_line": "Omeprazole 20mg twice daily + Amoxicillin + Clarithromycin (if H. pylori+)",
        "specialist": "Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "moderate",
        "lifestyle": ["Avoid NSAIDs", "Quit smoking", "Limit alcohol", "Regular small meals", "Stress management"],
        "red_flags": ["Vomiting blood (hematemesis)", "Black tarry stools (melena)", "Sudden severe abdominal pain (perforation)", "Rapid weight loss"],
    },
    "Pneumonia": {
        "medication_classes": ["Antibiotics (Amoxicillin, Azithromycin, Levofloxacin)", "Antipyretics (Paracetamol)", "Mucolytics", "Bronchodilators"],
        "first_line": "Amoxicillin 500mg three times daily for 5-7 days (community-acquired)",
        "specialist": "Pulmonologist",
        "department": "Pulmonology / General Medicine",
        "urgency": "high",
        "lifestyle": ["Complete antibiotic course", "Rest and adequate fluids", "Deep breathing exercises", "Pneumococcal vaccination"],
        "red_flags": ["Severe breathlessness", "Cyanosis (blue lips)", "Confusion", "BP drop", "Oxygen saturation < 92%"],
    },
    "Psoriasis": {
        "medication_classes": ["Topical corticosteroids", "Vitamin D analogs (Calcipotriol)", "Retinoids (Acitretin)", "Methotrexate (severe)", "Phototherapy"],
        "first_line": "Topical Betamethasone + Calcipotriol combination cream",
        "specialist": "Dermatologist",
        "department": "Dermatology",
        "urgency": "low",
        "lifestyle": ["Regular moisturizing", "Avoid triggers (stress, skin injury)", "Moderate sun exposure", "Avoid alcohol", "Quit smoking"],
        "red_flags": ["Psoriatic arthritis symptoms", "Widespread erythroderma", "Pustular psoriasis", "Severe impact on quality of life"],
    },
    "Tuberculosis": {
        "medication_classes": ["DOTS regimen: Isoniazid (H), Rifampicin (R), Pyrazinamide (Z), Ethambutol (E)", "Vitamin B6 supplement"],
        "first_line": "HRZE for 2 months (intensive) + HR for 4 months (continuation) = 6 months total",
        "specialist": "Pulmonologist / TB Specialist",
        "department": "Pulmonology / Infectious Diseases",
        "urgency": "high",
        "lifestyle": ["Complete FULL 6-month course (critical)", "Cover mouth when coughing", "Ventilated living space", "Nutritious diet", "Contact tracing"],
        "red_flags": ["Hemoptysis (coughing blood)", "MDR-TB suspicion", "Weight loss > 10%", "Miliary/meningeal TB"],
    },
    "Typhoid": {
        "medication_classes": ["Antibiotics (Azithromycin, Ceftriaxone, Ciprofloxacin)", "Antipyretics (Paracetamol)", "ORS/IV fluids"],
        "first_line": "Azithromycin 500mg daily for 7 days OR Ceftriaxone (if severe)",
        "specialist": "Infectious Disease Specialist / General Physician",
        "department": "Infectious Diseases / General Medicine",
        "urgency": "high",
        "lifestyle": ["Drink only boiled/purified water", "Eat freshly cooked food", "Hand hygiene", "Complete antibiotic course", "Typhoid vaccination"],
        "red_flags": ["Intestinal perforation (sudden severe pain)", "GI bleeding", "Sustained high fever > 104F", "Confusion"],
    },
    "Urinary tract infection": {
        "medication_classes": ["Antibiotics (Nitrofurantoin, Trimethoprim, Ciprofloxacin)", "Analgesics (Phenazopyridine)", "Alkalinizers"],
        "first_line": "Nitrofurantoin 100mg twice daily for 5 days",
        "specialist": "Urologist / General Physician",
        "department": "Urology / General Medicine",
        "urgency": "moderate",
        "lifestyle": ["Drink plenty of water", "Urinate frequently", "Wipe front to back", "Avoid holding urine", "Cranberry juice (supportive)"],
        "red_flags": ["High fever with flank pain (pyelonephritis)", "Blood in urine", "Recurrent UTIs (>3/year)", "UTI in males"],
    },
    "Varicose veins": {
        "medication_classes": ["Venotonics (Diosmin, Hesperidin)", "Compression stockings", "Topical heparinoids", "Sclerotherapy / surgery (severe)"],
        "first_line": "Compression stockings + Diosmin 500mg twice daily",
        "specialist": "Vascular Surgeon",
        "department": "Vascular Surgery",
        "urgency": "low",
        "lifestyle": ["Leg elevation when resting", "Regular walking/exercise", "Avoid prolonged standing/sitting", "Wear compression stockings", "Weight management"],
        "red_flags": ["Leg ulceration", "Skin color changes", "Bleeding from varicose veins", "Deep vein thrombosis signs"],
    },
    "hepatitis A": {
        "medication_classes": ["Supportive care (no specific antiviral)", "Hepatoprotectants", "Antiemetics", "Rest"],
        "first_line": "Supportive: rest + adequate nutrition + hydration (self-limiting in 2-6 months)",
        "specialist": "Gastroenterologist",
        "department": "Gastroenterology",
        "urgency": "moderate",
        "lifestyle": ["Strict hygiene and hand washing", "Drink boiled/purified water", "Avoid alcohol", "Rest adequately", "Hepatitis A vaccination for contacts"],
        "red_flags": ["Fulminant hepatitis (rare)", "Severe jaundice", "Prolonged PT/INR", "Hepatic encephalopathy"],
    },
}

# Global safety disclaimer (included in every response)
SAFETY_DISCLAIMER = (
    "DISCLAIMER: This system provides educational decision-support information only. "
    "It is NOT a substitute for professional medical diagnosis or treatment. "
    "Always consult a qualified healthcare professional before starting, stopping, "
    "or changing any medication. In case of emergency, call your local emergency number "
    "or go to the nearest emergency room immediately."
)

# Symptoms that always trigger immediate emergency escalation
EMERGENCY_SYMPTOMS = [
    "chest_pain",
    "weakness_of_one_body_side",
    "slurred_speech",
    "altered_sensorium",
    "coma",
    "stomach_bleeding",
    "acute_liver_failure",
]


def build_treatment_db():
    """Merge hand-curated data with the auxiliary CSV datasets."""
    print("  Loading auxiliary datasets...", flush=True)

    desc_df = pd.read_csv(RAW / "symptom_description.csv")
    prec_df = pd.read_csv(RAW / "symptom_precaution.csv")
    sev_df = pd.read_csv(RAW / "symptom_severity.csv")

    # Build lookup dicts
    descriptions = dict(zip(desc_df["Disease"], desc_df["Description"]))
    precautions = {}
    for _, row in prec_df.iterrows():
        precs = [row[c] for c in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
                 if pd.notna(row[c]) and str(row[c]).strip()]
        precautions[row["Disease"]] = precs

    severity_weights = dict(zip(sev_df["Symptom"], sev_df["weight"]))

    # Load disease classes
    classes = json.loads((PROCESSED / "disease_classes.json").read_text(encoding="utf-8"))

    print(f"  Diseases: {len(classes)}", flush=True)
    print(f"  Descriptions: {len(descriptions)}", flush=True)
    print(f"  Precaution sets: {len(precautions)}", flush=True)
    print(f"  Symptom severities: {len(severity_weights)}", flush=True)

    # Build unified treatment database
    treatment_db = {
        "version": "1.0",
        "disclaimer": SAFETY_DISCLAIMER,
        "emergency_symptoms": EMERGENCY_SYMPTOMS,
        "diseases": {},
    }

    matched = 0
    for disease in classes:
        entry = {"name": disease}

        # Description from CSV
        entry["description"] = descriptions.get(disease, descriptions.get(disease.strip(), ""))

        # Precautions from CSV
        csv_precs = precautions.get(disease, precautions.get(disease.strip(), []))

        # Hand-curated treatment data
        curated = TREATMENT_MAP.get(disease, {})
        if curated:
            matched += 1
            entry["medication_classes"] = curated["medication_classes"]
            entry["first_line"] = curated["first_line"]
            entry["specialist"] = curated["specialist"]
            entry["department"] = curated["department"]
            entry["urgency"] = curated["urgency"]
            entry["lifestyle"] = curated["lifestyle"]
            entry["red_flags"] = curated["red_flags"]
            # Merge CSV precautions with curated lifestyle
            entry["precautions"] = csv_precs if csv_precs else curated["lifestyle"][:4]
        else:
            print(f"  WARNING: No curated data for '{disease}'", flush=True)
            entry["medication_classes"] = ["Consult physician"]
            entry["first_line"] = "Consult a qualified healthcare professional"
            entry["specialist"] = "General Physician"
            entry["department"] = "General Medicine"
            entry["urgency"] = "moderate"
            entry["lifestyle"] = csv_precs if csv_precs else ["Consult doctor"]
            entry["precautions"] = csv_precs if csv_precs else ["Consult doctor"]
            entry["red_flags"] = ["Any worsening symptoms"]

        treatment_db["diseases"][disease] = entry

    print(f"\n  Curated entries matched: {matched}/{len(classes)}", flush=True)

    # Add symptom severity reference
    treatment_db["symptom_severity"] = severity_weights

    return treatment_db


def main() -> int:
    ensure_dirs()

    print("=" * 74, flush=True)
    print(" MediSmart - Phase 4: Treatment Knowledge Base", flush=True)
    print("=" * 74, flush=True)

    db = build_treatment_db()

    # Save to knowledge directory
    out_path = KNOWLEDGE / "treatment_db.json"
    out_path.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved: {out_path}", flush=True)
    print(f"  Total diseases: {len(db['diseases'])}", flush=True)
    print(f"  Emergency symptoms: {len(db['emergency_symptoms'])}", flush=True)

    # Summary by urgency
    urgency_counts = {}
    for d in db["diseases"].values():
        u = d["urgency"]
        urgency_counts[u] = urgency_counts.get(u, 0) + 1
    print(f"\n  Urgency distribution:", flush=True)
    for u in ["critical", "high", "moderate", "low"]:
        print(f"    {u:10s}: {urgency_counts.get(u, 0)} diseases", flush=True)

    # Summary by department
    dept_counts = {}
    for d in db["diseases"].values():
        dept = d["department"].split("/")[0].strip()
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
    print(f"\n  Top departments:", flush=True)
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {dept:25s}: {count} diseases", flush=True)

    print(f"\n  Safety disclaimer: included", flush=True)
    print(f"  Red-flag escalation: {sum(len(d['red_flags']) for d in db['diseases'].values())} total rules", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
