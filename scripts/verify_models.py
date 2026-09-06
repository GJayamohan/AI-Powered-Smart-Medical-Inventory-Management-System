"""Quick verification: test all trained models with example inputs."""

import os, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

MODELS = "artifacts/models"


def test_diabetes():
    print("=" * 60)
    print("  DIABETES MODEL - Project Brief Worked Example")
    print("=" * 60)

    model = tf.keras.models.load_model(f"{MODELS}/diabetes_model.keras")
    preprocessor = joblib.load(f"{MODELS}/diabetes_preprocessor.pkl")
    features = json.loads(open(f"{MODELS}/diabetes_features.json").read())

    # Age=45, Glucose=180, BMI=29, BP=140/90
    patient = {
        "Pregnancies": 2, "Glucose": 180.0, "BloodPressure": 140.0,
        "SkinThickness": float("nan"), "Insulin": float("nan"),
        "BMI": 29.0, "DiabetesPedigreeFunction": 0.5, "Age": 45,
    }
    print(f"\n  Input: Age=45, Glucose=180, BMI=29, BP=140/90")
    df = pd.DataFrame([patient])
    X = preprocessor.transform(df)
    prob = float(model.predict(X, verbose=0)[0][0])

    label = "Diabetes" if prob >= 0.5 else "No Diabetes"
    conf = max(prob, 1 - prob) * 100
    print(f"  Predicted condition: {label}")
    print(f"  Model confidence:   {conf:.1f}%")
    print(f"  Raw probability:    {prob*100:.1f}%")
    return prob >= 0.5


def test_symptom():
    print("\n" + "=" * 60)
    print("  SYMPTOM MODEL - Diabetes Symptom Test")
    print("=" * 60)

    model = tf.keras.models.load_model(f"{MODELS}/symptom_model.keras")
    encoder = joblib.load(f"{MODELS}/symptom_label_encoder.pkl")
    features = json.loads(open(f"{MODELS}/symptom_features.json").read())

    symptoms = [
        "fatigue", "weight_loss", "restlessness", "lethargy",
        "irregular_sugar_level", "blurred_and_distorted_vision",
        "excessive_hunger", "polyuria",
    ]
    print(f"\n  Input symptoms: {', '.join(symptoms)}")
    x = np.zeros((1, len(features)), dtype="float32")
    for s in symptoms:
        if s in features:
            x[0, features.index(s)] = 1

    proba = model.predict(x, verbose=0)[0]
    top3 = np.argsort(proba)[-3:][::-1]
    print(f"\n  Top-3 predictions:")
    for i, idx in enumerate(top3):
        print(f"    {i+1}. {encoder.classes_[idx]:35s} {proba[idx]*100:.1f}%")


def test_heart():
    print("\n" + "=" * 60)
    print("  HEART MODEL - Quick Sanity Check")
    print("=" * 60)

    model = tf.keras.models.load_model(f"{MODELS}/heart_model.keras")
    preprocessor = joblib.load(f"{MODELS}/heart_preprocessor.pkl")
    features = json.loads(open(f"{MODELS}/heart_features.json").read())

    # Typical high-risk patient
    patient = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    print(f"\n  Input: age=63, male, chest pain type 3, BP=145, chol=233")
    df = pd.DataFrame([patient])
    X = preprocessor.transform(df)
    prob = float(model.predict(X, verbose=0)[0][0])
    print(f"  Heart disease probability: {prob*100:.1f}%")


def test_expiry():
    print("\n" + "=" * 60)
    print("  EXPIRY MODEL - Batch Risk Assessment")
    print("=" * 60)

    model = tf.keras.models.load_model(f"{MODELS}/expiry_model.keras")
    preprocessor = joblib.load(f"{MODELS}/expiry_preprocessor.pkl")
    features = json.loads(open(f"{MODELS}/expiry_features.json").read())
    threshold_config = json.loads(open(f"{MODELS}/expiry_threshold.json").read())

    # High-risk batch: slow mover, lots remaining, close to expiry
    batch = {
        "days_to_expiry": 30, "qty_remaining": 150, "pct_batch_remaining": 0.75,
        "total_stock_all_batches": 200, "usage_rate_7d": 0.5, "usage_rate_30d": 0.6,
        "usage_rate_90d": 0.7, "usage_trend": 0.86, "days_of_cover": 300,
        "cover_ratio": 0.1, "shelf_life_days": 365, "pct_life_remaining": 0.08,
        "unit_cost": 85.0, "month": 6, "prescription_required": 0,
    }
    print(f"\n  Input: 150 units remaining, 30 days to expiry, usage=0.5/day")
    df = pd.DataFrame([batch])
    X = preprocessor.transform(df[features])
    clf_prob, reg_pred = model.predict(X, verbose=0)
    prob = float(clf_prob[0][0])
    var = float(reg_pred[0][0])

    threshold = threshold_config["threshold"]
    risk_score = int(prob * 100)
    if risk_score <= 25:
        bucket = "SAFE"
    elif risk_score <= 50:
        bucket = "WATCH"
    elif risk_score <= 75:
        bucket = "HIGH"
    else:
        bucket = "CRITICAL"

    print(f"  Expiry probability:  {prob*100:.1f}%")
    print(f"  Risk score:          {risk_score}/100")
    print(f"  Risk bucket:         {bucket}")
    print(f"  Value at risk:       Rs {max(0, var):.2f}")


if __name__ == "__main__":
    ok = test_diabetes()
    test_symptom()
    test_heart()
    test_expiry()
    print("\n" + "=" * 60)
    print("  ALL MODEL VERIFICATION COMPLETE")
    print("=" * 60)
