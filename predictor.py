import json
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "diabetes_model.joblib")
INFO_PATH = os.path.join(BASE_DIR, "model", "model_info.json")

FIELD_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "high_bp": "High blood pressure",
    "high_chol": "High cholesterol",
    "smoker": "Smoked at least 100 cigarettes",
    "stroke": "Stroke",
    "heart_disease": "Heart disease or heart attack",
    "phys_activity": "Physical activity in the past 30 days",
    "fruits": "Eats fruit daily",
    "veggies": "Eats vegetables daily",
    "heavy_alcohol": "Heavy alcohol drinking",
    "diff_walking": "Difficulty walking or climbing stairs",
    "gen_health": "General health",
    "ment_health_days": "Poor mental health days",
    "phys_health_days": "Poor physical health days",
}

YES_NO_FIELDS = [
    "high_bp", "high_chol", "smoker", "stroke", "heart_disease",
    "phys_activity", "fruits", "veggies", "heavy_alcohol", "diff_walking",
]


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found: " + model_path)
    return joblib.load(model_path)


def load_model_info(info_path=INFO_PATH):
    if not os.path.exists(info_path):
        raise FileNotFoundError("Model info file not found: " + info_path)
    with open(info_path, "r") as f:
        return json.load(f)


def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    return round(weight_kg / (height_m * height_m), 1)


def age_to_group(age):
    if age < 25:
        return 1
    group = (age - 25) // 5 + 2
    if group > 13:
        group = 13
    return int(group)


def to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_inputs(data):
    errors = []

    for key in FIELD_LABELS:
        if key not in data or data[key] is None or data[key] == "":
            errors.append("Please provide a value for: " + FIELD_LABELS[key] + ".")
    if len(errors) > 0:
        return errors

    age = to_number(data["age"])
    if age is None or age != int(age) or age < 18 or age > 120:
        errors.append("Age must be a whole number from 18 to 120.")

    if data["sex"] not in ("Female", "Male"):
        errors.append("Sex must be Female or Male.")

    height = to_number(data["height_cm"])
    if height is None or height < 100 or height > 250:
        errors.append("Height must be between 100 and 250 cm.")

    weight = to_number(data["weight_kg"])
    if weight is None or weight < 25 or weight > 300:
        errors.append("Weight must be between 25 and 300 kg.")

    if height is not None and weight is not None and 100 <= height <= 250 and 25 <= weight <= 300:
        bmi = calculate_bmi(height, weight)
        if bmi < 12 or bmi > 70:
            errors.append("This height and weight give a BMI of " + str(bmi) + ", which is not realistic. Please check them.")

    for key in YES_NO_FIELDS:
        if to_number(data[key]) not in (0.0, 1.0):
            errors.append(FIELD_LABELS[key] + " must be Yes or No.")

    general = to_number(data["gen_health"])
    if general is None or general not in (1, 2, 3, 4, 5):
        errors.append("General health must be a number from 1 (excellent) to 5 (poor).")

    for key in ("ment_health_days", "phys_health_days"):
        days = to_number(data[key])
        if days is None or days != int(days) or days < 0 or days > 30:
            errors.append(FIELD_LABELS[key] + " must be a whole number from 0 to 30.")

    return errors


def build_input_row(data, features):
    values = {
        "HighBP": int(data["high_bp"]),
        "HighChol": int(data["high_chol"]),
        "BMI": calculate_bmi(float(data["height_cm"]), float(data["weight_kg"])),
        "Smoker": int(data["smoker"]),
        "Stroke": int(data["stroke"]),
        "HeartDiseaseorAttack": int(data["heart_disease"]),
        "PhysActivity": int(data["phys_activity"]),
        "Fruits": int(data["fruits"]),
        "Veggies": int(data["veggies"]),
        "HvyAlcoholConsump": int(data["heavy_alcohol"]),
        "GenHlth": int(data["gen_health"]),
        "MentHlth": int(data["ment_health_days"]),
        "PhysHlth": int(data["phys_health_days"]),
        "DiffWalk": int(data["diff_walking"]),
        "Sex": 1 if data["sex"] == "Male" else 0,
        "Age": age_to_group(int(data["age"])),
    }
    return pd.DataFrame([values], columns=features)


def predict_risk(model, data, info):
    row = build_input_row(data, info["features"])
    score = float(model.predict_proba(row)[0][1])
    threshold = info.get("threshold", 0.5)
    higher = score >= threshold
    return {
        "score": score,
        "percent": round(score * 100, 1),
        "higher_risk": higher,
        "label": "Higher risk" if higher else "Lower risk",
        "bmi": float(row["BMI"].iloc[0]),
    }


def health_tips(data):
    tips = []
    bmi = calculate_bmi(float(data["height_cm"]), float(data["weight_kg"]))
    if bmi >= 25:
        tips.append("Your BMI is " + str(bmi) + ". A healthy weight lowers the chance of type 2 diabetes.")
    if int(data["phys_activity"]) == 0:
        tips.append("Try to be physically active, for example brisk walking for 30 minutes on most days.")
    if int(data["high_bp"]) == 1:
        tips.append("Keep your blood pressure monitored and follow your doctor's advice.")
    if int(data["high_chol"]) == 1:
        tips.append("Have your cholesterol checked regularly and ask a doctor about diet changes.")
    if int(data["smoker"]) == 1:
        tips.append("Quitting smoking improves overall health and lowers many risks.")
    if int(data["fruits"]) == 0 or int(data["veggies"]) == 0:
        tips.append("Eating fruits and vegetables every day supports a healthy diet.")
    tips.append("A blood sugar test (such as fasting glucose or HbA1c) is the only way to know if you have diabetes.")
    return tips


def get_score_band(score, info):
    bands = info.get("score_bands")
    if not bands:
        return None
    for band in bands:
        if band["low"] <= score < band["high"]:
            return band
    return None


def threshold_summary(info):
    rows = info.get("threshold_analysis")
    if not rows:
        return None
    for row in rows:
        if abs(row["Threshold"] - info.get("threshold", 0.5)) < 0.0001:
            return row
    return None
