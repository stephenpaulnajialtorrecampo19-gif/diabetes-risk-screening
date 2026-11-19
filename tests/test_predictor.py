import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import predictor

FEATURES = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
            'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']


def good_input():
    return {
        "age": 40, "sex": "Female", "height_cm": 165, "weight_kg": 65,
        "high_bp": 0, "high_chol": 0, "smoker": 0, "stroke": 0, "heart_disease": 0,
        "phys_activity": 1, "fruits": 1, "veggies": 1, "heavy_alcohol": 0, "diff_walking": 0,
        "gen_health": 3, "ment_health_days": 0, "phys_health_days": 0,
    }


def small_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(0, 2, size=(200, len(FEATURES))), columns=FEATURES)
    y = (X["HighBP"] + X["HighChol"] >= 1).astype(int)
    return LogisticRegression(max_iter=200).fit(X, y)


def test_bmi():
    assert predictor.calculate_bmi(170, 70) == 24.2


def test_age_groups():
    assert predictor.age_to_group(18) == 1
    assert predictor.age_to_group(24) == 1
    assert predictor.age_to_group(25) == 2
    assert predictor.age_to_group(79) == 12
    assert predictor.age_to_group(80) == 13
    assert predictor.age_to_group(100) == 13


def test_valid_input_has_no_errors():
    assert predictor.validate_inputs(good_input()) == []


def test_missing_value():
    data = good_input()
    data["age"] = None
    assert len(predictor.validate_inputs(data)) == 1


def test_out_of_range_values():
    data = good_input()
    data["age"] = 10
    data["height_cm"] = 20
    data["ment_health_days"] = 45
    assert len(predictor.validate_inputs(data)) == 3


def test_unrealistic_bmi():
    data = good_input()
    data["height_cm"] = 250
    data["weight_kg"] = 25
    assert len(predictor.validate_inputs(data)) == 1


def test_text_instead_of_number():
    data = good_input()
    data["weight_kg"] = "abc"
    assert len(predictor.validate_inputs(data)) == 1


def test_input_row_order_and_values():
    row = predictor.build_input_row(good_input(), FEATURES)
    assert list(row.columns) == FEATURES
    assert row["Sex"].iloc[0] == 0
    assert row["Age"].iloc[0] == 5
    assert row["BMI"].iloc[0] == 23.9


def test_prediction_output():
    info = {"features": FEATURES, "threshold": 0.5}
    result = predictor.predict_risk(small_model(), good_input(), info)
    assert 0 <= result["score"] <= 1
    assert result["label"] in ("Higher risk", "Lower risk")


def test_higher_risk_when_score_above_threshold():
    info = {"features": FEATURES, "threshold": 0.0}
    result = predictor.predict_risk(small_model(), good_input(), info)
    assert result["higher_risk"] is True


def test_tips_always_include_blood_test_advice():
    tips = predictor.health_tips(good_input())
    assert any("blood sugar test" in tip for tip in tips)
