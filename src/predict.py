"""
predict.py
----------
Loads the saved Random Forest model and makes predictions on new sensor
readings.  Used by both the CLI and the Streamlit app.
"""

import os
import joblib
import pandas as pd

from src.preprocess import (
    engineer_features,
    encode_type,
    FEATURE_COLS,
)

# ── Path helper ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.joblib")

# Risk thresholds — model-based categories, NOT real industrial safety limits
RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60}


# ──────────────────────────────────────────────────────────────────────────
def load_model(path: str = MODEL_PATH):
    """Load and return the saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Trained model not found at:\n  {path}\n\n"
            "Please run the training script first:\n"
            "    python -m src.train"
        )
    return joblib.load(path)


def build_input_df(
    machine_type: str,
    air_temp: float,
    process_temp: float,
    rot_speed: float,
    torque: float,
    tool_wear: float,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame from raw sensor values,
    applying the same feature engineering used during training.
    """
    raw = pd.DataFrame([{
        "Type":                      machine_type,
        "Air temperature [K]":       air_temp,
        "Process temperature [K]":   process_temp,
        "Rotational speed [rpm]":    rot_speed,
        "Torque [Nm]":               torque,
        "Tool wear [min]":           tool_wear,
    }])

    # Apply the same transformations as in preprocess.py
    raw = engineer_features(raw)
    raw = encode_type(raw)

    return raw[FEATURE_COLS]


def interpret_risk(probability: float) -> str:
    """
    Convert a failure probability to a human-readable risk category.
    These thresholds are model-based categories — not real safety standards.
    """
    if probability < RISK_THRESHOLDS["low"]:
        return "Low Risk"
    elif probability < RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    else:
        return "High Risk"


def predict(
    machine_type: str,
    air_temp: float,
    process_temp: float,
    rot_speed: float,
    torque: float,
    tool_wear: float,
    model=None,
) -> dict:
    """
    Run a single prediction.

    Parameters
    ----------
    machine_type : str   — "L", "M", or "H"
    air_temp     : float — Air temperature in Kelvin
    process_temp : float — Process temperature in Kelvin
    rot_speed    : float — Rotational speed in rpm
    torque       : float — Torque in Nm
    tool_wear    : float — Tool wear in minutes
    model        : optional pre-loaded model (avoids reloading from disk)

    Returns
    -------
    dict with keys: prediction (int), probability (float), risk (str)
    """
    if model is None:
        model = load_model()

    input_df    = build_input_df(machine_type, air_temp, process_temp,
                                 rot_speed, torque, tool_wear)
    prediction  = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    risk        = interpret_risk(probability)

    return {
        "prediction":  prediction,
        "probability": probability,
        "risk":        risk,
    }


# ── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[predict] Running sample prediction …\n")

    # Typical healthy-machine values from the dataset
    result = predict(
        machine_type="M",
        air_temp=298.1,
        process_temp=308.6,
        rot_speed=1551,
        torque=42.8,
        tool_wear=0,
    )

    print(f"  Prediction  : {'FAILURE' if result['prediction'] else 'No Failure'}")
    print(f"  Probability : {result['probability']:.4f} ({result['probability']*100:.1f}%)")
    print(f"  Risk Level  : {result['risk']}")
