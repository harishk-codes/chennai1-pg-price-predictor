"""
Inference script: load the saved model + preprocessing artifacts and
predict rent for a brand-new PG listing.

Usage (as a library):
    from src.predict import predict_rent
    rent = predict_rent({
        "occupancy": "DOUBLE",
        "gender": "Male",
        "available_for": "Students",
        "parking": "Two Wheeler",
        "locality": "Velachery",
        "latitude": 12.9756,
        "longitude": 80.2207,
        "transit_score": 7.5,
        "lifestyle_score": 6.0,
        "deposit": 10000,
        "attached_bathroom": True,
        "mess": True,
        "wifi": True,
        "laundry": False,
        "power_backup": True,
        "refrigerator": False,
        "common_tv": True,
        "room_cleaning": True,
        "room_ac": False,
        "room_cupboard": True,
        "room_tv": False,
        "room_geyser": True,
        "room_bedding": True,
        "room_attached_bath": True,
    })

Usage (from the command line, for a quick manual test):
    python -m src.predict
"""

import os
import joblib
import pandas as pd

from src import config
from src.preprocessing import transform_data

_MODEL = None
_ARTIFACTS = None


def _load_once():
    """Load the model + artifacts from disk exactly once, then cache in memory."""
    global _MODEL, _ARTIFACTS
    if _MODEL is None or _ARTIFACTS is None:
        model_path = os.path.join(config.MODELS_DIR, "gb_model.pkl")
        artifacts_path = os.path.join(config.MODELS_DIR, "preprocessing_artifacts.pkl")

        if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
            raise FileNotFoundError(
                "Trained model/artifacts not found in models/. "
                "Run `python -m src.train` first."
            )

        _MODEL = joblib.load(model_path)
        _ARTIFACTS = joblib.load(artifacts_path)
        print("[predict] Model and preprocessing artifacts loaded.")


def predict_rent(listing: dict) -> float:
    """
    Predict rent for a single new PG listing.

    `listing` must contain the same raw fields the model was trained on
    (occupancy, gender, available_for, parking, locality, latitude,
    longitude, transit_score, lifestyle_score, deposit, and the amenity
    boolean columns). Missing amenity fields default to False.
    """
    _load_once()

    row = dict(listing)  # don't mutate the caller's dict

    # default any missing amenity flags to False, same convention as training
    for col in config.AMENITY_COLS:
        row.setdefault(col, False)

    X_new = pd.DataFrame([row])
    X_transformed = transform_data(X_new, _ARTIFACTS)

    prediction = _MODEL.predict(X_transformed)[0]
    return float(prediction)


if __name__ == "__main__":
    # quick manual smoke test
    sample_listing = {
        "occupancy": "DOUBLE",
        "gender": "Male",
        "available_for": "Students",
        "parking": "Two Wheeler",
        "locality": "Velachery",
        "latitude": 12.9756,
        "longitude": 80.2207,
        "transit_score": 7.5,
        "lifestyle_score": 6.0,
        "deposit": 10000,
        "attached_bathroom": True,
        "mess": True,
        "wifi": True,
    }
    predicted = predict_rent(sample_listing)
    print(f"Predicted rent: Rs {predicted:.2f}")