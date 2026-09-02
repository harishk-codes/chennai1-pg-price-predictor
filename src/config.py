"""
Central configuration for the Chennai PG Rent Predictor.
All file paths, column lists, and constants live here so that
changing a value doesn't require editing multiple files.
"""

import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# project root = one level above src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "Data", "raw", "chennai_pg_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------
# Cleaning (Phase 1) settings
# ---------------------------------------------------------------------
DEDUP_SUBSET = ["id", "occupancy"]

DROP_COLS = [
    "id", "title", "address", "total_bathrooms",
    "gate_closing_time", "warden", "cooking_allowed",
    "guardian_required", "nonveg_allowed", "smoking_allowed",
]

FOOD_COLS = ["breakfast", "lunch", "dinner"]

ROW_FILTER_SUBSET = ["rent", "deposit", "occupancy", "attached_bathroom"]
MIN_RENT = 1000

AMENITY_COLS = [
    "attached_bathroom", "mess", "wifi", "laundry", "power_backup",
    "refrigerator", "common_tv", "room_cleaning", "room_ac",
    "room_cupboard", "room_tv", "room_geyser", "room_bedding",
    "room_attached_bath",
]

# ---------------------------------------------------------------------
# Preprocessing (Phase 2) settings
# ---------------------------------------------------------------------
OCCUPANCY_ORDER = ["SINGLE", "DOUBLE", "THREE", "FOUR"]

ONEHOT_COLS = ["gender", "available_for", "parking"]

TARGET_ENCODE_COL = "locality"
TARGET_ENCODE_ALPHA = 10

SCALE_COLS = [
    "latitude", "longitude", "transit_score",
    "lifestyle_score", "deposit", "locality_encoded",
]

# ---------------------------------------------------------------------
# Split settings
# ---------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE_FROM_FULL = 0.20      # train vs temp
VAL_TEST_SPLIT = 0.50           # temp -> val / test

# ---------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------
GB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 200,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

GB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
}