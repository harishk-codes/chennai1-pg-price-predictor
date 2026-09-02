 """
Phase 2: Preprocessing
fit_preprocessing() is called ONCE on training data - fits all encoders/scaler/mappings.
transform_data() is called on train/val/test/new-data - applies already-fitted objects only.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from src import config


def _clean_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Row-level fixes that don't need any fitted statistics (safe to repeat anywhere)."""
    X = X.copy()
    X['parking'] = X['parking'].fillna('none')
    X['available_for'] = X['available_for'].replace('Both', 'Anyone')
    return X


def fit_preprocessing(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Fit every preprocessing object using ONLY X_train / y_train.
    Returns a dict of fitted artifacts reused by transform_data() and predict.py.
    """
    artifacts = {}
    X_train = X_train.copy()
    X_train['transit_score'] = X_train['transit_score'].replace(-10, np.nan)

    # ---- imputation maps (train-only statistics) ----
    artifacts['transit_locality_median'] = (
        X_train.groupby('locality')['transit_score'].median().to_dict()
    )
    artifacts['overall_transit_median'] = X_train['transit_score'].median()

    artifacts['lifestyle_locality_median'] = (
        X_train.groupby('locality')['lifestyle_score'].median().to_dict()
    )
    artifacts['overall_lifestyle_median'] = X_train['lifestyle_score'].median()

    # ---- occupancy ordinal encoder ----
    occupancy_encoder = OrdinalEncoder(categories=[config.OCCUPANCY_ORDER])
    occupancy_encoder.fit(X_train[['occupancy']])
    artifacts['occupancy_encoder'] = occupancy_encoder

    # ---- one-hot encoder (gender, available_for, parking) ----
    X_train_clean = _clean_categoricals(X_train)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train_clean[config.ONEHOT_COLS])
    artifacts['ohe'] = ohe

    # ---- smoothed target encoding for locality ----
    locality_stats = X_train[['locality']].copy()
    locality_stats['rent'] = y_train.values
    locality_stats = locality_stats.groupby('locality')['rent'].agg(['mean', 'count'])

    global_mean = y_train.mean()
    alpha = config.TARGET_ENCODE_ALPHA
    locality_stats['smoothed_mean'] = (
        (locality_stats['count'] * locality_stats['mean'] + alpha * global_mean)
        / (locality_stats['count'] + alpha)
    )
    artifacts['locality_mapping'] = locality_stats['smoothed_mean'].to_dict()
    artifacts['global_mean'] = global_mean

    # ---- scaler: fit on the fully-encoded (but not-yet-scaled) train set ----
    X_train_encoded = _apply_imputation_and_encoding(X_train, artifacts)
    scaler = StandardScaler()
    scaler.fit(X_train_encoded[config.SCALE_COLS])
    artifacts['scaler'] = scaler

    # ---- lock in the final column order for consistency at inference time ----
    X_train_final = _apply_scaling(X_train_encoded, artifacts)
    artifacts['feature_columns'] = list(X_train_final.columns)

    print("[fit_preprocessing] Fitted all encoders/scaler/mappings on train data.")
    print("[fit_preprocessing] Final feature count:", len(artifacts['feature_columns']))

    return artifacts


def _apply_imputation_and_encoding(X: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Everything EXCEPT scaling: imputation, ordinal encode, one-hot encode, target encode."""
    X = X.copy()

    # transit_score
    X['transit_score'] = X['transit_score'].replace(-10, np.nan)
    X['transit_score_missing'] = X['transit_score'].isna()
    X['transit_score'] = X['transit_score'].fillna(
        X['locality'].map(artifacts['transit_locality_median'])
    )
    X['transit_score'] = X['transit_score'].fillna(artifacts['overall_transit_median'])

    # lifestyle_score
    X['lifestyle_score_missing'] = X['lifestyle_score'].isna()
    X['lifestyle_score'] = X['lifestyle_score'].fillna(
        X['locality'].map(artifacts['lifestyle_locality_median'])
    )
    X['lifestyle_score'] = X['lifestyle_score'].fillna(artifacts['overall_lifestyle_median'])

    # parking / available_for cleanup
    X = _clean_categoricals(X)

    # occupancy -> ordinal
    X['occupancy_encoded'] = artifacts['occupancy_encoder'].transform(X[['occupancy']])
    X = X.drop(columns=['occupancy'])

    # gender / available_for / parking -> one-hot
    encoded = artifacts['ohe'].transform(X[config.ONEHOT_COLS])
    encoded_cols = artifacts['ohe'].get_feature_names_out(config.ONEHOT_COLS)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
    X = X.drop(columns=config.ONEHOT_COLS)
    X = pd.concat([X, encoded_df], axis=1)

    # locality -> smoothed target encoding
    X['locality_encoded'] = X['locality'].map(artifacts['locality_mapping'])
    X['locality_encoded'] = X['locality_encoded'].fillna(artifacts['global_mean'])
    X = X.drop(columns=['locality'])

    return X


def _apply_scaling(X: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    X = X.copy()
    X[config.SCALE_COLS] = artifacts['scaler'].transform(X[config.SCALE_COLS])
    return X


def transform_data(X: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Apply ALL fitted preprocessing to X: imputation -> encoding -> scaling.
    Use for train/val/test AND for brand-new incoming data at prediction time.
    """
    X_transformed = _apply_imputation_and_encoding(X, artifacts)
    X_transformed = _apply_scaling(X_transformed, artifacts)

    # keep column order identical to what the model was trained on
    if 'feature_columns' in artifacts:
        X_transformed = X_transformed.reindex(
            columns=artifacts['feature_columns'], fill_value=0
        )

    return X_transformed