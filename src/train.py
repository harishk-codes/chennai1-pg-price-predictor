"""
Entry point for end-to-end model training.
Run with:  python -m src.train   (from the project root)

Order:
  1. load raw data
  2. clean data (Phase 1)
  3. split into train / val / test  (BEFORE any statistic-based preprocessing -> avoids leakage)
  4. fit preprocessing on train only
  5. transform train / val / test using the SAME fitted artifacts
  6. train Linear Regression, Random Forest, Gradient Boosting
  7. hyperparameter tuning for Gradient Boosting (comparison only)
  8. build a results table dynamically from computed metrics
  9. final evaluation on the TEST set using the untuned Gradient Boosting model
  10. save the model + every preprocessing artifact to models/
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.data_cleaning import load_raw_data, clean_data
from src.preprocessing import fit_preprocessing, transform_data


def evaluate(model, X, y):
    """Return (mae, rmse, r2) for a fitted model on a given X, y."""
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return mae, rmse, r2


def main():
    # ------------------------------------------------------------------
    # 1-2. Load + clean
    # ------------------------------------------------------------------
    df = load_raw_data()
    df = clean_data(df)

    X = df.drop(columns=["rent"])
    y = df["rent"]

    # ------------------------------------------------------------------
    # 3. Split FIRST (before any statistic-based preprocessing)
    # ------------------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=config.TEST_SIZE_FROM_FULL,
        random_state=config.RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.VAL_TEST_SPLIT,
        random_state=config.RANDOM_STATE,
    )
    print(f"[split] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 4-5. Fit preprocessing on train only, transform everything
    # ------------------------------------------------------------------
    artifacts = fit_preprocessing(X_train, y_train)

    X_train_t = transform_data(X_train, artifacts)
    X_val_t = transform_data(X_val, artifacts)
    X_test_t = transform_data(X_test, artifacts)

    # ------------------------------------------------------------------
    # 6. Train models
    # ------------------------------------------------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train_t, y_train)
    mae, rmse, r2 = evaluate(lr_model, X_val_t, y_val)

    rf_model = RandomForestRegressor(**config.RF_PARAMS)
    rf_model.fit(X_train_t, y_train)
    rf_mae, rf_rmse, rf_r2 = evaluate(rf_model, X_val_t, y_val)

    gb_model = GradientBoostingRegressor(**config.GB_PARAMS)
    gb_model.fit(X_train_t, y_train)
    gb_mae, gb_rmse, gb_r2 = evaluate(gb_model, X_val_t, y_val)

    # ------------------------------------------------------------------
    # 7. Hyperparameter tuning (comparison only - NOT used as final model
    #    unless it proves better on the untouched test set)
    # ------------------------------------------------------------------
    gb_tuning = GridSearchCV(
        GradientBoostingRegressor(random_state=config.RANDOM_STATE),
        param_grid=config.GB_PARAM_GRID,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gb_tuning.fit(X_train_t, y_train)
    tuned_mae, tuned_rmse, tuned_r2 = evaluate(gb_tuning.best_estimator_, X_val_t, y_val)
    print("[tuning] Best params:", gb_tuning.best_params_)

    # ------------------------------------------------------------------
    # 8. Results table (built dynamically, never hardcoded)
    # ------------------------------------------------------------------
    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "Tuned Gradient Boosting"],
        "MAE": [mae, rf_mae, gb_mae, tuned_mae],
        "RMSE": [rmse, rf_rmse, gb_rmse, tuned_rmse],
        "R2": [r2, rf_r2, gb_r2, tuned_r2],
    })
    print("\n[results] Validation set comparison:\n", results)

    # ------------------------------------------------------------------
    # 9. Final evaluation on TEST set
    #    Decision made earlier in this project: untuned gb_model generalizes
    #    at least as well as the tuned model on unseen data, so it is final.
    # ------------------------------------------------------------------
    test_mae, test_rmse, test_r2 = evaluate(gb_model, X_test_t, y_test)
    print("\nFINAL GRADIENT BOOSTING - TEST RESULTS")
    print("--------------------------------------")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)
    print("R2  :", test_r2)

    # ------------------------------------------------------------------
    # 10. Save model + every preprocessing artifact
    # ------------------------------------------------------------------
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    joblib.dump(gb_model, os.path.join(config.MODELS_DIR, "gb_model.pkl"))
    joblib.dump(artifacts, os.path.join(config.MODELS_DIR, "preprocessing_artifacts.pkl"))

    print(f"\n[save] Model and artifacts saved to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()