# Chennai PG Rent Price Predictor

ML regression model to predict PG (paying guest) rental prices in Chennai based on location, amenities, and property features.

## Dataset

- **Source**: `Data/raw/chennai_pg_dataset.csv` (not included in repo — see `.gitignore`)
- **Size**: 1661 raw rows → 1436 after cleaning (39 → 26 columns after dropping unnecessary fields)

## Pipeline

1. **Cleaning** (`Notebook/06-EDA-Preprocessing.ipynb`, Phase 1)
   - Deduplication on `id + occupancy`
   - Dropped irrelevant/redundant columns (id, title, address, food breakdown, etc.)
   - Filtered invalid rows (missing rent/deposit/occupancy, rent < ₹1000)
   - Filled amenity columns (boolean, missing → False)

2. **Preprocessing** (Phase 2)
   - Train/Val/Test split: 80/10/10 — done **before** any statistic-based imputation to avoid data leakage
   - `occupancy` → Ordinal encoding (natural order: SINGLE < DOUBLE < THREE < FOUR)
   - `gender`, `available_for`, `parking` → One-hot encoding
   - `locality` → Smoothed target encoding (alpha=10)
   - `transit_score`, `lifestyle_score` → locality-median imputation (train-only stats)
   - Numeric features → StandardScaler

3. **Modeling**
   - Linear Regression (baseline)
   - Random Forest Regressor
   - Gradient Boosting Regressor (final model)
   - Hyperparameter tuning via GridSearchCV (didn't outperform manual params on test set — kept for comparison)

## Final Results (Gradient Boosting, test set)

| Metric | Value |
|--------|-------|
| MAE    | 1265.07 |
| RMSE   | 2233.35 |
| R²     | 0.4465 |

## Saved Artifacts (`models/`)

- `gb_model.pkl` — final trained model
- `occupancy_encoder.pkl`, `onehot_encoder.pkl`, `scaler.pkl` — preprocessing objects
- `locality_mapping.pkl`, `transit_locality_median.pkl`, `lifestyle_locality_median.pkl` — encoding/imputation maps
- `feature_columns.pkl` — expected column order for inference

## How to run

```bash
pip install -r requirements.txt
```

Open `Notebook/06-EDA-Preprocessing.ipynb` and run cells top to bottom. Raw data expected at `Data/raw/chennai_pg_dataset.csv` relative to project root.

## Known limitations

- R² (~0.45) indicates moderate predictive power — further feature engineering or a larger dataset could improve this
- Small dataset (1436 rows) makes hyperparameter tuning results less stable across CV folds