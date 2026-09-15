"""
Phase 1: In this phase we clean the dataset like remove 
the duplicate values,drop unnecessary columns,food columns
and filter the invalid rows then finaly fix the amenity columns
"""
import pandas as pd
from src import config

def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"[load_raw_data] loaded shape: {df.shape}")
    return df

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df =df.drop_duplicates(subset=config.DEDUP_SUBSET , keep='first')
    after = df.shape[0]
    print(f"[deduplicate] removed. {before-after} rows. Shape{df.shape}")
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = config.DROP_COLS
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)

    print(f"[drop_unnecessary_columns] Dropped: {existing} Shape:{df.shape}")
    return df

def drop_food_columns(df: pd.DataFrame) ->pd.DataFrame:
    food_cols = config.FOOD_COLS
    existing = [c for c in food_cols if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[drop_food_columns] Dropped:{existing} Shape:{df.shape}")
    return df

def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=config.ROW_FILTER_SUBSET)
    df = df[df["rent"] >= config.MIN_RENT]
    print(f"[filter_invalid_rows] Shape after filtering: {df.shape}")
    return df

def fix_amenity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing amenity values with False (amenity not mentioned = not available), cast to bool."""
    for col in config.AMENITY_COLS:
        df[col] = df[col].fillna(False).infer_objects(copy=False)
    df[config.AMENITY_COLS] = df[config.AMENITY_COLS].astype(bool)
    print(f"[fix_amenity_columns] Shape: {df.shape}")
    return df
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full Phase 1 cleaning pipeline in order.
    This is the single function train.py will call.
    """
    df = deduplicate(df)
    df = drop_unnecessary_columns(df)
    df = drop_food_columns(df)
    df = filter_invalid_rows(df)
    df = fix_amenity_columns(df)

    print("\n[clean_data] Final shape:", df.shape)
    missing = df.isna().sum()
    print("[clean_data] Remaining missing values:\n", missing[missing > 0])
    print("[clean_data] Duplicate rows:", df.duplicated().sum())

    return df

