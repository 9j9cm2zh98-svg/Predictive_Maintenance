"""
preprocess.py
-------------
Loads the raw UCI AI4I 2020 dataset, cleans it, engineers a small number
of meaningful features, and returns train/test splits ready for model
training.

Run standalone:
    python -m src.preprocess
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH   = os.path.join(BASE_DIR, "data", "raw", "ai4i2020.csv")
PROC_PATH  = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# Columns the model will actually use (after engineering)
FEATURE_COLS = [
    "Type",               # Machine type: L / M / H → label-encoded
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",          # Engineered: process_temp − air_temp
    "power",              # Engineered: torque × rotational_speed
]
TARGET_COL = "Machine failure"

# Mapping for the categorical 'Type' column
TYPE_MAP = {"L": 0, "M": 1, "H": 2}


# ──────────────────────────────────────────────────────────────────────────
def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at:\n  {path}\n\n"
            "Please download the UCI AI4I 2020 Predictive Maintenance dataset\n"
            "from https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
            "and place the CSV file at:  data/raw/ai4i2020.csv"
        )
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier columns that carry no predictive signal:
      - 'UDI'         : row index
      - 'Product ID'  : product serial number
    We also drop individual failure-mode columns (TWF, HDF, PWF, OSF, RNF)
    because they are sub-components of 'Machine failure' — keeping them
    would leak the target and trivialise the classification task.
    """
    cols_to_drop = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drop)

    # Drop duplicates, reset index
    df = df.drop_duplicates().reset_index(drop=True)

    # The dataset has no missing values, but handle them defensively
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two physically meaningful features:

    temp_diff : difference between process and air temperature.
               A larger gap can indicate thermal stress on bearings.

    power     : torque × rotational speed — proportional to mechanical
               power delivered to the tool.  High power combined with
               high tool wear is a known failure precursor.
    """
    df = df.copy()
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power"]     = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    return df


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the 'Type' column using TYPE_MAP."""
    df = df.copy()
    df["Type"] = df["Type"].map(TYPE_MAP)
    return df


def get_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the processed DataFrame into train/test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test  (all pandas DataFrames / Series)
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # preserve class ratio in both splits
    )


def run_pipeline(save_processed: bool = True):
    """
    Full preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test.
    """
    print("[preprocess] Loading raw data …")
    df = load_raw()

    print(f"[preprocess] Raw shape: {df.shape}")
    df = clean(df)
    print(f"[preprocess] After cleaning: {df.shape}")

    df = engineer_features(df)
    df = encode_type(df)

    if save_processed:
        os.makedirs(os.path.dirname(PROC_PATH), exist_ok=True)
        df.to_csv(PROC_PATH, index=False)
        print(f"[preprocess] Processed data saved → {PROC_PATH}")

    X_train, X_test, y_train, y_test = get_splits(df)
    print(
        f"[preprocess] Train: {X_train.shape}  |  Test: {X_test.shape}\n"
        f"[preprocess] Failure rate (train): "
        f"{y_train.mean()*100:.2f}%"
    )
    return X_train, X_test, y_train, y_test


# ── Standalone run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
