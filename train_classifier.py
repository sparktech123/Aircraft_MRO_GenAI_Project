"""
sdr_common.py
=============
Small shared module used by both 01_train_model.py and 02_predict_2026.py.

Keeping this helper in its own importable module (rather than inline in
01_train_model.py) matters because joblib/pickle stores functions by their
module path. If `_to_dense` lived inside 01_train_model.py's __main__ block,
the saved pipeline could only ever be unpickled by re-running that exact
script — loading it from 02_predict_2026.py (or anywhere else) would fail
with "Can't get attribute '_to_dense' on <module '__main__'>". Importing it
from here instead makes the saved model portable.
"""


def to_dense(x):
    """sparse -> dense. Only used by the HistGradientBoosting fallback
    pipeline (activated automatically when xgboost isn't installed)."""
    return x.toarray()


def engineer_features(df, cat_cols, text_col, date_col="DifficultyDate"):
    """Builds the exact same feature set used at training time.

    Shared by 01_train_model.py and 02_predict_2026.py so the two scripts
    can never drift apart on feature engineering. Returns a new DataFrame;
    does not mutate the input in place.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    hour = df[date_col].dt.hour.fillna(0).astype(int)
    day = df[date_col].dt.day.fillna(1).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * day / 31)
    df["day_cos"] = np.cos(2 * np.pi * day / 31)

    df[cat_cols] = df[cat_cols].fillna("UNKNOWN")
    df[text_col] = df[text_col].fillna("").astype(str)
    return df
