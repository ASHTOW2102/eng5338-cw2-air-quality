"""Shared protocol for ENG5338 CW2 (owner: M1).

Every member imports this so all models use the SAME data, features, split,
CV folds, metrics and search harness. It mirrors Part A of
notebooks/ENG5338_CW2_notebook.ipynb. Do not change it without telling M1.

Usage in a notebook inside notebooks/:
    import sys; sys.path.insert(0, "../src")
    from cw2_common import *
    gs, row = run_search("Random Forest", pipe, grid)
"""
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

_REPO = Path(__file__).resolve().parent.parent
DATA_PATH = _REPO / "data" / "air-quality.csv"

SENSOR_COLS = ["pt08_s1_co", "pt08_s2_nmhc", "pt08_s3_nox", "pt08_s4_no2", "pt08_s5_o3"]
WEATHER_COLS = ["temperature_c", "relative_humidity_pct", "absolute_humidity"]
TIME_COLS = ["hour_sin", "hour_cos", "weekday", "is_weekend"]
FEATURES = SENSOR_COLS + WEATHER_COLS + TIME_COLS
TARGET = "co_gt"

# Constraints (C1 accuracy, C2 latency)
TOLERANCE = 0.5          # mg/m3
TARGET_PCT = 90          # % of predictions within TOLERANCE
LATENCY_LIMIT_MS = 1.0   # ms per sample


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df = df.replace(-200, np.nan)
    df = df.drop(columns=["nmhc_gt"])                     # 90% missing
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


df = load_data()
split_idx = int(len(df) * 0.8)                            # chronological 80/20
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]       # ONLY M5 touches this, once

CV = TimeSeriesSplit(n_splits=5)


def pct_within(y_true, y_pred, tol=TOLERANCE):
    return 100 * np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol)


SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "within": make_scorer(pct_within),
}
PRIMARY = "mae"


def make_preprocessor(scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), FEATURES)])


RUNTIME = f"{platform.processor() or platform.machine()} | {os.cpu_count()} CPU cores | Python {platform.python_version()}"

results = []


def evaluate_search(name, gs):
    r, i = gs.cv_results_, gs.best_index_
    row = {
        "model": name,
        "train_MAE": -r["mean_train_mae"][i],
        "val_MAE": -r["mean_test_mae"][i],
        "val_MAE_std": r["std_test_mae"][i],
        "train_R2": r["mean_train_r2"][i],
        "val_R2": r["mean_test_r2"][i],
        "train_within_%": r["mean_train_within"][i],
        "val_within_%": r["mean_test_within"][i],
        "mean_fit_time_s": r["mean_fit_time"][i],
        "best_params": gs.best_params_,
    }
    row["overfit_gap_MAE"] = row["val_MAE"] - row["train_MAE"]
    results.append(row)
    return pd.Series(row)


def run_search(name, pipe, grid):
    gs = GridSearchCV(pipe, grid, cv=CV, scoring=SCORING, refit=PRIMARY,
                      return_train_score=True, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs, evaluate_search(name, gs)
