from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def safe_log1p(x):
    return np.log1p(np.maximum(x, 0.0))

def rolling_mean(a, window: int):
    # a: 1d np array
    if window <= 1:
        return a.astype(float)
    s = pd.Series(a)
    return s.rolling(window, min_periods=1).mean().to_numpy()

def rolling_sum(a, window: int):
    if window <= 1:
        return a.astype(float)
    s = pd.Series(a)
    return s.rolling(window, min_periods=1).sum().to_numpy()

def ou_step(prev: float, mean: float, kappa: float, noise_sd: float, rng: np.random.Generator) -> float:
    # Discrete-time OU with mean reversion strength kappa (0..1)
    # x_t = x_{t-1} + kappa*(mean - x_{t-1}) + eps
    return float(prev + kappa*(mean - prev) + rng.normal(0.0, noise_sd))

def date_range(start_date: str, n_days: int):
    start = pd.to_datetime(start_date)
    return pd.date_range(start, periods=n_days, freq="D")

def as_float32(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    return df
