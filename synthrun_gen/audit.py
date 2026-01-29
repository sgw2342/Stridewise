from __future__ import annotations

"""Small, practical audits for synthetic realism + leakage smell tests.

These audits are designed to be:
  - fast (seconds, not minutes),
  - robust (work even with missing sensors),
  - actionable (produce simple CSV/JSON artifacts).

They do **not** guarantee realism, but they catch the most common failure modes:
  1) labels are mostly "user propensity" (easy on random splits, useless forward-time),
  2) key drivers have implausible directionality (e.g., ACWR high decile not riskier),
  3) time-drift in label rate (everything gets riskier late-season regardless of features).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, p))


def _safe_logloss(y_true: np.ndarray, p: np.ndarray, clip: float = 1e-7) -> float:
    from sklearn.metrics import log_loss

    p = np.asarray(p, dtype=float)
    p = np.clip(p, clip, 1.0 - clip)
    return float(log_loss(np.asarray(y_true, dtype=int), p))


def label_rate_by_decile(
    df: pd.DataFrame,
    label_col: str,
    cols: Iterable[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute label rate by quantile bin for a set of columns.

    Returns tidy long-form dataframe with:
      col, bin, n, injury_rate, x_mean, x_p50
    """
    out: List[pd.DataFrame] = []
    y = pd.to_numeric(df[label_col], errors="coerce")
    for col in cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < max(200, n_bins * 50):
            continue
        try:
            bins = pd.qcut(x[ok], q=n_bins, duplicates="drop")
        except Exception:
            continue
        g = (
            pd.DataFrame({"x": x[ok], "y": y[ok], "bin": bins})
            .groupby("bin", observed=True)
            .agg(n=("y", "size"), injury_rate=("y", "mean"), x_mean=("x", "mean"), x_p50=("x", "median"))
            .reset_index()
        )
        g.insert(0, "col", col)
        g["injury_rate"] = g["injury_rate"].astype(float)
        out.append(g)
    if not out:
        return pd.DataFrame(columns=["col", "bin", "n", "injury_rate", "x_mean", "x_p50"])
    return pd.concat(out, ignore_index=True)


def per_user_forward_time_rate_baseline(
    df: pd.DataFrame,
    label_col: str,
    user_col: str = "user_id",
    date_col: str = "date",
    val_frac: float = 0.2,
) -> Dict[str, float | int | None]:
    """A baseline that predicts a user's *train* label rate onto their *future* (val).

    If this baseline has high AUC, labels are dominated by per-user propensity rather than
    day-to-day signals. That's a warning sign for forward-time predictive modelling.
    """
    df = df[[user_col, date_col, label_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([user_col, date_col])
    y_all = []
    p_all = []
    for _, g in df.groupby(user_col, sort=False):
        y = pd.to_numeric(g[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
        n = len(y)
        if n < 10:
            continue
        n_val = max(1, int(round(n * val_frac)))
        tr = y[: n - n_val]
        va = y[n - n_val :]
        # user prevalence from training segment
        p_user = float(tr.mean())
        y_all.append(va)
        p_all.append(np.full_like(va, p_user, dtype=float))
    if not y_all:
        return {"n_val": 0, "auc": None, "logloss": None}
    yv = np.concatenate(y_all)
    pv = np.concatenate(p_all)
    return {
        "n_val": int(len(yv)),
        "auc": _safe_auc(yv, pv),
        "logloss": _safe_logloss(yv, pv),
        "p_mean": float(np.mean(pv)),
        "y_mean": float(np.mean(yv)),
    }


def label_time_trend(
    df: pd.DataFrame,
    label_col: str,
    date_col: str = "date",
    freq: str = "W",
) -> pd.DataFrame:
    """Compute label rate over time (global) to spot strong time drift."""
    d = df[[date_col, label_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d[label_col] = pd.to_numeric(d[label_col], errors="coerce").fillna(0).astype(int)
    d["bucket"] = d[date_col].dt.to_period(freq).dt.start_time
    g = d.groupby("bucket").agg(n=(label_col, "size"), rate=(label_col, "mean")).reset_index()
    return g
