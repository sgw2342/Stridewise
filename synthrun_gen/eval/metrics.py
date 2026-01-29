from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

def safe_auc_roc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None

def safe_auc_pr(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import average_precision_score
        if len(np.unique(y)) < 2:
            return None
        return float(average_precision_score(y, p))
    except Exception:
        return None

def safe_brier(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import brier_score_loss
        return float(brier_score_loss(y, p))
    except Exception:
        return None

def precision_at_top_pct(y: np.ndarray, p: np.ndarray, pct: float) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    k = int(np.ceil(n * (pct / 100.0)))
    k = max(1, min(n, k))
    idx = np.argsort(-p)[:k]
    prec = float(y[idx].mean()) if k > 0 else float("nan")
    return {"pct": float(pct), "k": int(k), "precision": prec, "positives_in_topk": int(y[idx].sum())}

def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_edges, bin_count, bin_mean_p, bin_frac_pos)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(p, edges[1:-1], right=False)  # 0..n_bins-1
    counts = np.zeros(n_bins, dtype=int)
    mean_p = np.full(n_bins, np.nan, dtype=float)
    frac_pos = np.full(n_bins, np.nan, dtype=float)

    for b in range(n_bins):
        m = (bin_idx == b)
        counts[b] = int(m.sum())
        if counts[b] > 0:
            mean_p[b] = float(np.mean(p[m]))
            frac_pos[b] = float(np.mean(y[m]))
    return edges, counts, mean_p, frac_pos

def expected_calibration_error(counts: np.ndarray, mean_p: np.ndarray, frac_pos: np.ndarray) -> Optional[float]:
    counts = np.asarray(counts, dtype=float)
    if counts.sum() <= 0:
        return None
    w = counts / counts.sum()
    d = np.abs(np.nan_to_num(frac_pos, nan=0.0) - np.nan_to_num(mean_p, nan=0.0))
    return float(np.sum(w * d))

def alert_rates(p: np.ndarray, amber_threshold: float, red_threshold: float) -> Dict[str, Any]:
    p = np.asarray(p, dtype=float)
    red = (p >= red_threshold)
    amber = (p >= amber_threshold) & (~red)
    green = (p < amber_threshold)
    return {
        "red_rate_actual": float(red.mean()),
        "amber_rate_actual": float(amber.mean()),
        "green_rate_actual": float(green.mean()),
        "n_red": int(red.sum()),
        "n_amber": int(amber.sum()),
        "n_green": int(green.sum()),
    }

def per_athlete_summary(athlete_id: np.ndarray, y: np.ndarray, p: np.ndarray) -> List[Dict[str, Any]]:
    athlete_id = np.asarray(athlete_id).astype(str)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    out: List[Dict[str, Any]] = []
    for aid in np.unique(athlete_id):
        m = (athlete_id == aid)
        yy = y[m]
        pp = p[m]
        prev = float(yy.mean()) if len(yy) else float("nan")
        auc = safe_auc_roc(yy, pp)
        brier = safe_brier(yy, pp)
        out.append({
            "athlete_id": aid,
            "n_rows": int(m.sum()),
            "prevalence": prev,
            "auc_roc": auc,
            "brier": brier,
            "p_mean": float(np.mean(pp)) if len(pp) else float("nan"),
            "p_std": float(np.std(pp)) if len(pp) else float("nan"),
            "y_sum": int(yy.sum()),
        })
    return out

def summarize_numeric(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
