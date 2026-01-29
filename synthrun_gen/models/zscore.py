from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class PerAthleteHealthyZScorer:
    """Per-athlete z-scoring using *healthy* baseline (train only, y==0).

    - Fit on CC0 train rows only.
    - For each athlete and each feature, compute mean/std using rows where y==0.
    - If insufficient healthy samples for that athlete-feature, fall back to global healthy mean/std.
    """
    min_count: int = 10
    eps: float = 1e-6
    feature_cols: Optional[List[str]] = None

    # learned
    global_mean_: Dict[str, float] | None = None
    global_std_: Dict[str, float] | None = None
    athlete_mean_: Dict[str, Dict[str, float]] | None = None  # athlete -> feature -> mean
    athlete_std_: Dict[str, Dict[str, float]] | None = None

    def fit(self, df: pd.DataFrame, athlete_col: str, y_col: str, feature_cols: List[str]) -> "PerAthleteHealthyZScorer":
        self.feature_cols = list(feature_cols)
        healthy = df[df[y_col] == 0]

        self.global_mean_ = {}
        self.global_std_ = {}
        for c in self.feature_cols:
            x = pd.to_numeric(healthy[c], errors="coerce")
            mu = float(np.nanmean(x))
            sd = float(np.nanstd(x))
            self.global_mean_[c] = mu
            self.global_std_[c] = max(sd, self.eps)

        self.athlete_mean_ = {}
        self.athlete_std_ = {}
        for aid, g in healthy.groupby(athlete_col, sort=False):
            aid = str(aid)
            self.athlete_mean_[aid] = {}
            self.athlete_std_[aid] = {}
            for c in self.feature_cols:
                x = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
                cnt = int(np.isfinite(x).sum())
                if cnt >= self.min_count:
                    mu = float(np.nanmean(x))
                    sd = float(np.nanstd(x))
                    self.athlete_mean_[aid][c] = mu
                    self.athlete_std_[aid][c] = max(sd, self.eps)

        return self

    def transform(self, df: pd.DataFrame, athlete_col: str) -> pd.DataFrame:
        if self.feature_cols is None:
            raise RuntimeError("ZScorer not fit.")
        out = df.copy()
        for c in self.feature_cols:
            x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)
            mu = np.zeros(len(out), dtype=float)
            sd = np.ones(len(out), dtype=float)
            # vectorized per-athlete fill
            athletes = out[athlete_col].astype(str).to_numpy()
            for i, aid in enumerate(athletes):
                if self.athlete_mean_ and aid in self.athlete_mean_ and c in self.athlete_mean_[aid]:
                    mu[i] = self.athlete_mean_[aid][c]
                    sd[i] = self.athlete_std_[aid][c]
                else:
                    mu[i] = self.global_mean_[c] if self.global_mean_ else 0.0
                    sd[i] = self.global_std_[c] if self.global_std_ else 1.0
            out[c] = (x - mu) / sd
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_count": self.min_count,
            "eps": self.eps,
            "feature_cols": self.feature_cols,
            "global_mean": self.global_mean_,
            "global_std": self.global_std_,
            "athlete_mean": self.athlete_mean_,
            "athlete_std": self.athlete_std_,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PerAthleteHealthyZScorer":
        z = PerAthleteHealthyZScorer(min_count=int(d["min_count"]), eps=float(d["eps"]))
        z.feature_cols = list(d["feature_cols"])
        z.global_mean_ = {k: float(v) for k, v in d["global_mean"].items()}
        z.global_std_ = {k: float(v) for k, v in d["global_std"].items()}
        z.athlete_mean_ = {str(a): {k: float(v) for k, v in fm.items()} for a, fm in d["athlete_mean"].items()}
        z.athlete_std_ = {str(a): {k: float(v) for k, v in fs.items()} for a, fs in d["athlete_std"].items()}
        return z
