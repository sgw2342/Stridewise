from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import GeneratorConfig

def _write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

def _write_parquet(df: pd.DataFrame, path: Path):
    # Try pyarrow; if missing, raise a clear error
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        raise RuntimeError(
            "Parquet write failed. Install pyarrow (recommended) or use --format csv."
        ) from e

def write_outputs(users: pd.DataFrame, daily: pd.DataFrame, activities: pd.DataFrame,
                  out_dir: Path, cfg: GeneratorConfig) -> dict:
    written = {}

    fmt = cfg.out_format
    if fmt in ("csv", "both"):
        up = out_dir / "users.csv"
        dp = out_dir / "daily.csv"
        ap = out_dir / "activities.csv"
        _write_csv(users, up); _write_csv(daily, dp); _write_csv(activities, ap)
        written["users_csv"] = str(up)
        written["daily_csv"] = str(dp)
        written["activities_csv"] = str(ap)

    if fmt in ("parquet", "both"):
        up = out_dir / "users.parquet"
        dp = out_dir / "daily.parquet"
        ap = out_dir / "activities.parquet"
        _write_parquet(users, up); _write_parquet(daily, dp); _write_parquet(activities, ap)
        written["users_parquet"] = str(up)
        written["daily_parquet"] = str(dp)
        written["activities_parquet"] = str(ap)

    return written
