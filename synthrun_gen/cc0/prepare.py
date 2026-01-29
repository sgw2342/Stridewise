from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from .schema import CC0FeatureSchema

def _detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _ensure_row_idx(df: pd.DataFrame, row_idx_col: str) -> pd.DataFrame:
    df = df.copy()
    if row_idx_col not in df.columns:
        df[row_idx_col] = np.arange(len(df), dtype=int)
    return df

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _make_window_cols(base: str, suffixes: List[str]) -> List[str]:
    return [f"{base}.{s}" for s in suffixes]

def prepare_cc0_features(
    day_csv: str,
    week_csv: Optional[str],
    schema_path: str,
    label_col: Optional[str] = None,
    keep_meta_cols: Optional[List[str]] = None,
    engineer_features: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load CC0 day/week timeseries and return aligned feature table.

    Output columns:
      row_idx, athlete_id, y, <flattened features>, plus engineered:
        - subjective_missing_* indicators
        - rest_day_* indicators
        - clipped/logged ratio features (weekly)
    """
    schema = CC0FeatureSchema.from_json(schema_path)

    day = pd.read_csv(day_csv)
    day = _ensure_row_idx(day, schema.row_idx_col)

    # detect id + label
    id_col = _detect_col(day, schema.id_col_candidates)
    if id_col is None:
        raise ValueError(f"Could not detect athlete id column in day file. Tried: {schema.id_col_candidates}")

    y_col = label_col or _detect_col(day, schema.label_col_candidates)
    if y_col is None:
        raise ValueError(f"Could not detect label column. Provide --label-col. Tried: {schema.label_col_candidates}")

    # Optional week
    if week_csv:
        week = pd.read_csv(week_csv)
        week = _ensure_row_idx(week, schema.row_idx_col)
        # join on row_idx, and also id if present (defensive)
        join_cols = [schema.row_idx_col]
        if id_col in week.columns:
            join_cols.append(id_col)
        df = day.merge(week, on=join_cols, how="left", suffixes=("","_wk"))
    else:
        df = day.copy()

    # Build list of windowed numeric columns to coerce
    numeric_cols = []
    for f in schema.day_features:
        numeric_cols.extend(_make_window_cols(f, schema.day_window_suffixes))
    for f in schema.week_features:
        numeric_cols.extend(_make_window_cols(f, schema.week_window_suffixes))
    df = _coerce_numeric(df, [c for c in numeric_cols if c in df.columns])

    # Subjectives: treat -0.01 as missing, add indicators per window day
    subj = ["perceived_exertion", "perceived_trainingSuccess", "perceived_recovery"]
    for f in subj:
        if f not in schema.day_features:
            continue
        for s in schema.day_window_suffixes:
            c = f"{f}.{s}"
            if c in df.columns:
                miss = (df[c] <= schema.subjective_rest_value + 1e-9) | df[c].isna()
                df[f"{c}_is_missing"] = miss.astype("int8")
                df.loc[miss, c] = np.nan

    # Rest-day indicators per day in window
    for s in schema.day_window_suffixes:
        c_sess = f"sessions.{s}"
        c_kms = f"kms.{s}"
        if c_sess in df.columns or c_kms in df.columns:
            sess0 = (df[c_sess] <= 0.0) if c_sess in df.columns else False
            kms0 = (df[c_kms] <= 0.0) if c_kms in df.columns else False
            df[f"rest_day.{s}"] = (sess0 | kms0).astype("int8")

    # Weekly ratio transforms: clip + optional log1p for any rel_* features
    clip_min, clip_max = schema.ratio_clip_min, schema.ratio_clip_max
    for f in schema.week_features:
        if not f.startswith("rel_"):
            continue
        for s in schema.week_window_suffixes:
            c = f"{f}.{s}"
            if c in df.columns:
                df[c] = df[c].clip(clip_min, clip_max)
                if schema.ratio_log1p:
                    df[f"{c}_log1p"] = np.log1p(df[c].astype(float))

    # Flattened feature list: all day/week window cols + engineered indicators/logs
    feature_cols = []
    for f in schema.day_features:
        for s in schema.day_window_suffixes:
            # For empty suffix, use base name without dot (real CC0 format)
            # For non-empty suffix, use f"{f}.{s}" format
            if s == '':
                c = f  # Base column name (e.g., "total km")
            else:
                c = f"{f}.{s}"  # With suffix (e.g., "total km.1")
            
            if c in df.columns:
                feature_cols.append(c)
            # indicators that may exist
            c_m = f"{c}_is_missing"
            if c_m in df.columns:
                feature_cols.append(c_m)
        # rest_day indicators already handled separately
    for s in schema.day_window_suffixes:
        if s == '':
            c = "rest_day"  # Base rest_day column
        else:
            c = f"rest_day.{s}"
        if c in df.columns:
            feature_cols.append(c)
    
    # If schema.day_features is empty or doesn't match, include all columns that look like CC0 features
    # This handles synthetic CC0 files that may have different column naming
    if len(feature_cols) == 0 or len(feature_cols) < 10:
        # Include all columns that match CC0 pattern (base.suffix)
        for col in df.columns:
            if col in ["row_idx", "athlete_id", "y", "maskedID", "anchor_date"]:
                continue
            # Check if it matches CC0 pattern (has a dot or is a known feature)
            if "." in col or col in ["sessions", "kms", "total km", "nr. sessions"]:
                if col not in feature_cols:
                    feature_cols.append(col)

    for f in schema.week_features:
        for s in schema.week_window_suffixes:
            c = f"{f}.{s}"
            if c in df.columns:
                feature_cols.append(c)
            c_log = f"{c}_log1p"
            if c_log in df.columns:
                feature_cols.append(c_log)

    # Preserve optional meta columns (e.g., anchor_date for synthetic CC0-view)
    meta_cols = []
    if keep_meta_cols is None:
        keep_meta_cols = ["anchor_date"]
    for c in keep_meta_cols:
        if c in df.columns:
            meta_cols.append(c)

    out = pd.DataFrame({
        "row_idx": df[schema.row_idx_col].astype(int),
        "athlete_id": df[id_col].astype(str),
        "y": df[y_col].astype(int),
    })
    for c in meta_cols:
        out[c] = df[c].astype(str)

    out = pd.concat([out, df[feature_cols]], axis=1)

    meta = {
        "schema_version": schema.version,
        "id_col": id_col,
        "label_col": y_col,
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "meta_cols": meta_cols,
    }

    return out, meta
