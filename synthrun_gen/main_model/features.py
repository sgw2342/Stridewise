from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

LEAKAGE_DROP_COLS = [
    "injury_next_7d",
    "injury_onset",
    "injury_ongoing",
    "illness_next_7d",
    "illness_onset",
    "illness_ongoing",
    "injury_next_14d",
    "injury_next_3d",
    "injury_next_1d",
]

ID_COLS = ["user_id", "date"]


@dataclass
class MainModelSchema:
    label_col: str
    id_cols: List[str]
    drop_cols: List[str]
    feature_cols: List[str]
    feature_dtypes: Dict[str, str]
    feature_notes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    return df


def _encode_known_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic encodings to avoid category-code drift between train and inference."""
    df = df.copy()
    if "session_type" in df.columns:
        mapping = {"rest": -1, "easy": 0, "tempo": 1, "interval": 2, "long": 3}
        df["session_type"] = df["session_type"].map(mapping).fillna(-1).astype("int16")
    if "sex" in df.columns:
        mapping = {"F": 0, "M": 1}
        df["sex"] = df["sex"].map(mapping).fillna(-1).astype("int16")
    return df


def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add robust rolling features per athlete (enhanced version)."""
    df = df.sort_values(["user_id", "date"]).copy()
    grp = df.groupby("user_id", sort=False)

    def roll_mean(col: str, w: int):
        return grp[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    def roll_std(col: str, w: int):
        # ddof=0 keeps small-window std stable
        return grp[col].transform(lambda s: s.shift(1).rolling(w, min_periods=2).std(ddof=0))

    def roll_sum(col: str, w: int):
        return grp[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum())
    
    def roll_max(col: str, w: int):
        return grp[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).max())
    
    def roll_min(col: str, w: int):
        return grp[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).min())

    # Standard rolling features
    for col in [
        "training_load",
        "km_total",
        "duration_min",
        "sleep_hours",
        "stress_score",
        "rhr_bpm",
        "hrv_ms",
    ]:
        if col in df.columns:
            df[f"{col}_mean7"] = roll_mean(col, 7)
            df[f"{col}_mean28"] = roll_mean(col, 28)
            df[f"{col}_std28"] = roll_std(col, 28)

    # Sum features for load metrics
    for col in ["training_load", "km_total", "duration_min"]:
        if col in df.columns:
            df[f"{col}_sum7"] = roll_sum(col, 7)
            df[f"{col}_sum28"] = roll_sum(col, 28)
            # Also add max for spikes
            df[f"{col}_max7"] = roll_max(col, 7)
    
    # Min features for recovery metrics
    for col in ["sleep_hours", "hrv_ms"]:
        if col in df.columns:
            df[f"{col}_min7"] = roll_min(col, 7)
            df[f"{col}_min28"] = roll_min(col, 28)
    
    # Max features for stress metrics
    for col in ["stress_score", "rhr_bpm"]:
        if col in df.columns:
            df[f"{col}_max7"] = roll_max(col, 7)
            df[f"{col}_max28"] = roll_max(col, 28)

    # Within-athlete deltas and z-scores (helps forward-time generalisation)
    eps = 1e-6
    for col in [
        "training_load",
        "km_total",
        "duration_min",
        "sleep_hours",
        "stress_score",
        "rhr_bpm",
        "hrv_ms",
    ]:
        m7 = f"{col}_mean7"
        m28 = f"{col}_mean28"
        s28 = f"{col}_std28"
        if m7 in df.columns and m28 in df.columns:
            df[f"{col}_delta7_28"] = pd.to_numeric(df[m7], errors="coerce") - pd.to_numeric(df[m28], errors="coerce")
        if m7 in df.columns and m28 in df.columns and s28 in df.columns:
            denom = pd.to_numeric(df[s28], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            denom = np.maximum(denom, eps)
            df[f"{col}_z7_28"] = (
                (pd.to_numeric(df[m7], errors="coerce") - pd.to_numeric(df[m28], errors="coerce")).to_numpy(dtype=float)
                / denom
            )

    if "acwr" in df.columns:
        df["acwr_clipped"] = np.clip(df["acwr"].astype(float), 0.0, 5.0)

    if {"kms_z3_4", "kms_z5_t1_t2", "km_total"}.issubset(df.columns):
        tot = np.maximum(df["km_total"].astype(float), 1e-6)
        df["hi_km"] = df["kms_z3_4"].astype(float) + df["kms_z5_t1_t2"].astype(float)
        df["hi_share"] = np.clip(df["hi_km"] / tot, 0.0, 1.0)

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features that capture profile-specific risk."""
    df = df.copy()
    
    # ACWR × fitness interaction (novices more sensitive to ACWR)
    if 'acwr_clipped' in df.columns and 'fitness' in df.columns:
        df['acwr_x_fitness'] = df['acwr_clipped'] * (1.0 - df['fitness'])
        df['acwr_x_high_fitness'] = df['acwr_clipped'] * df['fitness']
    
    # Absolute load × fitness (novices at risk with high absolute loads)
    if 'training_load_sum7' in df.columns and 'fitness' in df.columns:
        low_fitness = (df['fitness'] < 0.4).astype(float)
        high_load = np.clip(df['training_load_sum7'] / 500.0, 0, 2.0)
        df['load7_x_low_fitness'] = df['training_load_sum7'] * low_fitness
        df['load7_x_high_load'] = high_load
    
    # ACWR × absolute load (high ACWR + high load = extra risk)
    if 'acwr_clipped' in df.columns and 'training_load_sum7' in df.columns:
        load_normalized = np.clip(df['training_load_sum7'] / 500.0, 0, 2.0)
        df['acwr_x_load7'] = df['acwr_clipped'] * load_normalized
    
    # ACWR excess (above threshold) × fitness
    if 'acwr_clipped' in df.columns and 'fitness' in df.columns:
        acwr_excess = np.clip(df['acwr_clipped'] - 1.1, 0, 2.5)
        df['acwr_excess_x_fitness'] = acwr_excess * (1.0 - df['fitness'])
    
    return df


def _add_ramp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ramp and spike features for training load changes."""
    df = df.copy()
    df = df.sort_values(['user_id', 'date'])
    
    grp = df.groupby('user_id', sort=False)
    
    # Week-to-week ramp: last 7d vs previous 7d (both lagged)
    if 'training_load' in df.columns:
        load7 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).sum()
        )
        load7_prev = load7.shift(7)
        df['ramp_ratio'] = (load7 / (load7_prev + 1e-6)).fillna(1.0)
        df['ramp_excess'] = np.clip(df['ramp_ratio'] - 1.15, 0, 2.5)
        df['ramp_high'] = (df['ramp_ratio'] > 1.3).astype(float)
    
    # Session spike: max single-day load vs 7d mean (lagged)
    if 'training_load' in df.columns:
        load_daily = df['training_load']
        load_mean7 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).mean()
        )
        df['session_spike_ratio'] = (load_daily / (load_mean7 + 1e-6)).fillna(1.0)
        df['session_spike_mid'] = ((df['session_spike_ratio'] > 1.3) & 
                                    (df['session_spike_ratio'] <= 1.8)).astype(float)
        df['session_spike_high'] = (df['session_spike_ratio'] > 1.8).astype(float)
    
    # Similar for km_total
    if 'km_total' in df.columns:
        km7 = grp['km_total'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).sum()
        )
        km7_prev = km7.shift(7)
        df['km_ramp_ratio'] = (km7 / (km7_prev + 1e-6)).fillna(1.0)
        df['km_ramp_excess'] = np.clip(df['km_ramp_ratio'] - 1.15, 0, 2.5)
    
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features (day-of-week, week-of-year, time-since-injury)."""
    df = df.copy()
    
    if 'date' not in df.columns:
        return df
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['date'].dt.dayofweek.astype('int8')
    
    # Is weekend (Saturday or Sunday)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    
    # Week of year (1-52/53)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype('int8')
    
    # Month (1-12)
    df['month'] = df['date'].dt.month.astype('int8')
    
    # Days since last injury (if injury_onset available)
    if 'injury_onset' in df.columns:
        df = df.sort_values(['user_id', 'date'])
        grp = df.groupby('user_id', sort=False)
        
        # Create a flag for injury days
        injury_flag = (pd.to_numeric(df['injury_onset'], errors='coerce').fillna(0) == 1).astype(int)
        
        # Count days since last injury
        days_since = []
        for uid, group in grp:
            group_injuries = group[injury_flag[group.index] == 1].index
            group_dates = group['date'].values
            
            for i, date in enumerate(group_dates):
                # Find last injury before this date
                prev_injuries = group_injuries[group_injuries < group.index[i]]
                if len(prev_injuries) > 0:
                    last_injury_idx = prev_injuries[-1]
                    days = (date - group.loc[last_injury_idx, 'date']).days
                    days_since.append(days)
                else:
                    days_since.append(999)  # No previous injury
        
        df['days_since_injury'] = pd.Series(days_since, index=df.index).astype('int16')
        # Clip to reasonable range
        df['days_since_injury'] = np.clip(df['days_since_injury'], 0, 365)
    else:
        # If no injury_onset, set to a large value
        df['days_since_injury'] = 999
    
    # PHASE 2: Ensure we have sorted data and groupby for new features
    df = df.sort_values(['user_id', 'date'])
    grp = df.groupby('user_id', sort=False)
    
    # PHASE 2: ACWR slope and acceleration features
    if 'acwr_clipped' in df.columns:
        acwr_lag1 = grp['acwr_clipped'].transform(lambda s: s.shift(1))
        acwr_lag7 = grp['acwr_clipped'].transform(lambda s: s.shift(7))
        df['acwr_slope_7d'] = ((acwr_lag1 - acwr_lag7) / 7.0).astype('float32')
        
        # ACWR acceleration (change in slope)
        acwr_lag2 = grp['acwr_clipped'].transform(lambda s: s.shift(2))
        acwr_slope_prev = ((acwr_lag2 - acwr_lag7) / 7.0).astype('float32')
        df['acwr_acceleration'] = (df['acwr_slope_7d'] - acwr_slope_prev).astype('float32')
    
    # PHASE 2: Recovery slope features
    if 'recovery_index' in df.columns:
        recovery_lag1 = grp['recovery_index'].transform(lambda s: s.shift(1))
        recovery_lag7 = grp['recovery_index'].transform(lambda s: s.shift(7))
        df['recovery_slope_7d'] = ((recovery_lag1 - recovery_lag7) / 7.0).astype('float32')
        
        # Recovery improving vs declining (binary indicators)
        df['recovery_improving'] = (df['recovery_slope_7d'] > 0.01).astype('int8')
        df['recovery_declining'] = (df['recovery_slope_7d'] < -0.01).astype('int8')
    
    # PHASE 2: Lag features for key risk factors (t-2, t-3, t-7)
    lag_cols = ['acwr_clipped', 'risk_score', 'sleep_hours', 'hrv_ms', 'rhr_bpm', 'stress_score']
    available_lag_cols = [col for col in lag_cols if col in df.columns]
    
    for col in available_lag_cols:
        # Lag 2 days
        df[f'{col}_lag2'] = grp[col].transform(lambda s: s.shift(2)).astype('float32')
        # Lag 3 days
        df[f'{col}_lag3'] = grp[col].transform(lambda s: s.shift(3)).astype('float32')
        # Lag 7 days
        df[f'{col}_lag7'] = grp[col].transform(lambda s: s.shift(7)).astype('float32')
    
    # PHASE 2: Spike events in last 7/14 days (binary indicators)
    if 'has_long_run_spike' in df.columns:
        df['spike_in_last_7d'] = (
            grp['has_long_run_spike'].transform(
                lambda s: s.shift(1).rolling(7, min_periods=1).sum() > 0
            )
        ).fillna(0).astype('int8')
        
        df['spike_in_last_14d'] = (
            grp['has_long_run_spike'].transform(
                lambda s: s.shift(1).rolling(14, min_periods=1).sum() > 0
            )
        ).fillna(0).astype('int8')
    elif 'long_run_spike_category' in df.columns:
        df['spike_in_last_7d'] = (
            grp['long_run_spike_category'].transform(
                lambda s: (s.shift(1) > 0).astype(int).rolling(7, min_periods=1).sum() > 0
            )
        ).fillna(0).astype('int8')
        
        df['spike_in_last_14d'] = (
            grp['long_run_spike_category'].transform(
                lambda s: (s.shift(1) > 0).astype(int).rolling(14, min_periods=1).sum() > 0
            )
        ).fillna(0).astype('int8')
    
    return df


def _add_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add recovery-related features."""
    df = df.copy()
    df = df.sort_values(['user_id', 'date'])
    
    grp = df.groupby('user_id', sort=False)
    
    # Consecutive rest days (days with < 1km)
    if 'km_total' in df.columns:
        df['consecutive_rest_days'] = (
            grp['km_total'].transform(
                lambda s: (s.fillna(0) < 1.0).astype(int)
                .groupby((s.fillna(0) >= 1.0).astype(int).cumsum())
                .cumsum()
            )
        ).astype('int8')
    
    # Recovery index: sleep × HRV (higher = better recovery)
    if 'sleep_hours' in df.columns and 'hrv_ms' in df.columns:
        sleep_norm = np.clip((df['sleep_hours'] - 6.0) / 2.0, -1, 1)
        hrv_norm = np.clip((df['hrv_ms'] - 50.0) / 20.0, -1, 1)
        df['recovery_index'] = (sleep_norm * hrv_norm).astype('float32')
        df['sleep_adequate'] = (df['sleep_hours'] >= 7.0).astype('int8')
        df['hrv_adequate'] = (df['hrv_ms'] >= 55.0).astype('int8')
    
    # Training load monotony (coefficient of variation)
    if 'training_load' in df.columns:
        load_mean7 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).mean()
        )
        load_std7 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).std(ddof=0)
        )
        df['load_monotony'] = (load_std7 / (load_mean7 + 1e-6)).fillna(0.0).astype('float32')
    
    # Acute:chronic ratio squared (captures non-linearity)
    if 'acwr_clipped' in df.columns:
        df['acwr_squared'] = (df['acwr_clipped'] ** 2).astype('float32')
    
    # NEW: Multi-day recovery deficit (cumulative poor recovery)
    if 'recovery_index' in df.columns:
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        poor_recovery = np.clip(-recovery, 0, 1)  # Poor recovery = high value
        df['recovery_deficit_7d'] = (
            grp['recovery_index'].transform(
                lambda s: poor_recovery.shift(1).rolling(7, min_periods=7).sum()
            )
        ).astype('float32')
        df['recovery_deficit_14d'] = (
            grp['recovery_index'].transform(
                lambda s: poor_recovery.shift(1).rolling(14, min_periods=14).sum()
            )
        ).astype('float32')
    
    # NEW: Recovery trend (improving vs deteriorating)
    if 'recovery_index' in df.columns:
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        # FIX: Use min_periods=1 and forward-fill to handle NaN values in recovery_index
        # This prevents recovery_trend from having 94% NaN when recovery_index has 22% NaN
        recovery_mean7 = grp['recovery_index'].transform(
            lambda s: s.shift(1).ffill().fillna(0.0).rolling(7, min_periods=1).mean()
        )
        recovery_mean14 = grp['recovery_index'].transform(
            lambda s: s.shift(1).ffill().fillna(0.0).rolling(14, min_periods=1).mean()
        )
        df['recovery_trend'] = (recovery_mean7 - recovery_mean14).astype('float32')  # Positive = improving
    
    # NEW: Recovery × Load interaction (poor recovery + high load)
    if 'recovery_index' in df.columns and 'training_load' in df.columns:
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        poor_recovery = np.clip(-recovery, 0, 1)
        load_mean7 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).mean()
        )
        load_normalized = np.clip(load_mean7 / 500.0, 0, 2.0)  # Normalize load
        df['recovery_x_load7'] = (poor_recovery * load_normalized).astype('float32')
    
    # NEW: Recovery × ACWR interaction (poor recovery + high ACWR)
    if 'recovery_index' in df.columns and 'acwr_clipped' in df.columns:
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        poor_recovery = np.clip(-recovery, 0, 1)
        acwr_excess = np.clip(df['acwr_clipped'] - 1.1, 0.0, 2.5)
        df['recovery_x_acwr_excess'] = (poor_recovery * acwr_excess).astype('float32')
    
    return df


def _add_directional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add directional features matching injury generation logic.
    
    Injury generation uses directional transforms:
    - sleep_def = clip(-sleep_z, 0, 3)  # Only negative z-scores (deficit)
    - hrv_drop = clip(-hrv_z, 0, 3)     # Only negative z-scores (drop)
    - rhr_rise = clip(rhr_z, 0, 3)      # Only positive z-scores (rise)
    - stress_rise = clip(stress_z, 0, 3) # Only positive z-scores (rise)
    """
    df = df.copy()
    
    # Sleep deficit (only negative z-scores matter - low sleep is risky)
    if 'sleep_hours_z7_28' in df.columns:
        sleep_z = pd.to_numeric(df['sleep_hours_z7_28'], errors='coerce').fillna(0.0)
        df['sleep_def'] = np.clip(-sleep_z, 0.0, 3.0).astype('float32')
    
    # HRV drop (only negative z-scores matter - low HRV is risky)
    if 'hrv_ms_z7_28' in df.columns:
        hrv_z = pd.to_numeric(df['hrv_ms_z7_28'], errors='coerce').fillna(0.0)
        df['hrv_drop'] = np.clip(-hrv_z, 0.0, 3.0).astype('float32')
    
    # RHR rise (only positive z-scores matter - high RHR is risky)
    if 'rhr_bpm_z7_28' in df.columns:
        rhr_z = pd.to_numeric(df['rhr_bpm_z7_28'], errors='coerce').fillna(0.0)
        df['rhr_rise'] = np.clip(rhr_z, 0.0, 3.0).astype('float32')
    
    # Stress rise (only positive z-scores matter - high stress is risky)
    if 'stress_score_z7_28' in df.columns:
        stress_z = pd.to_numeric(df['stress_score_z7_28'], errors='coerce').fillna(0.0)
        df['stress_rise'] = np.clip(stress_z, 0.0, 3.0).astype('float32')
    
    return df


def _add_excess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add excess features matching injury generation thresholds.
    
    Injury generation only considers ACWR/ramp above thresholds:
    - acwr_excess = clip(acwr - 1.1, 0, 2.5)
    - ramp_excess = clip(ramp_ratio - 1.15, 0, 2.5)
    """
    df = df.copy()
    
    # ACWR excess (above 1.1 threshold, matching injury generation)
    if 'acwr_clipped' in df.columns:
        acwr = pd.to_numeric(df['acwr_clipped'], errors='coerce').fillna(1.0)
        df['acwr_excess'] = np.clip(acwr - 1.1, 0.0, 2.5).astype('float32')
    
    # Ramp excess (above 1.15 threshold, matching injury generation)
    if 'ramp_ratio' in df.columns:
        ramp = pd.to_numeric(df['ramp_ratio'], errors='coerce').fillna(1.0)
        df['ramp_excess'] = np.clip(ramp - 1.15, 0.0, 2.5).astype('float32')
    
    return df


def _add_acwr_recovery_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add ACWR × recovery interactions to capture true risk.
    
    High ACWR + poor recovery = high risk (fixes the reversed ACWR signal).
    """
    df = df.copy()
    
    # ACWR excess × poor recovery (high ACWR + poor recovery = high risk)
    if 'acwr_excess' in df.columns and 'recovery_index' in df.columns:
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        poor_recovery = (recovery < -0.5).astype(float)
        df['acwr_excess_x_poor_recovery'] = (df['acwr_excess'] * poor_recovery).astype('float32')
    
    # ACWR excess × sleep deficit
    if 'acwr_excess' in df.columns and 'sleep_def' in df.columns:
        df['acwr_excess_x_sleep_def'] = (df['acwr_excess'] * df['sleep_def']).astype('float32')
    
    # ACWR excess × HRV drop
    if 'acwr_excess' in df.columns and 'hrv_drop' in df.columns:
        df['acwr_excess_x_hrv_drop'] = (df['acwr_excess'] * df['hrv_drop']).astype('float32')
    
    # ACWR excess × RHR rise
    if 'acwr_excess' in df.columns and 'rhr_rise' in df.columns:
        df['acwr_excess_x_rhr_rise'] = (df['acwr_excess'] * df['rhr_rise']).astype('float32')
    
    return df


def _add_persistent_fatigue_state(df: pd.DataFrame) -> pd.DataFrame:
    """Add persistent fatigue state feature matching injury generation.
    
    Injury generation uses persistent fatigue state (alpha=0.9) which creates
    forecastable patterns. This feature captures that state.
    """
    df = df.copy()
    df = df.sort_values(['user_id', 'date'])
    grp = df.groupby('user_id', sort=False)
    
    # Build risk components matching injury generation
    if all(c in df.columns for c in ['acwr_excess', 'sleep_def', 'hrv_drop']):
        # Raw risk (matching injury generation weights)
        risk_raw = (
            0.55 * df['acwr_excess'].fillna(0) +
            0.20 * df['sleep_def'].fillna(0) +
            0.18 * df['hrv_drop'].fillna(0)
        )
        
        # Add RHR and stress if available
        if 'rhr_rise' in df.columns:
            risk_raw += 0.18 * df['rhr_rise'].fillna(0)
        if 'stress_rise' in df.columns:
            risk_raw += 0.12 * df['stress_rise'].fillna(0)
        
        # Apply persistent fatigue state (alpha=0.9, matching injury generation)
        alpha = 0.9
        fatigue_state = []
        for uid, group in grp:
            group_risk = risk_raw[group.index].values
            state = np.zeros(len(group_risk))
            for i in range(len(group_risk)):
                prev = state[i-1] if i > 0 else 0.0
                raw_val = float(group_risk[i]) if not np.isnan(group_risk[i]) else 0.0
                state[i] = alpha * prev + (1.0 - alpha) * raw_val
                state[i] = float(np.clip(state[i], 0.0, 4.0))
            fatigue_state.extend(state)
        
        df['fatigue_state'] = pd.Series(fatigue_state, index=df.index).astype('float32')
        
        # Also add lagged fatigue state (previous day)
        df['fatigue_state_lag1'] = grp['fatigue_state'].transform(lambda s: s.shift(1)).astype('float32')
        
    return df


def _add_proneness_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add injury proneness interaction features.
    
    High proneness users are more sensitive to:
    - High ACWR
    - High fatigue state
    - Low fitness
    - Poor recovery
    """
    df = df.copy()
    
    if 'injury_proneness' not in df.columns:
        return df
    
    proneness = pd.to_numeric(df['injury_proneness'], errors='coerce').fillna(0.5)
    
    # Proneness × ACWR (high proneness + high ACWR = very high risk)
    if 'acwr_excess' in df.columns:
        df['proneness_x_acwr_excess'] = (proneness * df['acwr_excess']).astype('float32')
    
    # Proneness × fatigue state (high proneness + high fatigue = very high risk)
    if 'fatigue_state' in df.columns:
        df['proneness_x_fatigue'] = (proneness * df['fatigue_state']).astype('float32')
    
    # Proneness × low fitness (high proneness + low fitness = very high risk)
    if 'fitness' in df.columns:
        fitness = pd.to_numeric(df['fitness'], errors='coerce').fillna(0.5)
        low_fitness = (1.0 - fitness)  # Invert: low fitness = high value
        df['proneness_x_low_fitness'] = (proneness * low_fitness).astype('float32')
    
    # Proneness × poor recovery (high proneness + poor recovery = very high risk)
    if 'recovery_index' in df.columns:
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        poor_recovery = np.clip(-recovery, 0, 1)  # Poor recovery = high value
        df['proneness_x_poor_recovery'] = (proneness * poor_recovery).astype('float32')
    
    # Proneness × sleep deficit (high proneness + sleep deficit = very high risk)
    if 'sleep_def' in df.columns:
        df['proneness_x_sleep_def'] = (proneness * df['sleep_def']).astype('float32')
    
    return df


def _add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sequence/pattern features (consecutive high-risk days, spike frequency, etc.)."""
    df = df.copy()
    df = df.sort_values(['user_id', 'date'])
    grp = df.groupby('user_id', sort=False)
    
    # PHASE 2: Fixed consecutive high-risk day features (using lagged ACWR)
    if 'acwr_clipped' in df.columns:
        # PHASE 2 FIX: Count how many of the last N days had high ACWR (using lagged values)
        acwr_lagged = grp['acwr_clipped'].transform(lambda s: s.shift(1))
        high_risk = (acwr_lagged > 1.2).astype(int)
        
        # Count high-risk days in rolling windows
        df['high_risk_days_last_7d'] = (
            grp['acwr_clipped'].transform(
                lambda s: high_risk.rolling(7, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        df['high_risk_days_last_14d'] = (
            grp['acwr_clipped'].transform(
                lambda s: high_risk.rolling(14, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        # Keep original consecutive_high_risk_days for backward compatibility
        df['consecutive_high_risk_days'] = (
            grp['acwr_clipped'].transform(
                lambda s: high_risk.groupby((high_risk == 0).cumsum()).cumsum()
            )
        ).fillna(0).astype('int8')
        
        # PHASE 2: Add categorical features for 2+, 3+, 5+ consecutive days
        df['consecutive_high_risk_2plus'] = (df['high_risk_days_last_7d'] >= 2).astype('int8')
        df['consecutive_high_risk_3plus'] = (df['high_risk_days_last_7d'] >= 3).astype('int8')
        df['consecutive_high_risk_5plus'] = (df['high_risk_days_last_14d'] >= 5).astype('int8')
    
    # PHASE 1: Enhanced spike frequency features (14d, 30d, 60d windows)
    # Use has_long_run_spike if available (binary), otherwise use long_run_spike_risk > 0
    if 'has_long_run_spike' in df.columns:
        spike_indicator = df['has_long_run_spike']
        spike_col = 'has_long_run_spike'
    elif 'long_run_spike_risk' in df.columns:
        spike_indicator = (df['long_run_spike_risk'] > 0).astype(int)
        spike_col = 'long_run_spike_risk'
    else:
        spike_col = None
    
    if spike_col is not None:
        # PHASE 1: Add 14d and 60d windows (already have 30d)
        df['spike_frequency_14d'] = (
            grp[spike_col].transform(
                lambda s: spike_indicator.shift(1).rolling(14, min_periods=1).sum()
            )
        ).fillna(0).astype('int16')
        
        df['spike_frequency_30d'] = (
            grp[spike_col].transform(
                lambda s: spike_indicator.shift(1).rolling(30, min_periods=1).sum()
            )
        ).fillna(0).astype('int16')
        
        df['spike_frequency_60d'] = (
            grp[spike_col].transform(
                lambda s: spike_indicator.shift(1).rolling(60, min_periods=1).sum()
            )
        ).fillna(0).astype('int16')
    
    # NEW: Recovery pattern (alternating rest/load days)
    if 'km_total' in df.columns:
        rest_days = (df['km_total'] < 1.0).astype(int)
        df['recovery_pattern_score'] = (
            grp['km_total'].transform(
                lambda s: rest_days.shift(1).rolling(7, min_periods=7).mean()
            )
        ).astype('float32')
        df['recovery_pattern_score'] = df['recovery_pattern_score'].fillna(0.0).astype('float32')
    
    # NEW: Training monotony (variance in daily load over 7 days)
    if 'training_load' in df.columns:
        load_mean7 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).mean()
        )
        load_std7 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).std(ddof=0)
        )
        df['training_monotony_7d'] = (load_std7 / (load_mean7 + 1e-6)).fillna(0.0).astype('float32')
    
    # RECOMMENDATION 1: Add consecutive hard days features (count, max, mean over rolling windows)
    if 'consecutive_hard_days' in df.columns:
        consecutive_hard = pd.to_numeric(df['consecutive_hard_days'], errors='coerce').fillna(0).astype(int)
        
        # Rolling window features for consecutive hard days
        df['consecutive_hard_days_mean7'] = (
            grp['consecutive_hard_days'].transform(
                lambda s: consecutive_hard.shift(1).rolling(7, min_periods=1).mean()
            )
        ).fillna(0.0).astype('float32')
        
        df['consecutive_hard_days_max7'] = (
            grp['consecutive_hard_days'].transform(
                lambda s: consecutive_hard.shift(1).rolling(7, min_periods=1).max()
            )
        ).fillna(0).astype('int8')
        
        df['consecutive_hard_days_max14'] = (
            grp['consecutive_hard_days'].transform(
                lambda s: consecutive_hard.shift(1).rolling(14, min_periods=1).max()
            )
        ).fillna(0).astype('int8')
        
        df['consecutive_hard_days_max28'] = (
            grp['consecutive_hard_days'].transform(
                lambda s: consecutive_hard.shift(1).rolling(28, min_periods=1).max()
            )
        ).fillna(0).astype('int8')
        
        # Count of days with 2+ consecutive hard days in rolling windows
        has_2plus = (consecutive_hard >= 2).astype(int)
        df['consecutive_hard_2plus_count_7d'] = (
            grp['consecutive_hard_days'].transform(
                lambda s: has_2plus.shift(1).rolling(7, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        df['consecutive_hard_2plus_count_14d'] = (
            grp['consecutive_hard_days'].transform(
                lambda s: has_2plus.shift(1).rolling(14, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        df['consecutive_hard_2plus_count_28d'] = (
            grp['consecutive_hard_days'].transform(
                lambda s: has_2plus.shift(1).rolling(28, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        # Count of days with 3+ consecutive hard days in rolling windows
        has_3plus = (consecutive_hard >= 3).astype(int)
        df['consecutive_hard_3plus_count_7d'] = (
            grp['consecutive_hard_days'].transform(
                lambda s: has_3plus.shift(1).rolling(7, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        df['consecutive_hard_3plus_count_14d'] = (
            grp['consecutive_hard_days'].transform(
                lambda s: has_3plus.shift(1).rolling(14, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
    
    return df


def _add_enhanced_acwr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced ACWR features (trajectory, acceleration, duration above threshold)."""
    df = df.copy()
    df = df.sort_values(['user_id', 'date'])
    grp = df.groupby('user_id', sort=False)
    
    if 'acwr_clipped' in df.columns:
        # NEW: ACWR trajectory (increasing vs decreasing)
        acwr_mean7 = grp['acwr_clipped'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).mean()
        )
        acwr_mean14 = grp['acwr_clipped'].transform(
            lambda s: s.shift(1).rolling(14, min_periods=14).mean()
        )
        df['acwr_trajectory'] = (acwr_mean7 - acwr_mean14).astype('float32')  # Positive = increasing
        
        # NEW: ACWR acceleration (rate of change)
        acwr_lag1 = grp['acwr_clipped'].transform(lambda s: s.shift(1))
        acwr_lag7 = grp['acwr_clipped'].transform(lambda s: s.shift(7))
        df['acwr_acceleration'] = (acwr_lag1 - acwr_lag7).astype('float32')
        
        # NEW: ACWR above threshold duration (consecutive days above 1.2)
        high_acwr = (df['acwr_clipped'] > 1.2).astype(int)
        df['acwr_above_threshold_days'] = (
            grp['acwr_clipped'].transform(
                lambda s: high_acwr.groupby((high_acwr == 0).cumsum()).cumsum()
            )
        ).fillna(0).astype('int8')
        
        # NEW: ACWR × Load interaction (high ACWR + high absolute load)
        if 'training_load' in df.columns:
            load_mean7 = grp['training_load'].transform(
                lambda s: s.shift(1).rolling(7, min_periods=7).mean()
            )
            load_normalized = np.clip(load_mean7 / 500.0, 0, 2.0)
            acwr_excess = np.clip(df['acwr_clipped'] - 1.1, 0.0, 2.5)
            df['acwr_x_load7'] = (acwr_excess * load_normalized).astype('float32')
    
    return df


def _add_user_profile_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add user profile interactions (profile × load, profile × recovery, profile × spike)."""
    df = df.copy()
    df = df.sort_values(['user_id', 'date'])
    grp = df.groupby('user_id', sort=False)
    
    # Profile encoding: novice=0, recreational=1, advanced=2, elite=3
    if 'profile' in df.columns:
        profile_map = {'novice': 0, 'recreational': 1, 'advanced': 2, 'elite': 3}
        profile_encoded = df['profile'].map(profile_map).fillna(1).astype('int8')
    elif 'fitness' in df.columns:
        # Use fitness as proxy for profile
        fitness = pd.to_numeric(df['fitness'], errors='coerce').fillna(0.5)
        profile_encoded = (fitness * 3).astype('int8')  # 0-3 scale
    else:
        return df  # No profile information available
    
    # NEW: Profile × Training Load (novice vs elite handle load differently)
    if 'training_load' in df.columns:
        load_mean7 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).mean()
        )
        load_normalized = np.clip(load_mean7 / 500.0, 0, 2.0)
        # Novice (0) more sensitive to high load
        novice_sensitivity = (profile_encoded == 0).astype(float) * 1.5
        elite_sensitivity = (profile_encoded >= 2).astype(float) * 0.7
        sensitivity = 1.0 + novice_sensitivity - elite_sensitivity
        df['profile_x_load7'] = (profile_encoded * load_normalized * sensitivity).astype('float32')
    
    # NEW: Profile × Recovery (novice needs more recovery)
    if 'recovery_index' in df.columns:
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        poor_recovery = np.clip(-recovery, 0, 1)
        # Novice (0) more sensitive to poor recovery
        novice_sensitivity = (profile_encoded == 0).astype(float) * 1.3
        sensitivity = 1.0 + novice_sensitivity
        df['profile_x_poor_recovery'] = ((3 - profile_encoded) * poor_recovery * sensitivity).astype('float32')
    
    # NEW: Profile × Spike (novice more sensitive to spikes)
    if 'long_run_spike_risk' in df.columns:
        # Novice (0) more sensitive to spikes
        novice_sensitivity = (profile_encoded == 0).astype(float) * 1.4
        sensitivity = 1.0 + novice_sensitivity
        df['profile_x_spike'] = ((3 - profile_encoded) * df['long_run_spike_risk'] * sensitivity).astype('float32')
    
    # NEW: Fitness × Load trajectory (low fitness + increasing load)
    if 'fitness' in df.columns and 'training_load' in df.columns:
        fitness = pd.to_numeric(df['fitness'], errors='coerce').fillna(0.5)
        low_fitness = 1.0 - fitness
        load_mean7 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).mean()
        )
        load_mean14 = grp['training_load'].transform(
            lambda s: s.shift(1).rolling(14, min_periods=14).mean()
        )
        load_trajectory = load_mean7 - load_mean14  # Positive = increasing
        load_trajectory_positive = np.clip(load_trajectory / 100.0, 0, 2.0)  # Normalize
        df['fitness_x_load_trajectory'] = (low_fitness * load_trajectory_positive).astype('float32')
    
    return df


def _add_long_run_spike_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add GARMIN RUNSAFE long run spike features.
    
    Biggest single risk: increasing long run by >10% of max long run in previous 30 days.
    Categories from Garmin Runsafe study:
      - Baseline (0-10%): Reference
      - Small Spike (>10-30%): 64% higher risk
      - Moderate Spike (>30-100%): 52% higher risk
      - Large Spike (>100%): 128% higher risk
    
    Also adds interactions with proneness, fitness, and sleep.
    """
    df = df.copy()
    df = df.sort_values(['user_id', 'date'])
    
    # If long_run_spike_risk is already in the data, use it
    if 'long_run_spike_risk' in df.columns:
        df['long_run_spike_risk'] = pd.to_numeric(df['long_run_spike_risk'], errors='coerce').fillna(0.0).astype('float32')
        
        # NEW: Alternative spike feature encodings for better model learning
        spike_risk = df['long_run_spike_risk'].values
        # 1. Log-scale encoding (handles wide range of values)
        spike_risk_log = np.log1p(spike_risk)  # log(1+x) to handle zeros
        df['spike_risk_log'] = spike_risk_log.astype('float32')
        # 2. Squared encoding (emphasizes larger spikes)
        df['spike_risk_squared'] = (spike_risk ** 2).astype('float32')
        # 3. Square root encoding (compresses large values)
        df['spike_risk_sqrt'] = (np.sqrt(spike_risk)).astype('float32')
        # 4. Binned encoding (categorical bins)
        spike_bins = pd.cut(spike_risk, bins=[-0.01, 0.01, 0.5, 1.0, 2.0, 10.0], labels=[0, 1, 2, 3, 4])
        df['spike_risk_bin'] = spike_bins.astype('int8')
        df['spike_risk_bin'] = df['spike_risk_bin'].fillna(0).astype('int8')
        # 5. Binary indicators for different thresholds
        df['spike_any'] = (spike_risk > 0).astype('int8')
        df['spike_small'] = ((spike_risk > 0) & (spike_risk <= 0.5)).astype('int8')
        df['spike_medium'] = ((spike_risk > 0.5) & (spike_risk <= 1.0)).astype('int8')
        df['spike_large'] = (spike_risk > 1.0).astype('int8')
        # 6. Normalized by max (relative to user's max spike)
        grp = df.groupby('user_id', sort=False)
        max_spike = grp['long_run_spike_risk'].transform('max')
        df['spike_risk_normalized'] = (spike_risk / (max_spike + 1e-6)).astype('float32')
        
        # Add categorical features for each spike category
        if 'long_run_spike_category' in df.columns:
            df['long_run_spike_category'] = pd.to_numeric(df['long_run_spike_category'], errors='coerce').fillna(0).astype('int8')
            # Create binary indicators for each category
            df['long_run_spike_small'] = (df['long_run_spike_category'] == 1).astype('int8')
            df['long_run_spike_moderate'] = (df['long_run_spike_category'] == 2).astype('int8')
            df['long_run_spike_large'] = (df['long_run_spike_category'] == 3).astype('int8')
        else:
            # Create categories from risk values
            df['long_run_spike_small'] = ((df['long_run_spike_risk'] > 0.0) & (df['long_run_spike_risk'] <= 0.64)).astype('int8')
            df['long_run_spike_moderate'] = ((df['long_run_spike_risk'] > 0.64) & (df['long_run_spike_risk'] <= 1.0)).astype('int8')
            df['long_run_spike_large'] = (df['long_run_spike_risk'] > 1.0).astype('int8')
        
        # Add interactions: spike × proneness, × fitness, × sleep, × ACWR, × fatigue, × recovery
        # High proneness + spike = very high risk
        if 'injury_proneness' in df.columns:
            proneness = pd.to_numeric(df['injury_proneness'], errors='coerce').fillna(0.5)
            df['spike_x_proneness'] = (df['long_run_spike_risk'] * proneness).astype('float32')
            # Also for categories
            if 'long_run_spike_category' in df.columns:
                any_spike = (df['long_run_spike_category'] > 0).astype(float)
                df['any_spike_x_proneness'] = (any_spike * proneness).astype('float32')
        
        # Low fitness + spike = very high risk
        if 'fitness' in df.columns:
            fitness = pd.to_numeric(df['fitness'], errors='coerce').fillna(0.5)
            low_fitness = 1.0 - fitness  # Invert: low fitness = high value
            df['spike_x_low_fitness'] = (df['long_run_spike_risk'] * low_fitness).astype('float32')
            if 'long_run_spike_category' in df.columns:
                any_spike = (df['long_run_spike_category'] > 0).astype(float)
                df['any_spike_x_low_fitness'] = (any_spike * low_fitness).astype('float32')
        
        # Poor sleep + spike = very high risk
        if 'sleep_def' in df.columns:
            df['spike_x_sleep_def'] = (df['long_run_spike_risk'] * df['sleep_def']).astype('float32')
        elif 'sleep_hours_z7_28' in df.columns:
            sleep_z = pd.to_numeric(df['sleep_hours_z7_28'], errors='coerce').fillna(0.0)
            sleep_def = np.clip(-sleep_z, 0.0, 3.0)
            df['spike_x_sleep_def'] = (df['long_run_spike_risk'] * sleep_def).astype('float32')
        
        # NEW: Spike × ACWR (high ACWR + spike = very high risk)
        if 'acwr_excess' in df.columns:
            df['spike_x_acwr_excess'] = (df['long_run_spike_risk'] * df['acwr_excess']).astype('float32')
        elif 'acwr_clipped' in df.columns:
            acwr_excess = np.clip(df['acwr_clipped'] - 1.1, 0.0, 2.5)
            df['spike_x_acwr_excess'] = (df['long_run_spike_risk'] * acwr_excess).astype('float32')
        
        # NEW: Spike × Fatigue State (high fatigue + spike = very high risk)
        if 'fatigue_state' in df.columns:
            df['spike_x_fatigue_state'] = (df['long_run_spike_risk'] * df['fatigue_state']).astype('float32')
        
        # NEW: Spike × Recovery (poor recovery + spike = very high risk)
        if 'recovery_index' in df.columns:
            recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
            poor_recovery = np.clip(-recovery, 0, 1)  # Poor recovery = high value
            df['spike_x_poor_recovery'] = (df['long_run_spike_risk'] * poor_recovery).astype('float32')
        elif 'sleep_hours' in df.columns and 'hrv_ms' in df.columns:
            # Compute recovery index if not available
            sleep_norm = np.clip((df['sleep_hours'] - 6.0) / 2.0, -1, 1)
            hrv_norm = np.clip((df['hrv_ms'] - 50.0) / 20.0, -1, 1)
            recovery_index = sleep_norm * hrv_norm
            poor_recovery = np.clip(-recovery_index, 0, 1)
            df['spike_x_poor_recovery'] = (df['long_run_spike_risk'] * poor_recovery).astype('float32')
        
        # Combined: spike × proneness × low fitness (very high risk)
        if 'injury_proneness' in df.columns and 'fitness' in df.columns:
            proneness = pd.to_numeric(df['injury_proneness'], errors='coerce').fillna(0.5)
            fitness = pd.to_numeric(df['fitness'], errors='coerce').fillna(0.5)
            low_fitness = 1.0 - fitness
            df['spike_x_proneness_x_low_fitness'] = (df['long_run_spike_risk'] * proneness * low_fitness).astype('float32')
        
        # NEW: Polynomial encodings for spike risk (captures non-linear relationships)
        # 2nd order polynomial
        df['spike_risk_poly2'] = (spike_risk ** 2).astype('float32')
        # 3rd order polynomial
        df['spike_risk_poly3'] = (spike_risk ** 3).astype('float32')
        # Interaction with itself (squared interaction)
        df['spike_risk_self_interaction'] = (spike_risk * spike_risk).astype('float32')
        
        # NEW: Interaction-based encodings (combine spike with multiple factors)
        if 'acwr_clipped' in df.columns and 'fatigue_state' in df.columns:
            acwr = pd.to_numeric(df['acwr_clipped'], errors='coerce').fillna(1.0)
            fatigue = pd.to_numeric(df['fatigue_state'], errors='coerce').fillna(0.0)
            # Triple interaction: spike × ACWR × fatigue
            df['spike_x_acwr_x_fatigue'] = (spike_risk * acwr * fatigue).astype('float32')
        
        if 'injury_proneness' in df.columns and 'fitness' in df.columns and 'recovery_index' in df.columns:
            proneness = pd.to_numeric(df['injury_proneness'], errors='coerce').fillna(0.5)
            fitness = pd.to_numeric(df['fitness'], errors='coerce').fillna(0.5)
            recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
            poor_recovery = np.clip(-recovery, 0, 1)
            low_fitness = 1.0 - fitness
            # Triple interaction: spike × proneness × low fitness × poor recovery
            df['spike_x_proneness_x_low_fitness_x_poor_recovery'] = (spike_risk * proneness * low_fitness * poor_recovery).astype('float32')
        
        return df
    
    # Otherwise, calculate it from daily data
    grp = df.groupby('user_id', sort=False)
    
    long_run_spike = []
    for uid, group in grp:
        group = group.sort_values('date').reset_index(drop=True)
        group_dates = pd.to_datetime(group['date']).values
        group_session = group['session_type'].values
        group_km = group['km_total'].fillna(0).values
        
        group_spike = np.zeros(len(group), dtype=float)
        for i, date in enumerate(group_dates):
            # Current day's long run distance (if it's a long run day)
            if i < len(group) and str(group_session[i]) == "long" and group_km[i] > 0:
                current_long_km = float(group_km[i])
                
                # Max long run in previous 30 days (excluding today)
                prev_indices = [j for j in range(i) if 
                               (date - pd.to_datetime(group_dates[j])).days <= 30 and
                               (date - pd.to_datetime(group_dates[j])).days > 0 and
                               str(group_session[j]) == "long" and
                               group_km[j] > 0]
                
                if len(prev_indices) > 0:
                    prev_long_distances = [float(group_km[j]) for j in prev_indices]
                    max_prev_long = max(prev_long_distances)
                    # Risk if current > 1.1 * max previous
                    if max_prev_long > 0:
                        excess_ratio = (current_long_km / max_prev_long) - 1.1
                        group_spike[i] = np.clip(excess_ratio, 0.0, 2.0)
        
        long_run_spike.extend(group_spike)
    
    df['long_run_spike_risk'] = pd.Series(long_run_spike, index=df.index).astype('float32')
    return df


def _add_resilience_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add injury resilience interaction features.
    
    Low resilience users are more sensitive to:
    - High ACWR
    - High fatigue state
    - Poor recovery
    - Sleep deficit
    
    Note: Resilience is inverted (low resilience = high risk)
    """
    df = df.copy()
    
    if 'injury_resilience' not in df.columns:
        return df
    
    resilience = pd.to_numeric(df['injury_resilience'], errors='coerce').fillna(0.5)
    low_resilience = 1.0 - resilience  # Invert: low resilience = high risk
    
    # Low resilience × ACWR (low resilience + high ACWR = very high risk)
    if 'acwr_excess' in df.columns:
        df['low_resilience_x_acwr_excess'] = (low_resilience * df['acwr_excess']).astype('float32')
    
    # Low resilience × fatigue state (low resilience + high fatigue = very high risk)
    if 'fatigue_state' in df.columns:
        df['low_resilience_x_fatigue'] = (low_resilience * df['fatigue_state']).astype('float32')
    
    # Low resilience × poor recovery (low resilience + poor recovery = very high risk)
    if 'recovery_index' in df.columns:
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        poor_recovery = np.clip(-recovery, 0, 1)  # Poor recovery = high value
        df['low_resilience_x_poor_recovery'] = (low_resilience * poor_recovery).astype('float32')
    
    # Low resilience × sleep deficit (low resilience + sleep deficit = very high risk)
    if 'sleep_def' in df.columns:
        df['low_resilience_x_sleep_def'] = (low_resilience * df['sleep_def']).astype('float32')
    
    # Proneness × resilience (high proneness + low resilience = very high risk)
    if 'injury_proneness' in df.columns:
        proneness = pd.to_numeric(df['injury_proneness'], errors='coerce').fillna(0.5)
        df['proneness_x_low_resilience'] = (proneness * low_resilience).astype('float32')
    
    return df


def _add_temporal_user_spike_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and user-level interactions with spike and post-injury signals.
    
    This helps the model learn that spikes/post-injury interact with:
    - Temporal patterns (week_of_year, month, day_of_week)
    - User characteristics (proneness, fitness, profile)
    """
    df = df.copy()
    
    # Temporal × Spike interactions
    if 'long_run_spike_risk' in df.columns:
        spike_risk = pd.to_numeric(df['long_run_spike_risk'], errors='coerce').fillna(0.0)
        
        # Spike × Week of Year (seasonal patterns)
        if 'week_of_year' in df.columns:
            week = pd.to_numeric(df['week_of_year'], errors='coerce').fillna(26)
            # Normalize week to 0-1 scale
            week_norm = (week - 1) / 51.0  # 1-52 -> 0-1
            df['spike_x_week_of_year'] = (spike_risk * week_norm).astype('float32')
        
        # Spike × Month (seasonal patterns)
        if 'month' in df.columns:
            month = pd.to_numeric(df['month'], errors='coerce').fillna(6)
            month_norm = (month - 1) / 11.0  # 1-12 -> 0-1
            df['spike_x_month'] = (spike_risk * month_norm).astype('float32')
        
        # Spike × Day of Week (weekend vs weekday)
        if 'day_of_week' in df.columns:
            dow = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(3)
            # Weekend = higher risk (Sat=5, Sun=6)
            is_weekend = ((dow >= 5).astype(float))
            df['spike_x_weekend'] = (spike_risk * is_weekend).astype('float32')
    
    # Temporal × Post-Injury interactions
    if 'days_since_injury' in df.columns:
        days_since = pd.to_numeric(df['days_since_injury'], errors='coerce').fillna(999)
        # Create post-injury flag (within 45 days)
        is_post_injury = ((days_since < 45) & (days_since < 999)).astype(float)
        
        # NEW: Alternative post-injury encodings for better model learning
        # 1. Acute phase (0-7 days) - highest risk
        is_acute = ((days_since < 7) & (days_since < 999)).astype('int8')
        df['post_injury_acute'] = is_acute
        
        # 2. Decay phase (7-45 days) - decaying risk
        is_decay = ((days_since >= 7) & (days_since < 45) & (days_since < 999)).astype('int8')
        df['post_injury_decay'] = is_decay
        
        # 3. Inverse days (higher for more recent injuries)
        days_since_clipped = np.clip(days_since, 0, 45)
        days_inverse = np.where(days_since < 999, 45.0 - days_since_clipped, 0.0)
        df['post_injury_days_inverse'] = (days_inverse / 45.0).astype('float32')  # Normalized 0-1
        
        # 4. Exponential decay encoding (emphasizes recent injuries)
        days_since_clipped = np.clip(days_since, 0, 45)
        decay_factor = np.where(days_since < 999, np.exp(-days_since_clipped / 15.0), 0.0)
        df['post_injury_decay_exp'] = decay_factor.astype('float32')
        
        # 5. Binned encoding
        days_bins = pd.cut(days_since, bins=[-1, 0, 7, 14, 30, 45, 999], labels=[0, 1, 2, 3, 4, 5])
        df['post_injury_bin'] = days_bins.astype('int8')
        df['post_injury_bin'] = df['post_injury_bin'].fillna(5).astype('int8')
        
        # Post-Injury × Week of Year
        if 'week_of_year' in df.columns:
            week = pd.to_numeric(df['week_of_year'], errors='coerce').fillna(26)
            week_norm = (week - 1) / 51.0
            df['post_injury_x_week_of_year'] = (is_post_injury * week_norm).astype('float32')
        
        # Post-Injury × Month
        if 'month' in df.columns:
            month = pd.to_numeric(df['month'], errors='coerce').fillna(6)
            month_norm = (month - 1) / 11.0
            df['post_injury_x_month'] = (is_post_injury * month_norm).astype('float32')
    
    # User × Spike interactions (beyond existing proneness/fitness)
    if 'long_run_spike_risk' in df.columns:
        spike_risk = pd.to_numeric(df['long_run_spike_risk'], errors='coerce').fillna(0.0)
        
        # Spike × Profile (if available)
        if 'profile' in df.columns:
            profile_map = {'novice': 0, 'recreational': 1, 'advanced': 2, 'elite': 3}
            profile_encoded = df['profile'].map(profile_map).fillna(1).astype('float32')
            # Novice (0) more sensitive to spikes
            novice_sensitivity = ((profile_encoded == 0).astype(float) * 1.5)
            df['spike_x_novice'] = (spike_risk * novice_sensitivity).astype('float32')
        
        # Spike × Resilience (if available)
        if 'injury_resilience' in df.columns:
            resilience = pd.to_numeric(df['injury_resilience'], errors='coerce').fillna(0.5)
            low_resilience = 1.0 - resilience
            df['spike_x_low_resilience'] = (spike_risk * low_resilience).astype('float32')
    
    # User × Post-Injury interactions
    if 'days_since_injury' in df.columns:
        days_since = pd.to_numeric(df['days_since_injury'], errors='coerce').fillna(999)
        is_post_injury = ((days_since < 45) & (days_since < 999)).astype(float)
        
        # Post-Injury × Proneness
        if 'injury_proneness' in df.columns:
            proneness = pd.to_numeric(df['injury_proneness'], errors='coerce').fillna(0.5)
            df['post_injury_x_proneness'] = (is_post_injury * proneness).astype('float32')
        
        # Post-Injury × Low Fitness
        if 'fitness' in df.columns:
            fitness = pd.to_numeric(df['fitness'], errors='coerce').fillna(0.5)
            low_fitness = 1.0 - fitness
            df['post_injury_x_low_fitness'] = (is_post_injury * low_fitness).astype('float32')
        
        # Post-Injury × Resilience
        if 'injury_resilience' in df.columns:
            resilience = pd.to_numeric(df['injury_resilience'], errors='coerce').fillna(0.5)
            low_resilience = 1.0 - resilience
            df['post_injury_x_low_resilience'] = (is_post_injury * low_resilience).astype('float32')
    
        # Combined: Spike × Post-Injury (very high risk)
        if 'long_run_spike_risk' in df.columns and 'days_since_injury' in df.columns:
            spike_risk = pd.to_numeric(df['long_run_spike_risk'], errors='coerce').fillna(0.0)
            days_since = pd.to_numeric(df['days_since_injury'], errors='coerce').fillna(999)
            is_post_injury = ((days_since < 45) & (days_since < 999)).astype(float)
            has_spike = (spike_risk > 0).astype(float)
            df['spike_x_post_injury'] = (has_spike * is_post_injury).astype('float32')
        
        # PHASE 1: Post-Injury × ACWR (re-injury risk is much higher with high ACWR)
        if 'days_since_injury' in df.columns and 'acwr_excess' in df.columns:
            days_since = pd.to_numeric(df['days_since_injury'], errors='coerce').fillna(999)
            is_post_injury = ((days_since < 45) & (days_since < 999)).astype(float)
            acwr_excess = pd.to_numeric(df['acwr_excess'], errors='coerce').fillna(0.0)
            # Very high risk: post-injury + high ACWR
            df['post_injury_x_acwr_excess'] = (is_post_injury * acwr_excess).astype('float32')
        
        # PHASE 1: Post-Injury × Recovery (re-injury risk higher with poor recovery)
        if 'days_since_injury' in df.columns and 'recovery_index' in df.columns:
            days_since = pd.to_numeric(df['days_since_injury'], errors='coerce').fillna(999)
            is_post_injury = ((days_since < 45) & (days_since < 999)).astype(float)
            recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
            poor_recovery = np.clip(-recovery, 0, 1)
            df['post_injury_x_poor_recovery'] = (is_post_injury * poor_recovery).astype('float32')
        
        # PHASE 1: Triple interaction: Post-Injury × Spike × ACWR (extremely high risk)
        if ('days_since_injury' in df.columns and 'long_run_spike_risk' in df.columns and 
            'acwr_excess' in df.columns):
            days_since = pd.to_numeric(df['days_since_injury'], errors='coerce').fillna(999)
            is_post_injury = ((days_since < 45) & (days_since < 999)).astype(float)
            spike_risk = pd.to_numeric(df['long_run_spike_risk'], errors='coerce').fillna(0.0)
            has_spike = (spike_risk > 0).astype(float)
            acwr_excess = pd.to_numeric(df['acwr_excess'], errors='coerce').fillna(0.0)
            df['post_injury_x_spike_x_acwr'] = (is_post_injury * has_spike * acwr_excess).astype('float32')
        
        return df


def _add_consecutive_hard_and_intensity_features(df: pd.DataFrame, activities: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add features for consecutive hard days, high intensity sessions per week, and double sessions per week.
    
    RECOMMENDATION 1: Feature Engineering
    - Consecutive hard days (count, max, mean over rolling windows) - already in _add_sequence_features
    - High intensity sessions per week (rolling 7-day, 14-day, 28-day counts)
    - Double sessions per week (rolling 7-day, 14-day, 28-day counts)
    - Interaction features: consecutive_hard_days × fitness, high_intensity_week × stress, etc.
    """
    df = df.copy()
    df = df.sort_values(['user_id', 'date'])
    grp = df.groupby('user_id', sort=False)
    
    # RECOMMENDATION 1: High intensity sessions per week (rolling 7-day, 14-day, 28-day counts)
    # High intensity: interval, tempo, or Z5 >= 2km
    if 'session_type' in df.columns:
        # Identify hard sessions (tempo, interval) from daily data
        is_hard_session = df['session_type'].isin(['tempo', 'interval']).astype(int)
        
        # Count hard sessions in rolling windows (lagged)
        df['hard_sessions_count_7d'] = (
            grp['session_type'].transform(
                lambda s: is_hard_session.shift(1).rolling(7, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        df['hard_sessions_count_14d'] = (
            grp['session_type'].transform(
                lambda s: is_hard_session.shift(1).rolling(14, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        df['hard_sessions_count_28d'] = (
            grp['session_type'].transform(
                lambda s: is_hard_session.shift(1).rolling(28, min_periods=1).sum()
            )
        ).fillna(0).astype('int16')
        
        # Categorical features for high intensity weeks (3+, 4+, 5+ sessions per week)
        df['has_3plus_hard_sessions_7d'] = (df['hard_sessions_count_7d'] >= 3).astype('int8')
        df['has_4plus_hard_sessions_7d'] = (df['hard_sessions_count_7d'] >= 4).astype('int8')
        df['has_5plus_hard_sessions_7d'] = (df['hard_sessions_count_7d'] >= 5).astype('int8')
    
    # Also use activities data if available for more accurate high intensity count (includes Z5 >= 2km)
    if activities is not None and len(activities) > 0:
        a = activities.copy()
        a['date'] = pd.to_datetime(a['date'])
        a['user_id'] = a['user_id'].astype('int64')
        
        # High intensity: interval, tempo, or Z5 >= 2km
        if 'session_type' in a.columns and 'kms_z5_t1_t2' in a.columns:
            a['is_high_intensity'] = (
                (a['session_type'].isin(['interval', 'tempo'])) |
                (pd.to_numeric(a['kms_z5_t1_t2'], errors='coerce').fillna(0) >= 2.0)
            ).astype(int)
            
            # Count high intensity sessions per week (aggregate to daily first)
            hi_per_day = a.groupby(['user_id', 'date'])['is_high_intensity'].sum().reset_index(name='hi_sessions_today')
            
            # Merge with daily data
            df['date_dt'] = pd.to_datetime(df['date'])
            df['user_id_int'] = df['user_id'].astype('int64')
            hi_per_day['date_dt'] = pd.to_datetime(hi_per_day['date'])
            hi_per_day['user_id_int'] = hi_per_day['user_id'].astype('int64')
            
            df = df.merge(hi_per_day[['user_id_int', 'date_dt', 'hi_sessions_today']], 
                         on=['user_id_int', 'date_dt'], how='left')
            df['hi_sessions_today'] = df['hi_sessions_today'].fillna(0).astype(int)
            
            # Calculate rolling counts (lagged) - need to re-sort and re-group after merge
            df = df.sort_values(['user_id', 'date'])
            grp = df.groupby('user_id', sort=False)
            
            df['high_intensity_sessions_7d'] = (
                grp['hi_sessions_today'].transform(
                    lambda s: s.shift(1).rolling(7, min_periods=1).sum()
                )
            ).fillna(0).astype('int8')
            
            df['high_intensity_sessions_14d'] = (
                grp['hi_sessions_today'].transform(
                    lambda s: s.shift(1).rolling(14, min_periods=1).sum()
                )
            ).fillna(0).astype('int16')
            
            df['high_intensity_sessions_28d'] = (
                grp['hi_sessions_today'].transform(
                    lambda s: s.shift(1).rolling(28, min_periods=1).sum()
                )
            ).fillna(0).astype('int16')
            
            # Categorical features for high intensity weeks
            df['has_3plus_hi_sessions_7d'] = (df['high_intensity_sessions_7d'] >= 3).astype('int8')
            df['has_4plus_hi_sessions_7d'] = (df['high_intensity_sessions_7d'] >= 4).astype('int8')
            df['has_5plus_hi_sessions_7d'] = (df['high_intensity_sessions_7d'] >= 5).astype('int8')
            
            # Clean up temporary columns
            df = df.drop(columns=['date_dt', 'user_id_int', 'hi_sessions_today'], errors='ignore')
        else:
            # Fallback: use hard_sessions_count if activities doesn't have required columns
            if 'hard_sessions_count_7d' in df.columns:
                df['high_intensity_sessions_7d'] = df['hard_sessions_count_7d']
                df['high_intensity_sessions_14d'] = df['hard_sessions_count_14d']
                df['high_intensity_sessions_28d'] = df['hard_sessions_count_28d']
                df['has_3plus_hi_sessions_7d'] = df['has_3plus_hard_sessions_7d']
                df['has_4plus_hi_sessions_7d'] = df['has_4plus_hard_sessions_7d']
                df['has_5plus_hi_sessions_7d'] = df['has_5plus_hard_sessions_7d']
    else:
        # Fallback: use hard_sessions_count if no activities data
        if 'hard_sessions_count_7d' in df.columns:
            df['high_intensity_sessions_7d'] = df['hard_sessions_count_7d']
            df['high_intensity_sessions_14d'] = df['hard_sessions_count_14d']
            df['high_intensity_sessions_28d'] = df['hard_sessions_count_28d']
            df['has_3plus_hi_sessions_7d'] = df['has_3plus_hard_sessions_7d']
            df['has_4plus_hi_sessions_7d'] = df['has_4plus_hard_sessions_7d']
            df['has_5plus_hi_sessions_7d'] = df['has_5plus_hard_sessions_7d']
    
    # RECOMMENDATION 1: Double sessions per week (rolling 7-day, 14-day, 28-day counts)
    if 'has_double' in df.columns:
        has_double = pd.to_numeric(df['has_double'], errors='coerce').fillna(0).astype(int)
        
        # Count double session days in rolling windows (lagged)
        df['double_sessions_count_7d'] = (
            grp['has_double'].transform(
                lambda s: has_double.shift(1).rolling(7, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        df['double_sessions_count_14d'] = (
            grp['has_double'].transform(
                lambda s: has_double.shift(1).rolling(14, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        
        df['double_sessions_count_28d'] = (
            grp['has_double'].transform(
                lambda s: has_double.shift(1).rolling(28, min_periods=1).sum()
            )
        ).fillna(0).astype('int16')
        
        # Categorical features for double session weeks (2+, 3+, 4+ double days per week)
        df['has_2plus_double_sessions_7d'] = (df['double_sessions_count_7d'] >= 2).astype('int8')
        df['has_3plus_double_sessions_7d'] = (df['double_sessions_count_7d'] >= 3).astype('int8')
        df['has_4plus_double_sessions_7d'] = (df['double_sessions_count_7d'] >= 4).astype('int8')
    else:
        # If has_double column doesn't exist, set all to 0
        df['double_sessions_count_7d'] = 0
        df['double_sessions_count_14d'] = 0
        df['double_sessions_count_28d'] = 0
        df['has_2plus_double_sessions_7d'] = 0
        df['has_3plus_double_sessions_7d'] = 0
        df['has_4plus_double_sessions_7d'] = 0
        df['double_sessions_count_7d'] = df['double_sessions_count_7d'].astype('int8')
        df['double_sessions_count_14d'] = df['double_sessions_count_14d'].astype('int8')
        df['double_sessions_count_28d'] = df['double_sessions_count_28d'].astype('int16')
        df['has_2plus_double_sessions_7d'] = df['has_2plus_double_sessions_7d'].astype('int8')
        df['has_3plus_double_sessions_7d'] = df['has_3plus_double_sessions_7d'].astype('int8')
        df['has_4plus_double_sessions_7d'] = df['has_4plus_double_sessions_7d'].astype('int8')
    
    # RECOMMENDATION 1: Interaction features
    # Consecutive hard days × fitness
    if 'consecutive_hard_days' in df.columns and 'fitness' in df.columns:
        consecutive_hard = pd.to_numeric(df['consecutive_hard_days'], errors='coerce').fillna(0)
        fitness = pd.to_numeric(df['fitness'], errors='coerce').fillna(0.5)
        df['consecutive_hard_x_fitness'] = (consecutive_hard * fitness).astype('float32')
        df['consecutive_hard_x_low_fitness'] = (consecutive_hard * (1.0 - fitness)).astype('float32')
    
    # High intensity week × stress
    if 'high_intensity_sessions_7d' in df.columns and 'stress_score' in df.columns:
        hi_sessions = pd.to_numeric(df['high_intensity_sessions_7d'], errors='coerce').fillna(0)
        stress = pd.to_numeric(df['stress_score'], errors='coerce').fillna(0.0)
        df['high_intensity_week_x_stress'] = (hi_sessions * stress).astype('float32')
        
        # Also use stress z-score for better scaling
        if 'stress_score_z7_28' in df.columns:
            stress_z = pd.to_numeric(df['stress_score_z7_28'], errors='coerce').fillna(0.0)
            df['high_intensity_week_x_stress_z'] = (hi_sessions * stress_z).astype('float32')
    
    # High intensity week × ACWR
    if 'high_intensity_sessions_7d' in df.columns and 'acwr_clipped' in df.columns:
        hi_sessions = pd.to_numeric(df['high_intensity_sessions_7d'], errors='coerce').fillna(0)
        acwr = pd.to_numeric(df['acwr_clipped'], errors='coerce').fillna(1.0)
        df['high_intensity_week_x_acwr'] = (hi_sessions * acwr).astype('float32')
        
        if 'acwr_excess' in df.columns:
            acwr_excess = pd.to_numeric(df['acwr_excess'], errors='coerce').fillna(0.0)
            df['high_intensity_week_x_acwr_excess'] = (hi_sessions * acwr_excess).astype('float32')
    
    # Consecutive hard days × high intensity week (stacked stress)
    if 'consecutive_hard_days' in df.columns and 'high_intensity_sessions_7d' in df.columns:
        consecutive_hard = pd.to_numeric(df['consecutive_hard_days'], errors='coerce').fillna(0)
        hi_sessions = pd.to_numeric(df['high_intensity_sessions_7d'], errors='coerce').fillna(0)
        df['consecutive_hard_x_hi_week'] = (consecutive_hard * hi_sessions).astype('float32')
    
    # Double sessions × hard sessions (high load days)
    if 'double_sessions_count_7d' in df.columns and 'hard_sessions_count_7d' in df.columns:
        double_sessions = pd.to_numeric(df['double_sessions_count_7d'], errors='coerce').fillna(0)
        hard_sessions = pd.to_numeric(df['hard_sessions_count_7d'], errors='coerce').fillna(0)
        df['double_sessions_x_hard_sessions'] = (double_sessions * hard_sessions).astype('float32')
    
    # High intensity week × recovery (poor recovery + high intensity = dangerous)
    if 'high_intensity_sessions_7d' in df.columns and 'recovery_index' in df.columns:
        hi_sessions = pd.to_numeric(df['high_intensity_sessions_7d'], errors='coerce').fillna(0)
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        poor_recovery = np.clip(-recovery, 0, 1)
        df['high_intensity_week_x_poor_recovery'] = (hi_sessions * poor_recovery).astype('float32')
    
    # Consecutive hard days × recovery (poor recovery + consecutive hard days = dangerous)
    if 'consecutive_hard_days' in df.columns and 'recovery_index' in df.columns:
        consecutive_hard = pd.to_numeric(df['consecutive_hard_days'], errors='coerce').fillna(0)
        recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
        poor_recovery = np.clip(-recovery, 0, 1)
        df['consecutive_hard_x_poor_recovery'] = (consecutive_hard * poor_recovery).astype('float32')
    
    return df


def _add_phase2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Phase 2 features: Enhanced recovery, sprinting, and intensity zone features.
    
    Phase 2 Actions:
    1. Action 2.1: Injury Recovery Features (HIGH IMPACT)
    2. Action 2.2: Sprinting Features Enhancement (MEDIUM IMPACT)
    3. Action 2.3: Intensity Zone Features (MEDIUM IMPACT)
    """
    df = df.copy()
    df = df.sort_values(['user_id', 'date'])
    grp = df.groupby('user_id', sort=False)
    
    # ============================================================================
    # Action 2.1: Injury Recovery Features (HIGH IMPACT)
    # ============================================================================
    
    # is_in_recovery_period: Binary flag for being in gradual return period
    if 'days_since_recovery_end' in df.columns and 'return_period_duration' in df.columns:
        days_since = pd.to_numeric(df['days_since_recovery_end'], errors='coerce').fillna(999)
        return_period = pd.to_numeric(df['return_period_duration'], errors='coerce').fillna(0)
        # In recovery if days_since_recovery_end < return_period_duration
        df['is_in_recovery_period'] = (
            (days_since < return_period) & (days_since < 999) & (return_period > 0)
        ).astype('int8')
        
        # recovery_period_progress: 0-1 scale (how far through recovery)
        # Progress = 1 - (days_since_recovery_end / return_period_duration)
        # When days_since_recovery_end = 0, progress = 1 (just started recovery)
        # When days_since_recovery_end = return_period_duration, progress = 0 (recovery complete)
        recovery_progress = np.where(
            (return_period > 0) & (days_since < 999),
            np.clip(1.0 - (days_since / (return_period + 1e-6)), 0.0, 1.0),
            0.0
        )
        df['recovery_period_progress'] = recovery_progress.astype('float32')
        
        # load_during_recovery: Training load during recovery period
        if 'training_load' in df.columns:
            training_load = pd.to_numeric(df['training_load'], errors='coerce').fillna(0.0)
            is_in_recovery = df['is_in_recovery_period'].astype(bool)
            df['load_during_recovery'] = (training_load * is_in_recovery).astype('float32')
        
        # acwr_during_recovery: ACWR during recovery period
        if 'acwr_clipped' in df.columns:
            acwr = pd.to_numeric(df['acwr_clipped'], errors='coerce').fillna(1.0)
            is_in_recovery = df['is_in_recovery_period'].astype(bool)
            df['acwr_during_recovery'] = (acwr * is_in_recovery).astype('float32')
    
    # ============================================================================
    # Action 2.2: Sprinting Features Enhancement (MEDIUM IMPACT)
    # ============================================================================
    
    # NEW: Add spike_absolute_risk as a feature (the additive risk value from data generation)
    # This is the direct additive risk that affects injury probability
    # Model will learn: when spike_absolute_risk > 0, predict much higher risk
    # This matches how sprinting_absolute_risk is used as a feature
    if 'spike_absolute_risk' in df.columns:
        df['spike_absolute_risk'] = pd.to_numeric(df['spike_absolute_risk'], errors='coerce').fillna(0.0).astype('float32')
        # Add binary indicator for easier learning
        df['has_spike_abs_risk'] = (df['spike_absolute_risk'] > 0).astype('int8')
        # Add log-scale encoding (handles wide range of values)
        df['spike_abs_risk_log'] = np.log1p(df['spike_absolute_risk']).astype('float32')
    
    # NEW: Add sprinting_absolute_risk as a feature (the additive risk value from data generation)
    # This is the direct additive risk that affects injury probability
    # Model will learn: when sprinting_absolute_risk > 0, predict much higher risk
    # This matches how spike_absolute_risk is used as a feature
    if 'sprinting_absolute_risk' in df.columns:
        df['sprinting_absolute_risk'] = pd.to_numeric(df['sprinting_absolute_risk'], errors='coerce').fillna(0.0).astype('float32')
        # Add binary indicator for easier learning
        df['has_sprinting_abs_risk'] = (df['sprinting_absolute_risk'] > 0).astype('int8')
        # Add log-scale encoding (handles wide range of values)
        df['sprinting_abs_risk_log'] = np.log1p(df['sprinting_absolute_risk']).astype('float32')
    
    if 'kms_sprinting' in df.columns:
        sprinting = pd.to_numeric(df['kms_sprinting'], errors='coerce').fillna(0.0)
        
        # sprinting_last_7d: Sprinting volume in last 7 days
        df['sprinting_last_7d'] = (
            grp['kms_sprinting'].transform(
                lambda s: pd.to_numeric(s, errors='coerce').fillna(0.0).shift(1).rolling(7, min_periods=1).sum()
            )
        ).astype('float32')
        
        # sprinting_last_14d: Sprinting volume in last 14 days
        df['sprinting_last_14d'] = (
            grp['kms_sprinting'].transform(
                lambda s: pd.to_numeric(s, errors='coerce').fillna(0.0).shift(1).rolling(14, min_periods=1).sum()
            )
        ).astype('float32')
        
        # sprinting_spike_7d: Spike indicator (current sprinting > 1.5x 7d mean)
        sprinting_mean7 = (
            grp['kms_sprinting'].transform(
                lambda s: pd.to_numeric(s, errors='coerce').fillna(0.0).shift(1).rolling(7, min_periods=7).mean()
            )
        )
        df['sprinting_spike_7d'] = (
            (sprinting > 1.5 * sprinting_mean7) & (sprinting_mean7 > 0)
        ).astype('int8')
        
        # sprinting_share_7d: Sprinting as % of total km (last 7 days)
        if 'km_total' in df.columns:
            km_total_7d = (
                grp['km_total'].transform(
                    lambda s: pd.to_numeric(s, errors='coerce').fillna(0.0).shift(1).rolling(7, min_periods=1).sum()
                )
            )
            sprinting_share = np.where(
                km_total_7d > 0,
                np.clip(df['sprinting_last_7d'] / km_total_7d, 0.0, 1.0),
                0.0
            )
            df['sprinting_share_7d'] = sprinting_share.astype('float32')
        
        # sprinting_x_acwr: Sprinting + ACWR interaction
        if 'acwr_clipped' in df.columns:
            acwr = pd.to_numeric(df['acwr_clipped'], errors='coerce').fillna(1.0)
            sprinting_norm = np.clip(df['sprinting_last_7d'] / 10.0, 0.0, 3.0)  # Normalize sprinting
            df['sprinting_x_acwr'] = (sprinting_norm * acwr).astype('float32')
        
        # sprinting_x_fatigue: Sprinting + fatigue interaction
        if 'fatigue_state' in df.columns:
            fatigue = pd.to_numeric(df['fatigue_state'], errors='coerce').fillna(0.0)
            sprinting_norm = np.clip(df['sprinting_last_7d'] / 10.0, 0.0, 3.0)
            df['sprinting_x_fatigue'] = (sprinting_norm * fatigue).astype('float32')
        elif 'perceived_exertion' in df.columns:
            # Use perceived exertion as proxy for fatigue
            exertion = pd.to_numeric(df['perceived_exertion'], errors='coerce').fillna(0.0)
            fatigue_proxy = np.clip((exertion - 5.0) / 5.0, 0.0, 1.0)  # Normalize to 0-1
            sprinting_norm = np.clip(df['sprinting_last_7d'] / 10.0, 0.0, 3.0)
            df['sprinting_x_fatigue'] = (sprinting_norm * fatigue_proxy).astype('float32')
    
    # ============================================================================
    # Action 2.3: Intensity Zone Features (MEDIUM IMPACT)
    # ============================================================================
    
    # Calculate total intensity km (Z3-4 + Z5-T1-T2 + sprinting)
    intensity_cols = []
    if 'kms_z3_4' in df.columns:
        intensity_cols.append('kms_z3_4')
    if 'kms_z5_t1_t2' in df.columns:
        intensity_cols.append('kms_z5_t1_t2')
    if 'kms_sprinting' in df.columns:
        intensity_cols.append('kms_sprinting')
    
    if intensity_cols and 'km_total' in df.columns:
        # Total intensity km per day
        intensity_km = pd.Series(0.0, index=df.index)
        for col in intensity_cols:
            intensity_km += pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df['intensity_km_total'] = intensity_km.astype('float32')
        
        # intensity_share_7d: Intensity zones as % of total km (last 7 days)
        intensity_km_7d = (
            grp['intensity_km_total'].transform(
                lambda s: s.shift(1).rolling(7, min_periods=1).sum()
            )
        )
        km_total_7d = (
            grp['km_total'].transform(
                lambda s: pd.to_numeric(s, errors='coerce').fillna(0.0).shift(1).rolling(7, min_periods=1).sum()
            )
        )
        intensity_share = np.where(
            km_total_7d > 0,
            np.clip(intensity_km_7d / km_total_7d, 0.0, 1.0),
            0.0
        )
        df['intensity_share_7d'] = intensity_share.astype('float32')
        
        # intensity_spike_7d: Intensity spike indicator (current > 1.5x 7d mean)
        intensity_mean7 = (
            grp['intensity_km_total'].transform(
                lambda s: s.shift(1).rolling(7, min_periods=7).mean()
            )
        )
        df['intensity_spike_7d'] = (
            (df['intensity_km_total'] > 1.5 * intensity_mean7) & (intensity_mean7 > 0)
        ).astype('int8')
        
        # high_intensity_days_last_7d: Count of days with high intensity in last 7 days
        # High intensity = intensity_km_total > threshold (e.g., > 5km)
        high_intensity_threshold = 5.0
        df['high_intensity_flag'] = (df['intensity_km_total'] > high_intensity_threshold).astype(int)
        df['high_intensity_days_last_7d'] = (
            grp['high_intensity_flag'].transform(
                lambda s: s.shift(1).rolling(7, min_periods=1).sum()
            )
        ).fillna(0).astype('int8')
        # Drop temporary flag column
        df = df.drop(columns=['high_intensity_flag'], errors='ignore')
        
        # intensity_x_acwr: Intensity + ACWR interaction
        if 'acwr_clipped' in df.columns:
            acwr = pd.to_numeric(df['acwr_clipped'], errors='coerce').fillna(1.0)
            intensity_norm = np.clip(intensity_km_7d / 50.0, 0.0, 3.0)  # Normalize intensity
            df['intensity_x_acwr'] = (intensity_norm * acwr).astype('float32')
        
        # intensity_x_recovery: Intensity + recovery interaction
        if 'recovery_index' in df.columns:
            recovery = pd.to_numeric(df['recovery_index'], errors='coerce').fillna(0.0)
            poor_recovery = np.clip(-recovery, 0.0, 1.0)  # Poor recovery = high value
            intensity_norm = np.clip(intensity_km_7d / 50.0, 0.0, 3.0)
            df['intensity_x_recovery'] = (intensity_norm * poor_recovery).astype('float32')
    
    return df


def build_main_model_table(
    daily: pd.DataFrame,
    users: Optional[pd.DataFrame] = None,
    activities: Optional[pd.DataFrame] = None,
    label_col: str = "injury_next_7d",
    include_users: bool = True,
    include_activity_aggs: bool = True,
) -> Tuple[pd.DataFrame, MainModelSchema]:
    """Build main_model (rich-path) feature table from synthetic outputs."""
    df = _ensure_datetime(daily)

    if include_users and users is not None:
        u = users.copy()
        u["user_id"] = u["user_id"].astype("int64")
        df["user_id"] = df["user_id"].astype("int64")
        # Merge all user columns (fitness, profile, etc.)
        df = df.merge(u, on="user_id", how="left", suffixes=("", "_user"))
        # If fitness/profile got suffixed, rename them back
        if "fitness_user" in df.columns and "fitness" not in df.columns:
            df["fitness"] = df["fitness_user"]
        if "profile_user" in df.columns and "profile" not in df.columns:
            df["profile"] = df["profile_user"]

    if include_activity_aggs and activities is not None and len(activities) > 0:
        a = activities.copy()
        a["user_id"] = a["user_id"].astype("int64")
        a["date"] = pd.to_datetime(a["date"])
        agg = a.groupby(["user_id", "date"]).agg(
            act_sessions=("activity_id", "count"),
            act_km=("distance_km", "sum"),
            act_dur_min=("duration_min", "sum"),
            act_avg_hr=("avg_hr_bpm", "mean"),
            act_elev_gain_m=("elev_gain_m", "sum"),
        ).reset_index()
        df = df.merge(agg, on=["user_id", "date"], how="left")
        for c in ["act_sessions", "act_km", "act_dur_min", "act_avg_hr", "act_elev_gain_m"]:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)

    df = _rolling_features(df)
    df = _rolling_features(df)
    df = _add_directional_features(df)  # CRITICAL: Match injury generation logic
    df = _add_excess_features(df)        # CRITICAL: Match injury generation thresholds
    df = _add_interaction_features(df)
    df = _add_ramp_features(df)
    df = _add_temporal_features(df)
    df = _add_recovery_features(df)
    df = _add_acwr_recovery_interactions(df)  # CRITICAL: Fix reversed ACWR signal
    df = _add_persistent_fatigue_state(df)  # CRITICAL: Add persistent fatigue state
    df = _add_proneness_interactions(df)  # CRITICAL: Add proneness interactions
    df = _add_resilience_interactions(df)  # CRITICAL: Add resilience interactions
    df = _add_long_run_spike_feature(df)  # GARMIN RUNSAFE: Long run spike risk
    # NEW: Additional improvements
    df = _add_sequence_features(df)
    df = _add_enhanced_acwr_features(df)
    df = _add_user_profile_interactions(df)
    df = _add_temporal_user_spike_interactions(df)  # NEW: Temporal/user interactions with spike/post-injury
    # RECOMMENDATION 1: Add consecutive hard days, high intensity sessions, and double sessions features
    df = _add_consecutive_hard_and_intensity_features(df, activities)  # NEW: Consecutive hard days, HI sessions, double sessions
    # PHASE 2: Temporal features now include lag features, slopes, and spike indicators
    # PHASE 2: Add Phase 2 features (Enhanced recovery, sprinting, and intensity zone features)
    df = _add_phase2_features(df)  # PHASE 2: Recovery state, sprinting features, intensity zone features
    df = _encode_known_categoricals(df)

    if label_col not in df.columns:
        raise ValueError(f"Label col '{label_col}' not found in daily table")

    drop_cols = set(LEAKAGE_DROP_COLS)
    drop_cols.update(ID_COLS)

    feature_cols = [c for c in df.columns if c not in drop_cols and c != label_col]
    df_model = df[ID_COLS + [label_col] + feature_cols].copy()

    # Any remaining object columns get a stable (sorted) mapping, so inference can replicate if needed.
    for c in feature_cols:
        if str(df_model[c].dtype) == "bool":
            df_model[c] = df_model[c].astype("int8")
        elif df_model[c].dtype == "object":
            cats = sorted([x for x in df_model[c].dropna().unique().tolist()])
            mapping = {k: i for i, k in enumerate(cats)}
            df_model[c] = df_model[c].map(mapping).fillna(-1).astype("int16")

    feature_dtypes = {c: str(df_model[c].dtype) for c in feature_cols}

    schema = MainModelSchema(
        label_col=label_col,
        id_cols=ID_COLS,
        drop_cols=sorted(list(drop_cols)),
        feature_cols=feature_cols,
        feature_dtypes=feature_dtypes,
        feature_notes={
            "include_users": include_users,
            "include_activity_aggs": include_activity_aggs,
            "categoricals_encoded_as": "deterministic mappings (session_type/sex + sorted uniques for others)",
            "nan_handling": "left as NaN; XGBoost handles missing",
            "rolling_features": "mean7/mean28/std28, sum7/sum28, max7/min7, delta7_28, z7_28, acwr_clipped, hi_share",
            "directional_features": "sleep_def, hrv_drop, rhr_rise, stress_rise (matching injury generation logic)",
            "excess_features": "acwr_excess, ramp_excess (matching injury generation thresholds)",
            "interaction_features": "acwr_x_fitness, load7_x_low_fitness, acwr_x_load7, acwr_excess_x_fitness",
            "ramp_features": "ramp_ratio, ramp_excess, session_spike_ratio, session_spike_mid/high, km_ramp_ratio",
            "temporal_features": "day_of_week, is_weekend, week_of_year, month, days_since_injury",
            "recovery_features": "consecutive_rest_days, recovery_index, load_monotony, acwr_squared, sleep_adequate, hrv_adequate",
            "acwr_recovery_interactions": "acwr_excess_x_poor_recovery, acwr_excess_x_sleep_def, acwr_excess_x_hrv_drop, acwr_excess_x_rhr_rise",
            "persistent_fatigue": "fatigue_state, fatigue_state_lag1 (matching injury generation alpha=0.9)",
            "proneness_interactions": "proneness_x_acwr_excess, proneness_x_fatigue, proneness_x_low_fitness, proneness_x_poor_recovery, proneness_x_sleep_def",
            "resilience_interactions": "low_resilience_x_acwr_excess, low_resilience_x_fatigue, low_resilience_x_poor_recovery, low_resilience_x_sleep_def, proneness_x_low_resilience",
            "long_run_spike": "long_run_spike_risk, long_run_spike_category, long_run_spike_small/moderate/large (GARMIN RUNSAFE: biggest single risk - long run >1.1x max in previous 30d)",
            "long_run_spike_interactions": "spike_x_proneness, spike_x_low_fitness, spike_x_sleep_def, any_spike_x_proneness, any_spike_x_low_fitness, spike_x_proneness_x_low_fitness",
            "spike_absolute_risk": "spike_absolute_risk, has_spike_abs_risk, spike_abs_risk_log (ADDITIVE RISK: direct risk value from data generation, 0.07-0.10 per spike, capped at 40%)",
            "sprinting_absolute_risk": "sprinting_absolute_risk, has_sprinting_abs_risk, sprinting_abs_risk_log (ADDITIVE RISK: direct risk value from data generation, 0.50 per km, capped at 40%)",
            "spike_interactions_enhanced": "spike_x_acwr_excess, spike_x_fatigue_state, spike_x_poor_recovery (NEW)",
            "sequence_features": "consecutive_high_risk_days, spike_frequency_30d, recovery_pattern_score, training_monotony_7d (NEW)",
            "enhanced_acwr_features": "acwr_trajectory, acwr_acceleration, acwr_above_threshold_days, acwr_x_load7 (NEW)",
            "recovery_composite_features": "recovery_deficit_7d, recovery_deficit_14d, recovery_trend, recovery_x_load7, recovery_x_acwr_excess (NEW)",
            "user_profile_interactions": "profile_x_load7, profile_x_poor_recovery, profile_x_spike, fitness_x_load_trajectory (NEW)",
        },
    )

    return df_model, schema
