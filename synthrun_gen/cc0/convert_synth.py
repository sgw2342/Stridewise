from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from .schema import CC0FeatureSchema

def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df

# Real CC0 column order and naming (exact match)
# Order: Features first (all suffixes), then metadata columns at the end
# Suffix order: '' (no suffix = 7 days before), '.1', '.2', '.3', '.4', '.5', '.6' (day before)
REAL_CC0_BASE_FEATURES = [
    "nr. sessions",
    "total km",
    "km Z3-4",
    "km Z5-T1-T2",
    "km sprinting",
    "strength training",
    "hours alternative",
    "perceived exertion",
    "perceived trainingSuccess",
    "perceived recovery",
]

# NEW: Additional features from synthetic data (not in original CC0, but useful for model)
ADDITIONAL_BASE_FEATURES = [
    "spike absolute risk",
    "sprinting absolute risk",
]

# Mapping from synthetic column names to real CC0 base feature names
SYNTH_TO_REAL_MAPPING = {
    "sessions": "nr. sessions",
    "km_total": "total km",
    "kms_z3_4": "km Z3-4",
    "kms_z5_t1_t2": "km Z5-T1-T2",
    "kms_sprinting": "km sprinting",
    "strength_training": "strength training",
    "hours_alternative": "hours alternative",
    "perceived_exertion": "perceived exertion",
    "perceived_trainingSuccess": "perceived trainingSuccess",
    "perceived_recovery": "perceived recovery",
}

# NEW: Mapping for additional features
ADDITIONAL_SYNTH_TO_REAL_MAPPING = {
    "spike_absolute_risk": "spike absolute risk",
    "sprinting_absolute_risk": "sprinting absolute risk",
}

# Suffix order: '' (7 days before), '.1', '.2', '.3', '.4', '.5', '.6' (day before)
REAL_CC0_SUFFIXES = ['', '.1', '.2', '.3', '.4', '.5', '.6']

def convert_synth_to_cc0_timeseries(
    daily_csv: str,
    schema_path: str,
    out_day_csv: str,
    out_week_csv: Optional[str] = None,  # DEPRECATED: Week conversion removed
    anchor_every_day: bool = True,  # Default: Create rows for all event days (matches CC0 structure)
    include_control_rows: bool = False,  # DEPRECATED: Not needed when anchor_every_day=True (all days are event days)
) -> Dict[str, Any]:
    """Create CC0-shaped day timeseries windows from synthetic daily.
    
    This creates an EXACT match to the real CC0 file structure:
    - Column names match exactly (e.g., "nr. sessions", "total km", "Athlete ID", "injury", "Date")
    - Column order matches exactly (features first, then metadata at end)
    - Feature suffix order matches exactly ('' for 7 days before, then '.1' to '.6')
    
    Based on the CC0 paper (Lövdal et al., 2021) and README:
    - Each row corresponds to a 7-day period ending in an "event day" (injury or no injury)
    - For each event day, features are taken from the 7 days BEFORE the event (t-7 to t-1, where t-0 is event day)
    - CC0 suffixes: '' = 7 days before event day (t-7), '.1' = 6 days before (t-6), ..., '.6' = day before (t-1)
    - Label "injury" = 1 if the EVENT DAY (t-0) was an injury, 0 if it was a non-injury event
    - The dataset includes ALL event days (both injury and non-injury), not just injury events

    Args:
        daily_csv: Path to synthetic daily CSV file
        schema_path: Path to CC0 feature schema JSON (used for validation, but structure follows real CC0)
        out_day_csv: Output path for day approach CSV
        out_week_csv: DEPRECATED - Week conversion removed, ignored if provided
        anchor_every_day: If True (default), create rows for all event days (matches CC0 structure).
                          If False, only create rows for injury events (deprecated - doesn't match CC0).
        include_control_rows: DEPRECATED - Not needed when anchor_every_day=True (all days are event days).

    Returns metadata dict.
    """
    schema = CC0FeatureSchema.from_json(schema_path)  # Load schema for validation (but structure follows real CC0)
    d = pd.read_csv(daily_csv)
    d = _ensure_dt(d)
    d["user_id"] = d["user_id"].astype("int64")
    d = d.sort_values(["user_id","date"]).reset_index(drop=True)

    # ensure required columns exist
    req = ["sessions", "km_total", "kms_z3_4", "kms_z5_t1_t2", "kms_sprinting", "strength_training", "hours_alternative",
           "perceived_exertion", "perceived_trainingSuccess", "perceived_recovery", "injury_onset"]
    for c in req:
        if c not in d.columns:
            # allow sessions to be derived
            if c == "sessions":
                if "sessions" not in d.columns:
                    d["sessions"] = (d["km_total"] > 0).astype(int)
            else:
                d[c] = 0.0

    rows_day = []
    row_idx = 0
    
    # NEW: Filter out injury events within 3 weeks (21 days) of a previous injury
    # Based on CC0 paper: "injury events shortly following (within 3 wk of) a new
    # injury have been filtered out, as they are considered to correspond to the same injury."
    def filter_consecutive_injuries(df: pd.DataFrame, window_days: int = 21) -> Tuple[pd.DataFrame, int]:
        """Filter out injury events within window_days of a previous injury onset.
        
        Args:
            df: Daily dataframe sorted by date
            window_days: Number of days to filter after injury onset (default: 21 days = 3 weeks)
        
        Returns:
            Tuple of (filtered DataFrame, count of filtered injury onsets)
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
        
        # Track last injury onset date per athlete
        last_injury_date = {}
        filtered_injury_onsets = 0
        
        for idx, row in df.iterrows():
            uid = row["user_id"]
            current_date = row["date"]
            is_injury = bool(row["injury_onset"] == 1)
            
            if is_injury:
                # Check if this injury is within window_days of previous injury
                if uid in last_injury_date:
                    days_since_last_injury = (current_date - last_injury_date[uid]).days
                    if 0 < days_since_last_injury <= window_days:
                        # This injury is within window_days of previous injury - filter it out
                        df.loc[idx, "injury_onset"] = 0
                        filtered_injury_onsets += 1
                        continue  # Don't update last_injury_date for filtered injuries
                
                # Update last injury date for this athlete (first injury or outside window)
                last_injury_date[uid] = current_date
        
        return df, filtered_injury_onsets
    
    # Apply filtering before processing
    original_injury_count = d["injury_onset"].sum()
    print(f"Filtering injury events within 21 days of previous injury onset...")
    d_filtered, n_filtered = filter_consecutive_injuries(d, window_days=21)
    filtered_injury_count = d_filtered["injury_onset"].sum()
    print(f"  Original injury onsets: {original_injury_count:,}")
    print(f"  Filtered out {n_filtered:,} injury events within 21 days of previous injury")
    print(f"  Remaining injury onsets: {filtered_injury_count:,}")
    if original_injury_count > 0:
        filter_rate = (n_filtered / original_injury_count) * 100
        print(f"  Filter rate: {filter_rate:.1f}% of injury events filtered")
    d = d_filtered

    for uid, g in d.groupby("user_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        
        # Date counter starts at 0 for each athlete (matches real CC0 format)
        athlete_date_counter = 0
        
        if anchor_every_day:
            # Create rows for every day (starting from day 8, since we need 7 days before)
            # According to CC0 README: Each row = 7-day period ending in an "event day" (injury or no injury)
            # We create a row for every day that can serve as an event day (has at least 7 days before it)
            anchor_days = range(7, len(g))  # Start from day 7 to have 7 days of history (t-7 to t-1)
        else:
            # OLD BEHAVIOR (deprecated): Only create rows for injury events
            # This doesn't match CC0 structure - CC0 includes ALL event days (injury and non-injury)
            # Find all injury days (after filtering)
            injury_days = g[g["injury_onset"] == 1].index.tolist()
            anchor_days = [t for t in injury_days if t >= 7]  # Need at least 7 days before injury
            
            # Optionally add matched control rows (non-injury days)
            if include_control_rows:
                # Find non-injury days that can serve as controls
                # For now, randomly sample non-injury days with sufficient history
                non_injury_days = g[(g["injury_onset"] == 0) & (g.index >= 7)].index.tolist()
                # Match control to injury ratio (roughly 1:1 or adjust as needed)
                n_controls = min(len(injury_days), len(non_injury_days))
                if n_controls > 0:
                    import random
                    random.seed(42)  # For reproducibility
                    control_days = random.sample(non_injury_days, n_controls)
                    anchor_days.extend(control_days)
                    anchor_days = sorted(anchor_days)  # Keep chronological order
        
        for t in anchor_days:
            # Check if we have enough history for the day features (need 7 days before)
            if t < 7:
                continue  # Skip if not enough history (need at least 7 days before event day)
            
            # According to CC0 README:
            # - Each row = 7-day period ending in an "event day" (injury or no injury)
            # - Event day is t-0 (the anchor day itself)
            # - Features are from the 7 days BEFORE the event day (t-7 to t-1)
            # - Real CC0 suffixes: '' (no suffix) = 7 days before (t-7), '.1' = 6 days before (t-6), ..., '.6' = day before (t-1)
            # - Label "injury" = 1 if the EVENT DAY (t-0) was an injury, 0 if not
            
            anchor = g.loc[t, "date"]
            is_injury_event = bool(g.loc[t, "injury_onset"])  # Event day is injury or not
            
            # Day window: 7 days BEFORE the event day (t-7 to t-1, where t-0 is the event day)
            # Features are from t-7 to t-1 (excluding the event day itself)
            hist_start = max(0, t - 7)
            hist_end = t - 1  # Up to but not including event day (t-0)
            hist = g.loc[hist_start:hist_end].copy()
            
            # If we don't have exactly 7 days, skip
            if len(hist) < 7:
                continue
                
            # Ensure we have exactly the 7 most recent days before the event day
            if len(hist) > 7:
                hist = hist.tail(7).copy()  # Take last 7 days (t-7 to t-1)
            
            # hist is chronological: [t-7, t-6, t-5, t-4, t-3, t-2, t-1]
            # Real CC0 suffix order: ['' (t-7, index 0), '.1' (t-6, index 1), ..., '.6' (t-1, index 6)]
            
            # Create row with REAL CC0 column names and order
            # Order: Features first (all suffixes), then metadata columns at end
            day_row = {}
            
            # Add features in REAL CC0 order: all features for each suffix (day), then next suffix
            # Real CC0 structure: All day 0 (7 days before), then all day 1, ..., then all day 6 (day before)
            # For each suffix (day), iterate through all base features
            for suffix_idx, real_suffix in enumerate(REAL_CC0_SUFFIXES):
                # real_suffix: '' (7 days before, index 0), '.1' (6 days before, index 1), ..., '.6' (day before, index 6)
                # hist index: 0 (t-7), 1 (t-6), ..., 6 (t-1)
                hist_idx = suffix_idx  # Map suffix index to history index
                
                # Add standard CC0 features
                for real_base_feature in REAL_CC0_BASE_FEATURES:
                    # Find the corresponding synthetic column name
                    synth_col = None
                    for synth_key, real_key in SYNTH_TO_REAL_MAPPING.items():
                        if real_key == real_base_feature:
                            synth_col = synth_key
                            break
                    
                    if synth_col is None:
                        continue
                    
                    # Get value from history (hist[hist_idx] corresponds to t-7+hist_idx)
                    val = hist[synth_col].iloc[hist_idx]
                    
                    # Normalize perceived ratings to 0-1 scale (CC0 uses 0-1, synthetic uses 1-10)
                    if real_base_feature in ["perceived exertion", "perceived trainingSuccess", "perceived recovery"]:
                        # Convert from 1-10 scale to 0-1 scale, keeping -0.01 for rest days
                        if val == -0.01:
                            val = -0.01
                        else:
                            val = float(np.clip((val - 1.0) / 9.0, 0.0, 1.0))  # Map 1-10 to 0-1
                    else:
                        val = float(val)
                    
                    # Round to 2 decimal places (matching real CC0 format), except for -0.01 (rest day marker)
                    if val != -0.01 and np.isfinite(val):
                        val = round(val, 2)
                    
                    # Create column name: "nr. sessions" + "" = "nr. sessions", "nr. sessions" + ".1" = "nr. sessions.1"
                    col_name = real_base_feature if real_suffix == '' else f"{real_base_feature}{real_suffix}"
                    
                    # Store value (convert NaN to np.nan if needed)
                    day_row[col_name] = val if np.isfinite(val) else np.nan
                
                # NEW: Add additional features (spike_absolute_risk, sprinting_absolute_risk)
                for real_base_feature in ADDITIONAL_BASE_FEATURES:
                    # Find the corresponding synthetic column name
                    synth_col = None
                    for synth_key, real_key in ADDITIONAL_SYNTH_TO_REAL_MAPPING.items():
                        if real_key == real_base_feature:
                            synth_col = synth_key
                            break
                    
                    if synth_col is None:
                        continue
                    
                    # Get value from history (hist[hist_idx] corresponds to t-7+hist_idx)
                    # Use 0.0 as default if column doesn't exist
                    if synth_col in hist.columns:
                        val = hist[synth_col].iloc[hist_idx]
                    else:
                        val = 0.0
                    
                    val = float(val) if np.isfinite(val) else 0.0
                    
                    # Round to 4 decimal places for risk values (more precision needed)
                    if np.isfinite(val):
                        val = round(val, 4)
                    
                    # Create column name: "spike absolute risk" + "" = "spike absolute risk", etc.
                    col_name = real_base_feature if real_suffix == '' else f"{real_base_feature}{real_suffix}"
                    
                    # Store value
                    day_row[col_name] = val
            
            # Add metadata columns at the end (matching real CC0 order)
            day_row["Athlete ID"] = int(uid)  # Real CC0 uses "Athlete ID" (not "maskedID")
            day_row["injury"] = 1 if is_injury_event else 0  # Real CC0 uses "injury" (not "y")
            day_row["Date"] = int(athlete_date_counter)  # Real CC0 uses integer count starting from 0 (not actual date)
            
            rows_day.append(day_row)
            row_idx += 1
            athlete_date_counter += 1  # Increment date counter for next row for this athlete

    # Create DataFrame and ensure column order matches real CC0 exactly
    day_df = pd.DataFrame(rows_day)
    
    # Define exact column order to match real CC0
    # Real CC0 structure: All features for day 0 ('' suffix), then all features for day 1 ('.1' suffix), etc.
    # Then metadata columns at the end
    expected_columns = []
    
    # Add features in order: all features for each suffix (day), then next suffix
    # First add standard CC0 features
    for suffix in REAL_CC0_SUFFIXES:
        for real_base_feature in REAL_CC0_BASE_FEATURES:
            col_name = real_base_feature if suffix == '' else f"{real_base_feature}{suffix}"
            if col_name in day_df.columns:
                expected_columns.append(col_name)
    
    # Then add additional features (spike_absolute_risk, sprinting_absolute_risk)
    for suffix in REAL_CC0_SUFFIXES:
        for real_base_feature in ADDITIONAL_BASE_FEATURES:
            col_name = real_base_feature if suffix == '' else f"{real_base_feature}{suffix}"
            if col_name in day_df.columns:
                expected_columns.append(col_name)
    
    # Add metadata columns at the end (matching real CC0 order)
    for meta_col in ["Athlete ID", "injury", "Date"]:
        if meta_col in day_df.columns:
            expected_columns.append(meta_col)
    
    # Reorder columns to match real CC0 exactly
    day_df = day_df[expected_columns]
    
    # Save to CSV
    day_df.to_csv(out_day_csv, index=False)

    return {
        "n_rows": int(row_idx),
        "n_injury_rows": int(day_df["injury"].sum()),
        "n_control_rows": int((day_df["injury"] == 0).sum()),
        "injury_rate": float(day_df["injury"].mean()),
        "original_injury_count": int(original_injury_count),
        "filtered_injury_count": int(n_filtered),
        "remaining_injury_count": int(filtered_injury_count),
        "out_day_csv": out_day_csv,
    }
