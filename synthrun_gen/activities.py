from __future__ import annotations
import numpy as np
import pandas as pd
from .config import GeneratorConfig
from .utils import clip01, date_range

SESSION_TYPES = ["easy", "tempo", "interval", "long"]

def _pick_training_pattern(rng: np.random.Generator):
    # Weekly session mix template (counts)
    # easy dominates, 1 quality, 1 long, optional extra easy
    if rng.random() < 0.55:
        return {"easy": 3, "tempo": 1, "interval": 0, "long": 1}
    elif rng.random() < 0.75:
        return {"easy": 2, "tempo": 0, "interval": 1, "long": 1}
    else:
        return {"easy": 3, "tempo": 0, "interval": 1, "long": 1}

def _pace_from_easy(pace_easy: float, session_type: str):
    # relative pace multipliers
    if session_type == "easy":
        return pace_easy * np.random.normal(1.0, 0.03)
    if session_type == "long":
        return pace_easy * np.random.normal(1.03, 0.03)
    if session_type == "tempo":
        return pace_easy * np.random.normal(0.88, 0.03)
    if session_type == "interval":
        return pace_easy * np.random.normal(0.80, 0.04)
    return pace_easy

def _hr_intensity_factor(session_type: str):
    # fraction of HRR used (for avg HR)
    if session_type == "easy":
        return 0.62
    if session_type == "long":
        return 0.66
    if session_type == "tempo":
        return 0.78
    if session_type == "interval":
        return 0.86
    return 0.65

def generate_activities(cfg: GeneratorConfig, users: pd.DataFrame, daily_plan: pd.DataFrame,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Generate per-activity sessions based on a daily plan (distance/intensity)."""
    # daily_plan includes: user_id, date, km_total, session_type (or rest)
    rows = []
    act_id = 1

    # Ensure consistent dtypes for merges (bug fix)
    users_local = users.copy()
    users_local["user_id"] = users_local["user_id"].astype("int64")
    daily_plan_local = daily_plan.copy()
    daily_plan_local["user_id"] = daily_plan_local["user_id"].astype("int64")

    u_lookup = users_local.set_index("user_id")

    # FIXED: Handle single and double sessions per day
    # Process each row in daily_plan (one row per day, but may have double session info)
    for idx, row in daily_plan_local.iterrows():
        uid = int(row["user_id"])
        date = row["date"]
        
        # Primary session (AM)
        km_primary = float(row["km_total"])
        st_primary = str(row["session_type"])
        if st_primary not in SESSION_TYPES:
            st_primary = "easy"
        
        # Check for double session
        has_double = int(row.get("has_double", 0)) if "has_double" in row else 0
        
        if km_primary > 0:
            # Generate primary session (AM)
            _generate_activity(rows, act_id, u_lookup, uid, date, km_primary, st_primary, cfg, rng)
            act_id += 1
            
            # Generate secondary session (PM) if double session
            if has_double:
                km_secondary = float(row.get("double_km", 0.0))
                st_secondary = str(row.get("double_session_type", "easy"))
                if st_secondary not in SESSION_TYPES:
                    st_secondary = "easy"
                
                if km_secondary > 0:
                    _generate_activity(rows, act_id, u_lookup, uid, date, km_secondary, st_secondary, cfg, rng)
                    act_id += 1

    acts = pd.DataFrame(rows)
    if len(acts) == 0:
        acts = pd.DataFrame(columns=[
            "activity_id","user_id","date","session_type","distance_km","duration_min","pace_min_per_km",
            "avg_hr_bpm","elev_gain_m","kms_z3_4","kms_z5_t1_t2","kms_sprinting",
            "cadence_spm","gct_ms","stride_length_cm","vertical_oscillation_cm","gct_balance"
        ])
    acts["user_id"] = acts["user_id"].astype("int64")
    acts["activity_id"] = acts["activity_id"].astype("int64")
    return acts

def _generate_activity(rows: list, act_id: int, u_lookup: pd.DataFrame, uid: int, date: str, 
                       km: float, st: str, cfg: GeneratorConfig, rng: np.random.Generator):
    """Helper function to generate a single activity."""
    if km <= 0.0:
        return
    
    u = u_lookup.loc[uid]
    pace_easy = float(u["pace_easy_minpkm"])
    pace = float(np.clip(_pace_from_easy(pace_easy, st), 3.4, 10.0))
    duration_min = float(max(8.0, km * pace))

    # Avg HR with HRmax/HRR + noise + cardiac drift for long sessions
    rhr = float(u["rhr_base"])
    hrr = float(u["hrr"])
    base_factor = _hr_intensity_factor(st)
    avg_hr = rhr + hrr * base_factor + rng.normal(0, 3.0)
    drift = cfg.hr_drift_per_hour * max(0.0, (duration_min - 60.0) / 60.0)
    avg_hr = float(np.clip(avg_hr + drift, rhr + 15, u["hrmax"] - 5))

    # simple elevation gain proxy: more in long, random
    elev_gain_m = float(max(0.0, rng.normal(8.0*km, 40.0) * (1.35 if st=="long" else 1.0)))

    # PHASE 1: Reduced intensity zones for advanced/elite to match real CC0 distributions
    # Target: Intensity share 6-7% (from 12%), Z3-4 distance 0.75-0.80 km/day (from 1.03 km/day)
    # Only applies to advanced and elite profiles (novice/recreational unchanged)
    is_elite = False
    is_advanced = False
    if "profile" in u:
        profile_str = str(u["profile"]).lower()
        is_elite = profile_str == "elite"
        is_advanced = profile_str == "advanced"
    elif "fitness" in u:
        fitness_val = float(u["fitness"])
        is_elite = fitness_val >= 0.85
        is_advanced = 0.7 <= fitness_val < 0.85
    
    is_advanced_or_elite = is_advanced or is_elite
    
    if st == "tempo":
        if is_elite:
            # Z3-4 DISTRIBUTION FIX v4: Match real CC0 (10.99% of days, 6.29 km/day when present)
            # Real CC0: 10.99% of days have Z3-4, mean 6.29 km/day when present
            # If tempo is ~15% of days: 10.99% / 15% = 73% of tempo sessions should have Z3-4
            # For 6 km tempo session: need 6.29/6 = 105% fraction (cap at 100%)
            # Strategy: 73% of tempo sessions have Z3-4 with frac_z3_4 = 1.0 (100% of session)
            if rng.random() < 0.73:  # 73% of tempo sessions have Z3-4
                # High Z3-4 amount when present (target: 6.29 km/day for 6 km session = 100% fraction)
                # Use maximum fraction to ensure we get enough Z3-4 distance
                frac_z3_4 = float(np.clip(rng.normal(0.95, 0.05), 0.85, 1.0))  # Very high fraction (near 100%)
            else:
                # No Z3-4 in this tempo session (27% of tempo sessions)
                frac_z3_4 = 0.0
            # Z5 DISTRIBUTION FIX: Remove Z5 from tempo sessions entirely
            # Real CC0 shows 13.32% of days have Z5
            # Tempo sessions should be Z3-4 focused, not Z5
            # Set frac_z5 = 0.0 for all tempo sessions (no Z5 in tempo)
            frac_z5 = 0.0  # No Z5 in tempo sessions (tempo is Z3-4 focused only)
        elif is_advanced:
            # Z3-4 DISTRIBUTION FIX v4: Match real CC0 (10.99% of days, 6.29 km/day when present)
            # Strategy: 73% of tempo sessions have Z3-4
            if rng.random() < 0.73:  # 73% of tempo sessions have Z3-4
                # High Z3-4 amount when present (target: 6.29 km/day)
                frac_z3_4 = float(np.clip(rng.normal(0.95, 0.05), 0.85, 1.0))  # Very high fraction (near 100%)
            else:
                # No Z3-4 in this tempo session (27% of tempo sessions)
                frac_z3_4 = 0.0
            # Z5 DISTRIBUTION FIX: Remove Z5 from tempo sessions entirely
            # Tempo sessions should be Z3-4 focused, not Z5
            frac_z5 = 0.0  # No Z5 in tempo sessions (tempo is Z3-4 focused only)
        else:
            # Novice/Recreational: Keep original values (unchanged)
            frac_z3_4 = float(np.clip(rng.normal(0.30, 0.10), 0.15, 0.50))  # Original value
            frac_z5 = float(np.clip(rng.normal(0.02, 0.01), 0.0, 0.06))  # Original value
    elif st == "interval":
        if is_elite:
            # Z5 DISTRIBUTION FIX v5: Match real CC0 (13.32% of days, 4.35 km/day when present)
            # Real CC0: 13.32% of days have Z5, mean 4.35 km/day when present
            # Current: 25.86% of days, mean 3.04 km/day when present
            # Need: Reduce frequency (25.86% → 13.32% = 52% of current)
            #       Increase amounts (3.04 → 4.35 km/day = 43% increase)
            # Strategy: 50% of interval sessions have Z5 (reduced from 100%)
            #          Significantly increase frac_z5 to get 4.35 km/day
            #          For 6 km interval session: need 4.35/6 = 73% fraction
            # Keep Z3-4 at zero for intervals (intervals are Z5-focused only)
            frac_z3_4 = 0.0  # No Z3-4 in intervals (intervals are Z5-focused only)
            # Z5 DISTRIBUTION FIX v6: Match real CC0 (13.32% of days, 4.35 km/day when present)
            # Real CC0: 13.32% of days have Z5, mean 4.35 km/day when present
            # Current: 13.96% of days (close!), mean 5.88 km/day when present (too high)
            # Need to reduce amounts: 5.88 → 4.35 km/day = 74% of current
            # Strategy: 100% of interval sessions have Z5 (frequency is correct)
            #          Reduce fraction to get 4.35 km/day: 0.95 * 0.74 = 0.70
            #          But account for session length variation, use 0.75-0.80
            # All interval sessions should have Z5 (100%)
            # Moderate-high Z5 amount (target: 4.35 km/day when present)
            # For 6 km session: need 4.35/6 = 73% fraction
            frac_z5 = float(np.clip(rng.normal(0.75, 0.08), 0.65, 0.90))  # Reduced from 0.95 to match amounts
        elif is_advanced:
            # Z5 DISTRIBUTION FIX v5: Match real CC0 (13.32% of days, 4.35 km/day when present)
            # Strategy: 50% of interval sessions have Z5 (reduced from 100%)
            frac_z3_4 = 0.0  # No Z3-4 in intervals (intervals are Z5-focused only)
            # Z5 DISTRIBUTION FIX v6: Match real CC0 (13.32% of days, 4.35 km/day when present)
            # Strategy: 100% of interval sessions have Z5 (frequency is correct)
            #          Reduce fraction to get 4.35 km/day when present
            # All interval sessions should have Z5 (100%)
            frac_z5 = float(np.clip(rng.normal(0.75, 0.08), 0.65, 0.90))  # Reduced to match amounts
        else:
            # Novice/Recreational: Keep original values (unchanged)
            frac_z3_4 = float(np.clip(rng.normal(0.10, 0.05), 0.0, 0.20))  # Original value
            frac_z5 = float(np.clip(rng.normal(0.22, 0.08), 0.10, 0.40))  # Original value
    elif st == "long":
        # FIXED: Long runs should have NO intensity zones to match real CC0 pattern
        # Real CC0 shows 86-89% zeros for intensity zones, meaning most days (including long runs) have no intensity
        # Long runs in break weeks are converted to easy runs (0% intensity)
        # Long runs in training weeks should also have 0% intensity to match real CC0 distribution
        frac_z3_4 = 0.0  # Long runs have no intensity zones (matching real CC0 pattern)
        frac_z5 = 0.0    # Long runs have no high-intensity zones
    else:
        # FIXED: Easy runs should have NO intensity zones (matching easy/base week logic)
        # Easy runs in easy/base weeks should have 0% intensity to match real CC0 pattern
        # Previously: frac_z3_4 = 0.05 (5% intensity), now 0% for true easy runs
        frac_z3_4 = 0.0  # Easy runs have no intensity zones
        frac_z5 = 0.0    # Easy runs have no high-intensity zones
    
    # FIXED: Sprint only for tempo/interval sessions (not easy/long)
    # INCREASED FOR ADVANCED/ELITE ONLY: Real data shows 0.1376 km mean sprinting at injury (vs 0.0482 km in synthetic)
    # Real data shows sprinting is the #1 injury driver (1.92x ratio)
    # Need to significantly increase sprinting amount to match real data, but ONLY for advanced/elite runners
    # FITNESS-LEVEL DEPENDENT SPRINTING FREQUENCY: Novice/recreational runners are less likely to do sprinting at all
    # This reduces sprinting-related injuries by reducing the chance they do sprinting, not the risk per km
    if st in ["interval", "tempo"]:
        # Check if user is advanced or elite (by profile or fitness)
        is_advanced_or_elite = False
        if "profile" in u:
            profile_str = str(u["profile"]).lower()
            is_advanced_or_elite = profile_str in ["advanced", "elite"]
        # Also check fitness level as fallback
        if not is_advanced_or_elite and "fitness" in u:
            is_advanced_or_elite = float(u["fitness"]) >= 0.7
        
        if is_advanced_or_elite:
            # SPRINTING DISTRIBUTION FIX: Match real CC0 distribution for advanced/elite
            # Real CC0: 7.54% of days have sprinting, mean 0.968 km/day when present
            # Synthetic (current): 27.62% of days have sprinting, mean 0.324 km/day when present
            # Target: Reduce frequency to 7.54%, increase amounts to 0.968 km/day when present
            # Also: 27% of advanced/elite users should never sprint
            
            # Check if user never sprints (27% of advanced/elite)
            never_sprints = False
            if "never_sprints" in u:
                never_sprints = bool(int(u["never_sprints"]))
            
            if never_sprints:
                # User never sprints (27% of advanced/elite)
                frac_sprint = 0.0
            else:
                # SPRINTING FREQUENCY FIX: Reduce from 27.62% to 7.54% of days
                # Current: Every tempo/interval session has sprinting (fraction 0.052)
                # Target: Only 7.54% of days should have sprinting
                # Strategy: Only add sprinting to 7.54% / (tempo+interval frequency) of sessions
                # For advanced/elite: ~2-3 tempo/interval sessions per week = ~30% of days
                # So: 7.54% / 30% = ~25% of tempo/interval sessions should have sprinting
                # Adjusted to 35% to account for never_sprints users reducing overall frequency
                # Target: 7.54% of days with sprinting
                # With never_sprints (27%) and 35% of tempo/interval sessions, we get ~7.5% of days
                # SPRINTING FREQUENCY FIX: Reduce from 27.62% to 7.54% of days
                # Current: Every tempo/interval session has sprinting (fraction 0.052)
                # Target: Only 7.54% of days should have sprinting
                # Strategy: Only add sprinting to 7.54% / (tempo+interval frequency) of sessions
                # For advanced/elite: ~2-3 tempo/interval sessions per week = ~30% of days
                # So: 7.54% / 30% = ~25% of tempo/interval sessions should have sprinting
                # Adjusted to 35% to account for never_sprints users reducing overall frequency
                # Target: 7.54% of days with sprinting
                # With never_sprints (27%) and 35% of tempo/interval sessions, we get ~7.5% of days
                sprinting_session_prob = 0.35  # 35% of tempo/interval sessions have sprinting (adjusted for never_sprints)
                
                if rng.random() < sprinting_session_prob:
                    # SPRINTING AMOUNT FIX: Increase from 0.324 to 0.968 km/day when present
                    # Real CC0: mean = 0.968 km/day, median = 0.6 km/day for non-zero days
                    # Current: mean = 0.052 fraction → ~0.324 km/day
                    # Target: mean = 0.968 km/day for non-zero days
                    # For a typical tempo/interval session of 8-12 km, need ~8-12% fraction
                    # But we want higher variance and occasional extreme outliers (up to 40 km/day in real CC0)
                    # Use higher mean with more variance
                    base_fraction = 0.12  # Base 12% of session distance (increased from 10%)
                    # Add variability: some sessions have much more sprinting
                    if rng.random() < 0.15:  # 15% chance of high sprinting session
                        # High sprinting session (like sprint workout)
                        frac_sprint = float(np.clip(rng.normal(0.30, 0.12), 0.20, 0.60))
                    elif rng.random() < 0.05:  # 5% chance of extreme sprinting session
                        # Extreme sprinting session (matching real CC0 max of 40 km/day)
                        # For a 10 km session, this would be 100%+ fraction, so cap at reasonable max
                        # Allow very high fractions to match real CC0 extreme outliers
                        frac_sprint = float(np.clip(rng.normal(0.60, 0.25), 0.50, 2.0))  # Allow up to 200% (for very long sessions)
                    else:
                        # Normal sprinting session (increased mean to match target)
                        frac_sprint = float(np.clip(rng.normal(base_fraction, 0.06), 0.06, 0.25))
                else:
                    # No sprinting in this session (75% of tempo/interval sessions)
                    frac_sprint = 0.0
        else:
            # Novice/Recreational: Reduced probability of doing sprinting at all
            # Get sprinting probability from config (default 50% chance)
            sprinting_prob = getattr(cfg, 'sprinting_probability_novice_rec', 0.50)
            if rng.random() < sprinting_prob:
                # Include sprinting (but still lower amount than advanced/elite)
                frac_sprint = float(np.clip(rng.normal(0.02, 0.015), 0.0, 0.10))  # 2% mean, 10% max
            else:
                # Skip sprinting entirely (50% chance for novice/rec)
                frac_sprint = 0.0
    else:
        # Easy/long runs: No sprinting
        frac_sprint = 0.0

    # Gait metrics
    cadence_base = 170.0 - (pace - 5.0) * 5.0
    cadence_spm = float(np.clip(rng.normal(cadence_base, 5.0), 150, 200))
    gct_base = 250.0 + (duration_min - 30.0) * 0.5
    gct_ms = float(np.clip(rng.normal(gct_base, 15.0), 180, 350))
    stride_length_cm = float((pace * 1000 / cadence_spm) * 100)
    stride_length_cm = float(np.clip(stride_length_cm, 100, 200))
    vo_base = 8.0 + (duration_min - 30.0) * 0.05
    vertical_oscillation_cm = float(np.clip(rng.normal(vo_base, 1.5), 5.0, 15.0))
    gct_balance = float(np.clip(rng.normal(0.5, 0.03), 0.45, 0.55))

    rows.append({
        "activity_id": act_id,
        "user_id": int(uid),
        "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
        "session_type": st,
        "distance_km": round(km, 3),
        "duration_min": round(duration_min, 2),
        "pace_min_per_km": round(pace, 3),
        "avg_hr_bpm": round(avg_hr, 1),
        "elev_gain_m": round(elev_gain_m, 1),
        "kms_z3_4": round(km*frac_z3_4, 3),
        "kms_z5_t1_t2": round(km*frac_z5, 3),
        "kms_sprinting": round(km*frac_sprint, 3),
        "cadence_spm": round(cadence_spm, 1),
        "gct_ms": round(gct_ms, 1),
        "stride_length_cm": round(stride_length_cm, 1),
        "vertical_oscillation_cm": round(vertical_oscillation_cm, 1),
        "gct_balance": round(gct_balance, 3),
    })
