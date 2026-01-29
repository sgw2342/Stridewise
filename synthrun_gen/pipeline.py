from __future__ import annotations
import json, os, hashlib, platform, datetime, sys
from pathlib import Path
import numpy as np
import pandas as pd

from .config import GeneratorConfig
from .users import generate_users
from .daily import build_daily_plan, generate_daily_signals
from .activities import generate_activities
from .events import generate_events
from .sanity import run_sanity_checks
from .io import write_outputs
import numpy as np

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _smooth_ramp(progress: float, start_value: float, end_value: float, steepness: float) -> float:
    """Smooth ramp from start_value to end_value as progress goes from 0 to 1.
    
    Uses sigmoid-like curve for natural progression.
    
    Args:
        progress: float in [0.0, 1.0] - progress through return period
        start_value: float - starting multiplier value (e.g., 0.25 for 25% volume)
        end_value: float - ending multiplier value (e.g., 1.0 for 100% volume)
        steepness: float - Higher = faster transition (default 3.0 gives gradual ramp)
    
    Returns:
        float - multiplier value between start_value and end_value
    """
    # Clamp progress to [0, 1]
    progress = max(0.0, min(1.0, progress))
    
    # Sigmoid curve centered at 0.5 for smooth S-curve
    sigmoid = 1.0 / (1.0 + np.exp(-steepness * (progress - 0.5)))
    return float(start_value + (end_value - start_value) * sigmoid)


def _adjust_perceived_features_for_injuries(daily: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Post-process perceived features to strengthen injury associations.
    
    Real CC0 shows strong associations on injury days (t-0) when looking at t-7 to t-1:
    - perceived exertion: +35.8% higher on injury days (looking at t-7)
    - perceived trainingSuccess: +31.8% higher on injury days (also 22% should be zero)
    - perceived recovery: +26.3% higher on injury days
    
    Synthetic currently shows weak associations:
    - perceived exertion: +11.3% higher on injury days
    - perceived trainingSuccess: +8.9% higher on injury days (0.07% are zero)
    - perceived recovery: +10.5% higher on injury days
    
    NOTE: Injury_onset days are typically rest days (100% have -0.01), so we adjust
    perceived features on days BEFORE the injury (t-1 to t-7) to create the association.
    """
    daily = daily.copy()
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)
    
    # Get injury onset days
    injury_days = daily["injury_onset"] == 1
    non_injury_days = daily["injury_onset"] == 0
    
    if injury_days.sum() == 0:
        return daily  # No injuries, nothing to adjust
    
    # For each injury_onset day, adjust perceived features on days t-1 to t-7 (before the injury)
    # This creates the association that the model sees in CC0 format
    for idx in daily[injury_days].index:
        user_id = daily.loc[idx, "user_id"]
        # Get days t-1 to t-7 (7 days before injury)
        user_data = daily[daily["user_id"] == user_id].sort_values("date").reset_index(drop=True)
        injury_idx_in_user = user_data[user_data["injury_onset"] == 1].index
        
        for injury_user_idx in injury_idx_in_user:
            # Adjust days t-1 to t-7 (up to 7 days before injury)
            for days_before in range(1, min(8, injury_user_idx + 1)):
                adjust_idx = user_data.index[injury_user_idx - days_before]
                global_idx = user_data.loc[adjust_idx, "date"]
                # Find the global index for this date
                global_row = daily[(daily["user_id"] == user_id) & (daily["date"] == global_idx)]
                if len(global_row) > 0:
                    global_idx_actual = global_row.index[0]
                    
                    # Only adjust active days (not rest days with -0.01)
                    if daily.loc[global_idx_actual, "perceived_exertion"] != -0.01:
                        # Adjust perceived exertion (increase by smaller amount, distributed across t-7 to t-1)
                        # Target: +35.8% on t-7, so apply ~5% boost per day, but only once per day
                        # Use a smaller multiplier to avoid over-correction
                        exertion_mult = 1.0 + (0.025 * (8 - days_before))  # t-7 gets 17.5% boost, t-1 gets 1.75% boost
                        daily.loc[global_idx_actual, "perceived_exertion"] = min(10.0, daily.loc[global_idx_actual, "perceived_exertion"] * exertion_mult)
                        
                        # Adjust perceived trainingSuccess (increase by smaller amount)
                        if daily.loc[global_idx_actual, "perceived_trainingSuccess"] != 0.0:
                            success_mult = 1.0 + (0.022 * (8 - days_before))  # t-7 gets 15.4% boost, t-1 gets 1.54% boost
                            daily.loc[global_idx_actual, "perceived_trainingSuccess"] = min(10.0, daily.loc[global_idx_actual, "perceived_trainingSuccess"] * success_mult)
                        
                        # Adjust perceived recovery (increase by larger amount to match target)
                        # Target: +26.3% on injury days, so need stronger boost on t-7 to t-1
                        recovery_mult = 1.0 + (0.030 * (8 - days_before))  # t-7 gets 21% boost, t-1 gets 2.1% boost (increased from 0.018)
                        daily.loc[global_idx_actual, "perceived_recovery"] = min(10.0, daily.loc[global_idx_actual, "perceived_recovery"] * recovery_mult)
    
    # Also add zeros for trainingSuccess: 22% of days should be zero
    if "perceived_trainingSuccess" in daily.columns:
        all_active = daily["perceived_trainingSuccess"] != -0.01
        n_active = all_active.sum()
        n_zeros_needed = int(n_active * 0.22)  # 22% should be zero
        
        if n_zeros_needed > 0:
            zero_candidates = daily[all_active].index.tolist()
            rng.shuffle(zero_candidates)
            zeros_to_set = zero_candidates[:n_zeros_needed]
            daily.loc[zeros_to_set, "perceived_trainingSuccess"] = 0.0
    
    # 1. PERCEIVED EXERTION: Increase on injury days (+35.8% vs +11.3%)
    # Real CC0: Injury days have 35.8% more exertion than non-injury days
    # Current synthetic: Only 11.3% more (needs +24.5% improvement)
    # Strategy: Increase multiplier to 1.35x to reach ~35.8% increase
    # Current 1.25x only gives ~11-13%, need stronger boost
    if "perceived_exertion" in daily.columns:
        # Only adjust non-rest days (exertion != -0.01)
        injury_active = injury_days & (daily["perceived_exertion"] != -0.01)
        if injury_active.sum() > 0:
            # INCREASED multiplier: 1.35x to reach target (~35% increase)
            # This should bring association from ~11% to ~35%
            daily.loc[injury_active, "perceived_exertion"] = daily.loc[injury_active, "perceived_exertion"] * 1.35
            daily.loc[injury_active, "perceived_exertion"] = daily.loc[injury_active, "perceived_exertion"].clip(1.0, 10.0)
    
    # 2. PERCEIVED TRAININGSUCCESS: Increase on injury days (+31.8% vs +8.9%) AND add zeros (22% should be zero)
    # Real CC0: Injury days have 31.8% more trainingSuccess, and 22% of days have zero trainingSuccess
    # Current synthetic: Only 8.9% more, and 0.07% are zero
    # Strategy: 
    #   a) Add zeros: 22% of days should be zero (matching real CC0)
    #   b) Directly multiply injury day values by 1.23x to reach ~31.8% increase
    if "perceived_trainingSuccess" in daily.columns:
        # Only adjust non-rest days (trainingSuccess != -0.01)
        injury_active = injury_days & (daily["perceived_trainingSuccess"] != -0.01)
        all_active = daily["perceived_trainingSuccess"] != -0.01
        
        # a) Add zeros: 22% of days should be zero (matching real CC0)
        # Set zeros on both injury and non-injury days proportionally
        n_active = all_active.sum()
        n_zeros_needed = int(n_active * 0.22)  # 22% should be zero
        
        if n_zeros_needed > 0:
            # Randomly select days to set to zero (preserving injury/non-injury ratio)
            zero_candidates = daily[all_active].index.tolist()
            rng.shuffle(zero_candidates)
            zeros_to_set = zero_candidates[:n_zeros_needed]
            daily.loc[zeros_to_set, "perceived_trainingSuccess"] = 0.0
        
        # b) Increase trainingSuccess on injury days (excluding zeros and rest days)
        injury_active_not_zero = injury_active & (daily["perceived_trainingSuccess"] != 0.0)
        if injury_active_not_zero.sum() > 0:
            # INCREASED multiplier: 1.32x to reach target (~32% increase)
            # Current 1.23x only gives ~9%, need stronger boost
            # This should bring association from ~9% to ~32%
            daily.loc[injury_active_not_zero, "perceived_trainingSuccess"] = daily.loc[injury_active_not_zero, "perceived_trainingSuccess"] * 1.32
            daily.loc[injury_active_not_zero, "perceived_trainingSuccess"] = daily.loc[injury_active_not_zero, "perceived_trainingSuccess"].clip(1.0, 10.0)
    
    # 3. PERCEIVED RECOVERY: Increase on injury days (+26.3% vs +11.4%)
    # Real CC0: Injury days have 26.3% more recovery than non-injury days
    # Current synthetic: Only 11.4% more (needs +14.9% improvement)
    # Strategy: Increase multiplier to 1.35x to reach ~26.3% increase
    # Current 1.30x still only gives ~11%, need even stronger boost
    # Note: The t-7 to t-1 adjustment also helps, but injury day adjustment is primary
    if "perceived_recovery" in daily.columns:
        # Only adjust non-rest days (recovery != -0.01)
        injury_active = injury_days & (daily["perceived_recovery"] != -0.01)
        if injury_active.sum() > 0:
            # INCREASED multiplier: 1.35x to reach target (~26% increase)
            # This should bring association from ~11% to ~26%
            daily.loc[injury_active, "perceived_recovery"] = daily.loc[injury_active, "perceived_recovery"] * 1.35
            daily.loc[injury_active, "perceived_recovery"] = daily.loc[injury_active, "perceived_recovery"].clip(1.0, 10.0)
    
    return daily

def generate_dataset(cfg: GeneratorConfig, out_dir: str, run_checks: bool = True, elite_only: bool = False) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # 1) users - generate only advanced/elite if flag is set
    users = generate_users(cfg, rng, elite_only=elite_only)
    
    if elite_only:
        # Verify we got advanced/elite users
        profile_counts = users["profile"].value_counts()
        non_elite = users[~users["profile"].isin(["advanced", "elite"])]
        if len(non_elite) > 0:
            print(f"⚠️  WARNING: {len(non_elite)} non-elite users generated in elite-only mode")
            print(f"   Profiles generated: {profile_counts.to_dict()}")
            # Filter to ensure only advanced/elite (shouldn't happen if generate_users works correctly)
            users = users[users["profile"].isin(["advanced", "elite"])].copy()
        
        if len(users) == 0:
            raise ValueError("No advanced/elite users generated! Try increasing --n-users or adjusting elite generation parameters.")
        
        print(f"✅ Elite-only mode: Generated {len(users)} users")
        print(f"   Profiles: {users['profile'].value_counts().to_dict()}")
        print(f"   Average base_km_week: {users['base_km_week'].mean():.1f} km/week")
        print(f"   Average fitness: {users['fitness'].mean():.3f}")
        print(f"   Fitness range: {users['fitness'].min():.3f} - {users['fitness'].max():.3f}")

    # 2) daily plan and activities
    daily_plan = build_daily_plan(cfg, users, rng)
    activities = generate_activities(cfg, users, daily_plan, rng)

    # 3) daily signals
    daily = generate_daily_signals(cfg, users, daily_plan, activities, rng)

    # 4) events + onset labels
    # NOTE: Warmup data is still in daily/activities at this point, which is correct
    # because events.py needs it for spike detection and rolling features
    daily = generate_events(cfg, daily, users, activities, rng)
    
    # 4.25) POST-PROCESSING: Adjust perceived features on injury days to match real CC0 associations
    # Real CC0 shows strong associations: exertion +35.8%, trainingSuccess +31.8%, recovery +26.3% on injury days
    # NOTE: Perceived features are now generated algorithmically in daily.py
    # (removed multiplier-based post-processing to ensure valid algorithmic generation)
    # daily = _adjust_perceived_features_for_injuries(daily, rng)  # REMOVED: Using algorithmic fixes instead

    # 4.5) WARMUP PERIOD: Filter out warmup days from final output
    # Keep warmup data for internal calculations (spike detection, rolling features),
    # but exclude from final output to match requested date range
    warmup_days = int(getattr(cfg, "warmup_days", 0))
    if warmup_days > 0:
        start_date_dt = pd.to_datetime(cfg.start_date)
        daily["date_dt"] = pd.to_datetime(daily["date"])
        daily = daily[daily["date_dt"] >= start_date_dt].copy()
        daily = daily.drop(columns=["date_dt"])
        
        # Also filter activities
        if activities is not None and len(activities) > 0:
            activities["date_dt"] = pd.to_datetime(activities["date"])
            activities = activities[activities["date_dt"] >= start_date_dt].copy()
            activities = activities.drop(columns=["date_dt"])

    # 5) apply illness/injury effects to physiology + activity HR proxies (realism patch)
    # (We only adjust daily physio here; per-activity HR already has drift, and main_model can use daily.)
    daily = daily.sort_values(["user_id","date"]).reset_index(drop=True)
    ill = daily["illness_ongoing"].to_numpy(dtype=int)
    inj = daily["injury_ongoing"].to_numpy(dtype=int)

    # NEW: Enforced rest period during injury recovery (injury_ongoing = 1)
    # During recovery: km_total=0, sessions=0, session_type=None, rest day markers
    if "km_total" in daily.columns:
        daily.loc[inj==1, "km_total"] = 0.0
    if "sessions" in daily.columns:
        daily.loc[inj==1, "sessions"] = 0
    if "session_type" in daily.columns:
        daily.loc[inj==1, "session_type"] = None
    if "perceived_exertion" in daily.columns:
        daily.loc[inj==1, "perceived_exertion"] = -0.01
    if "perceived_trainingSuccess" in daily.columns:
        daily.loc[inj==1, "perceived_trainingSuccess"] = -0.01
    if "perceived_recovery" in daily.columns:
        daily.loc[inj==1, "perceived_recovery"] = -0.01
    
    # Also enforce rest for double sessions
    if "has_double" in daily.columns:
        daily.loc[inj==1, "has_double"] = 0  # Use 0 instead of False for int64 dtype
    if "double_session_type" in daily.columns:
        daily.loc[inj==1, "double_session_type"] = None
    if "double_km" in daily.columns:
        daily.loc[inj==1, "double_km"] = 0.0

    # illness bumps RHR + temp + resp + stress; reduces HRV + sleep and tends to reduce km/duration in real behavior
    if "rhr_bpm" in daily.columns:
        daily.loc[ill==1, "rhr_bpm"] = daily.loc[ill==1, "rhr_bpm"] + cfg.illness_hr_bump
    if "hrv_ms" in daily.columns:
        daily.loc[ill==1, "hrv_ms"] = daily.loc[ill==1, "hrv_ms"] - 8.0
    if "skin_temp_c" in daily.columns:
        daily.loc[ill==1, "skin_temp_c"] = daily.loc[ill==1, "skin_temp_c"] + cfg.temp_illness_bump
    if "resp_rate_rpm" in daily.columns:
        daily.loc[ill==1, "resp_rate_rpm"] = daily.loc[ill==1, "resp_rate_rpm"] + 1.2
    if "stress_score" in daily.columns:
        daily.loc[ill==1, "stress_score"] = daily.loc[ill==1, "stress_score"] + 7.0
    if "sleep_hours" in daily.columns:
        daily.loc[ill==1, "sleep_hours"] = daily.loc[ill==1, "sleep_hours"] - 0.4

    # injury increases stress; also slightly bumps RHR (pain/fatigue), reduces sleep
    # NOTE: km/duration already set to 0 above during enforced rest
    daily.loc[inj==1, "stress_score"] = daily.loc[inj==1, "stress_score"] + 5.0
    daily.loc[inj==1, "sleep_hours"] = daily.loc[inj==1, "sleep_hours"] - 0.2
    if "rhr_bpm" in daily.columns:
        daily.loc[inj==1, "rhr_bpm"] = daily.loc[inj==1, "rhr_bpm"] + 1.8
    
    # NEW: Gradual return-to-training period (after recovery ends)
    # Apply volume and intensity multipliers based on days since recovery ended
    if "days_since_recovery_end" in daily.columns and "return_period_duration" in daily.columns:
        # Filter for days in gradual return period (not injured, within return period)
        in_return_period = (inj == 0) & (daily["days_since_recovery_end"] < daily["return_period_duration"]) & (daily["days_since_recovery_end"] >= 0)
        
        # Process each row in gradual return period
        for idx in daily[in_return_period].index:
            days_since = int(daily.loc[idx, "days_since_recovery_end"])
            return_period = daily.loc[idx, "return_period_duration"]
            
            # Skip if return period is NaN or invalid
            if pd.isna(return_period) or return_period <= 0:
                continue
            
            return_period = int(return_period)
            if return_period <= 0:
                continue
            
            # Calculate progress through return period (0.0 to 1.0)
            progress = float(days_since) / float(return_period - 1) if return_period > 1 else 1.0
            progress = max(0.0, min(1.0, progress))
            
            # Calculate volume multiplier based on return period duration
            if return_period <= 2:  # Short return (Severity 1-2)
                volume_multiplier = _smooth_ramp(progress, cfg.injury_return_volume_start_short, 1.0, 5.0)
            elif return_period <= 10:  # Moderate return (Severity 3-5)
                volume_multiplier = _smooth_ramp(progress, cfg.injury_return_volume_start_moderate, 1.0, cfg.injury_return_ramp_steepness_moderate)
            else:  # Long return (Severity 6-10)
                volume_multiplier = _smooth_ramp(progress, cfg.injury_return_volume_start_long, 1.0, cfg.injury_return_ramp_steepness_long)
            
            # Calculate intensity multiplier (starts later, ramps up)
            intensity_multiplier = 0.0  # Default to 0% high intensity
            if return_period <= 2:  # Short return (Severity 1-2)
                if days_since >= 1:  # Intensity starts Day 2
                    intensity_progress = float(days_since - 1) / float(return_period - 1) if return_period > 1 else 1.0
                    intensity_progress = max(0.0, min(1.0, intensity_progress))
                    intensity_multiplier = _smooth_ramp(intensity_progress, 0.5, 1.0, 5.0)
            elif return_period <= 10:  # Moderate return (Severity 3-5)
                delay_days = int(np.round(return_period * cfg.injury_return_intensity_delay_ratio))
                if days_since >= delay_days:
                    intensity_progress = float(days_since - delay_days) / float(return_period - delay_days) if return_period > delay_days else 1.0
                    intensity_progress = max(0.0, min(1.0, intensity_progress))
                    intensity_multiplier = _smooth_ramp(intensity_progress, 0.0, 1.0, cfg.injury_return_ramp_steepness_moderate)
            else:  # Long return (Severity 6-10)
                delay_days = cfg.injury_return_week_delay_long  # Fixed 1 week delay
                if days_since >= delay_days:
                    intensity_progress = float(days_since - delay_days) / float(return_period - delay_days) if return_period > delay_days else 1.0
                    intensity_progress = max(0.0, min(1.0, intensity_progress))
                    intensity_multiplier = _smooth_ramp(intensity_progress, 0.0, 1.0, cfg.injury_return_ramp_steepness_long)
            
            # Apply volume multiplier to km_total (only on training days)
            if "km_total" in daily.columns and daily.loc[idx, "km_total"] > 0:
                original_km = daily.loc[idx, "km_total"]
                daily.loc[idx, "km_total"] = float(original_km * volume_multiplier)
            
            # Restrict high-intensity sessions based on intensity_multiplier
            if "session_type" in daily.columns and not pd.isna(daily.loc[idx, "session_type"]):
                session_type = str(daily.loc[idx, "session_type"])
                if intensity_multiplier < 0.3:  # Very early return: no high intensity
                    if session_type in ["tempo", "interval"]:
                        # Convert to easy session
                        daily.loc[idx, "session_type"] = "easy"
                        # Also reduce km if it was a high-intensity session
                        if "km_total" in daily.columns:
                            daily.loc[idx, "km_total"] = daily.loc[idx, "km_total"] * 0.8  # Slight reduction for easy vs tempo/interval
                elif intensity_multiplier < 0.7:  # Moderate return: limit high intensity
                    if session_type == "interval":  # Most intense - convert to tempo
                        daily.loc[idx, "session_type"] = "tempo"
            
            # Store multipliers for tracking (optional, for debugging/analysis)
            if "volume_multiplier" not in daily.columns:
                daily["volume_multiplier"] = 1.0
            if "intensity_multiplier" not in daily.columns:
                daily["intensity_multiplier"] = 1.0
            daily.loc[idx, "volume_multiplier"] = volume_multiplier
            daily.loc[idx, "intensity_multiplier"] = intensity_multiplier
        
        # Initialize multiplier columns for days not in return period
        if "volume_multiplier" not in daily.columns:
            daily["volume_multiplier"] = 1.0
        if "intensity_multiplier" not in daily.columns:
            daily["intensity_multiplier"] = 1.0

    # Clip physiology to plausible ranges after bumps
    daily["stress_score"] = daily["stress_score"].clip(5, 100)
    daily["sleep_hours"] = daily["sleep_hours"].clip(3.5, 11.0)
    if "rhr_bpm" in daily.columns:
        daily["rhr_bpm"] = daily["rhr_bpm"].clip(35, 110)
    if "hrv_ms" in daily.columns:
        daily["hrv_ms"] = daily["hrv_ms"].clip(10, 220)
    if "resp_rate_rpm" in daily.columns:
        daily["resp_rate_rpm"] = daily["resp_rate_rpm"].clip(8, 28)
    if "skin_temp_c" in daily.columns:
        daily["skin_temp_c"] = daily["skin_temp_c"].clip(35.8, 38.5)

    # 6) write outputs + metadata
    written = write_outputs(users, daily, activities, out_path, cfg)

    meta = {
        "generated_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": cfg.to_dict(),
        "counts": {
            "n_users": int(users.shape[0]),
            "n_daily_rows": int(daily.shape[0]),
            "n_activities": int(activities.shape[0]),
        },
        "prevalence": {
            "injury_onset_rate": float(daily["injury_onset"].mean()),
            "injury_next_7d_rate": float(daily["injury_next_7d"].mean()),
            "illness_onset_rate": float(daily["illness_onset"].mean()),
            "device_worn_rate": float(daily["device_worn"].mean()),
        },
        "files": {},
    }

    for name, path in written.items():
        meta["files"][name] = {
            "path": str(path),
            "sha256": _sha256_file(Path(path)),
        }

    meta_path = out_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if run_checks:
        report = run_sanity_checks(users, daily, activities)
        report_path = out_path / "sanity_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Extra audit artifacts (CSV/JSON) - helpful when iterating on label realism.
        try:
            from .audit import label_rate_by_decile, label_time_trend

            # 1) label rate over time (weekly)
            if {"date", "injury_next_7d"}.issubset(daily.columns):
                tt = label_time_trend(daily, label_col="injury_next_7d", freq="W")
                tt.to_csv(out_path / "sanity_injury_time_trend_weekly.csv", index=False)

            # 2) label rate by deciles for a few daily drivers (raw, not rolled)
            driver_cols = [
                c
                for c in [
                    "acwr",
                    "training_load",
                    "sleep_hours",
                    "stress_score",
                    "rhr_bpm",
                    "hrv_ms",
                    "wear_7d_rate",
                ]
                if c in daily.columns
            ]
            if driver_cols and "injury_next_7d" in daily.columns:
                dec = label_rate_by_decile(daily, label_col="injury_next_7d", cols=driver_cols, n_bins=10)
                dec.to_csv(out_path / "sanity_label_rate_by_decile_daily.csv", index=False)
        except Exception:
            pass

    return {"metadata_path": str(meta_path), "outputs": written}
