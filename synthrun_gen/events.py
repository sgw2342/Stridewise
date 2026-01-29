from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from .config import GeneratorConfig


def _sample_positive_int(mean: float, sd: float, rng: np.random.Generator, lo: int, hi: int) -> int:
    x = rng.normal(mean, sd)
    x = int(np.clip(round(x), lo, hi))
    return max(lo, x)


def _z_7_28_lagged(x: np.ndarray) -> np.ndarray:
    """Lagged z-score: (mean_7 - mean_28) / std_28 using history up to t-1."""
    s = pd.Series(x)
    m7 = s.shift(1).rolling(7, min_periods=7).mean()
    m28 = s.shift(1).rolling(28, min_periods=28).mean()
    sd28 = s.shift(1).rolling(28, min_periods=28).std()
    z = (m7 - m28) / (sd28 + 1e-6)
    return z.to_numpy(dtype=float)


def _get_acwr_sensitivity(fitness: float, cfg: GeneratorConfig) -> float:
    """HYBRID: Get ACWR sensitivity multiplier based on fitness level.
    Lower fitness = more sensitive to high ACWR (like novices).
    """
    if fitness < 0.4:
        return cfg.acwr_sensitivity_low_fitness
    elif fitness < 0.7:
        return cfg.acwr_sensitivity_mid_fitness
    elif fitness < 0.85:
        return cfg.acwr_sensitivity_high_fitness
    else:
        return cfg.acwr_sensitivity_elite_fitness


def _get_injury_duration_mean(fitness: float, cfg: GeneratorConfig) -> float:
    """HYBRID: Get injury duration mean based on fitness level.
    Lower fitness = longer recovery time.
    NOTE: This is the BASE duration before severity adjustment.
    Actual duration will be calculated from severity using _get_recovery_duration_from_severity.
    """
    base = cfg.injury_dur_mean_base
    if fitness < 0.4:
        return base + cfg.injury_dur_low_fitness_add
    elif fitness < 0.5:
        return base + cfg.injury_dur_mid_low_add
    elif fitness < 0.7:
        return base  # Baseline
    elif fitness < 0.85:
        return base - cfg.injury_dur_high_subtract
    else:
        return base - cfg.injury_dur_elite_subtract


def _calculate_injury_severity(
    acwr: float,
    spike_category: int,
    training_load: float,
    baseline_load: float,
    consecutive_hard_days: int,
    recovery_index: float,
    rest_day_deficit_score: float,
    cfg: GeneratorConfig
) -> float:
    """Calculate injury severity (1-10 scale) based on risk factors at injury onset.
    
    Args:
        acwr: Acute-to-chronic workload ratio
        spike_category: Long run spike category (0=none, 1=small, 2=moderate, 3=large)
        training_load: Current training load (TRIMP)
        baseline_load: Athlete's baseline training load (TRIMP)
        consecutive_hard_days: Number of consecutive hard days
        recovery_index: Recovery index (0.0=poor, 1.0=good)
        rest_day_deficit_score: Rest day deficit score (0.0=no deficit, 1.0=severe deficit)
        cfg: Generator configuration
    
    Returns:
        float: Severity score (1.0-10.0, where 1=mild, 10=worst)
    """
    severity_base = cfg.injury_severity_base
    
    # ACWR contribution (ACWR > 1.0 adds to severity)
    acwr_contribution = max(0.0, (acwr - 1.0) / 0.5) * cfg.injury_severity_acwr_weight
    acwr_contribution = min(acwr_contribution, 2.0)  # Cap at +2 points
    
    # Spike contribution (spike size matters)
    spike_contribution = spike_category * cfg.injury_severity_spike_weight
    # spike_category: 0=0, 1=1.5, 2=3.0, 3=4.5 points
    
    # Training load contribution (load above baseline)
    if baseline_load > 0:
        load_ratio = max(0.0, (training_load - baseline_load) / baseline_load)
        training_load_contribution = load_ratio * cfg.injury_severity_load_weight
        training_load_contribution = min(training_load_contribution, 1.5)  # Cap at +1.5 points
    else:
        training_load_contribution = 0.0
    
    # Consecutive hard days contribution
    consecutive_hard_days_contribution = min(consecutive_hard_days / 3.0, 1.5) * cfg.injury_severity_hard_days_weight
    consecutive_hard_days_contribution = min(consecutive_hard_days_contribution, 1.5)  # Cap at +1.5 points
    
    # Poor recovery contribution
    poor_recovery_contribution = (1.0 - recovery_index) * cfg.injury_severity_recovery_weight
    poor_recovery_contribution = min(poor_recovery_contribution, 1.0)  # Cap at +1.0 points
    
    # Rest day deficit contribution
    rest_day_deficit_contribution = rest_day_deficit_score * cfg.injury_severity_rest_deficit_weight
    rest_day_deficit_contribution = min(rest_day_deficit_contribution, 0.5)  # Cap at +0.5 points
    
    # Total severity (clipped to 1-10 range)
    severity = severity_base + acwr_contribution + spike_contribution + \
               training_load_contribution + consecutive_hard_days_contribution + \
               poor_recovery_contribution + rest_day_deficit_contribution
    
    return float(np.clip(severity, 1.0, 10.0))


def _get_recovery_duration_from_severity(
    severity: float,
    fitness: float,
    injury_resilience: float,
    cfg: GeneratorConfig
) -> int:
    """Calculate recovery duration from severity with fitness/resilience adjustments.
    
    Args:
        severity: Injury severity (1.0-10.0)
        fitness: Athlete fitness level (0.0-1.0)
        injury_resilience: Athlete injury resilience (0.0-1.0)
        cfg: Generator configuration
    
    Returns:
        int: Recovery duration in days (min 2, max 120)
    """
    # Map severity to base recovery duration using exponential formula
    # Formula: recovery_days = base_days * (severity_factor ^ exponent)
    # severity_factor = severity / 5.0  # Normalize to 0.2-2.0
    severity_factor = severity / 5.0
    base_recovery_days = cfg.injury_recovery_base_days
    exponent = cfg.injury_recovery_exponent
    
    # Calculate base recovery duration
    recovery_days = base_recovery_days * (severity_factor ** exponent)
    
    # Apply fitness adjustment
    if fitness < 0.4:
        fitness_adjustment = 1.3  # Low fitness: +30% recovery time
    elif fitness < 0.5:
        fitness_adjustment = 1.15  # Mid-low: +15%
    elif fitness < 0.7:
        fitness_adjustment = 1.0  # Baseline
    elif fitness < 0.85:
        fitness_adjustment = 0.9  # High: -10%
    else:
        fitness_adjustment = 0.75  # Elite: -25%
    
    # Apply resilience adjustment
    # High resilience (1.0) = 0.8x duration, low resilience (0.0) = 1.2x duration
    resilience_adjustment = 1.2 - 0.4 * injury_resilience
    
    # Final duration with adjustments
    recovery_duration = recovery_days * fitness_adjustment * resilience_adjustment
    
    # Clip to min/max bounds
    recovery_duration = np.clip(recovery_duration, cfg.injury_recovery_min_days, cfg.injury_recovery_max_days)
    
    return int(np.round(recovery_duration))


def _get_return_period_from_recovery(
    recovery_duration: int,
    cfg: GeneratorConfig
) -> int:
    """Calculate gradual return period duration from recovery duration.
    
    Uses piecewise linear mapping:
    - Recovery 2 days (severity 1) → Return 1 day
    - Recovery 7 days (severity 3) → Return 3 days
    - Recovery 14 days (severity 5) → Return 10 days
    - Recovery 30 days (severity 7) → Return 17 days
    - Recovery 60 days (severity 9) → Return 24 days
    - Recovery 90 days (severity 10) → Return 28 days (max 4 weeks)
    
    Args:
        recovery_duration: Recovery duration in days
        cfg: Generator configuration
    
    Returns:
        int: Return period duration in days (min 1, max 28)
    """
    recovery_points = cfg.injury_return_recovery_points
    return_points = cfg.injury_return_period_points
    
    # Use numpy interpolation for piecewise linear mapping
    return_days = np.interp(recovery_duration, recovery_points, return_points)
    
    # Clip to min/max bounds
    return_days = np.clip(return_days, cfg.injury_return_min_days, cfg.injury_return_max_days)
    
    return int(np.round(return_days))


def _calculate_rest_day_deficit_score(
    rest_days_last_7: int,
    athlete_rest_frequency: float
) -> float:
    """Calculate rest day deficit score (0.0-1.0) based on actual vs expected rest days.
    
    Args:
        rest_days_last_7: Number of rest days in last 7 days [t-6, t-1]
        athlete_rest_frequency: Athlete's baseline rest day frequency (0.0-1.0)
    
    Returns:
        float: Rest day deficit score (0.0=no deficit, 1.0=severe deficit)
    """
    # Calculate expected rest days per week
    baseline_rest_days = athlete_rest_frequency * 7.0
    
    # Calculate deficit (how many fewer rest days than expected)
    rest_day_deficit = max(0.0, baseline_rest_days - rest_days_last_7)
    
    # Normalize deficit (0.0 = no deficit, 1.0 = severe deficit)
    # Severe deficit = fewer than 1 rest day in last 7 days (or deficit >= baseline)
    if baseline_rest_days > 0:
        rest_day_deficit_score = min(1.0, rest_day_deficit / max(1.0, baseline_rest_days))
    else:
        rest_day_deficit_score = 0.0 if rest_days_last_7 > 0 else 1.0
    
    return float(rest_day_deficit_score)


def generate_events(cfg: GeneratorConfig, daily: pd.DataFrame, users: pd.DataFrame, activities: Optional[pd.DataFrame] = None, rng: np.random.Generator = None) -> pd.DataFrame:
    """Adds illness/injury events and forward labels - HYBRID VERSION.

    Key enhancements:
    - Fitness-based ACWR sensitivity (low fitness = more sensitive)
    - Fitness-based injury duration (low fitness = longer recovery)
    - Absolute load risk for low-fitness runners
    - All features properly lagged (no leakage)

    Key design goal: make **injury_next_7d** forecastable under forward-time splits.
    """

    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)
    
    # Set default rng if not provided
    if rng is None:
        rng = np.random.default_rng(42)
    
    # DEBUG: Initialize debug info list (shared across all users)
    if not hasattr(generate_events, '_debug_spike_info'):
        generate_events._debug_spike_info = []
    
        # HYBRID: Merge users to get fitness, injury_proneness, injury_resilience, and rest_day_frequency for profile-based effects
        user_cols = ["fitness"]
        if "injury_proneness" in users.columns:
            user_cols.append("injury_proneness")
        if "injury_resilience" in users.columns:
            user_cols.append("injury_resilience")
        if "rest_day_frequency" in users.columns:
            user_cols.append("rest_day_frequency")
        users_lookup = users.set_index("user_id")[user_cols]
        daily = daily.merge(
            users_lookup,
            left_on="user_id",
            right_index=True,
            how="left"
        )
        daily["fitness"] = daily["fitness"].fillna(0.5)  # Default if missing
        if "injury_proneness" in daily.columns:
            daily["injury_proneness"] = daily["injury_proneness"].fillna(0.5)  # Default if missing
        if "injury_resilience" in daily.columns:
            daily["injury_resilience"] = daily["injury_resilience"].fillna(0.5)  # Default if missing
        if "rest_day_frequency" in daily.columns:
            daily["rest_day_frequency"] = daily["rest_day_frequency"].fillna(0.4)  # Default to 40% if missing
        else:
            # If not in users, calculate from daily data for each athlete (fallback)
            daily["rest_day_frequency"] = 0.4  # Default to 40%
    
    out_rows = []

    for _, g in daily.groupby("user_id", sort=False):
        g = g.copy().reset_index(drop=True)
        fitness = float(g["fitness"].iloc[0])
        # HYBRID: Get user-level injury proneness (0=resilient, 1=prone)
        injury_proneness = float(g["injury_proneness"].iloc[0]) if "injury_proneness" in g.columns else 0.5
        # HYBRID: Get user-level injury resilience (0=low, 1=high) - affects recovery and baseline risk
        injury_resilience = float(g["injury_resilience"].iloc[0]) if "injury_resilience" in g.columns else 0.5
        # NEW: Get athlete's baseline rest day frequency for rest day deficit calculation
        athlete_rest_frequency = float(g["rest_day_frequency"].iloc[0]) if "rest_day_frequency" in g.columns else 0.4

        # --- Lagged rolling load proxies ---
        tl = pd.Series(g["training_load"].to_numpy(dtype=float))
        acute7 = tl.shift(1).rolling(7, min_periods=7).sum().to_numpy(dtype=float)
        chronic28 = (tl.shift(1).rolling(28, min_periods=28).sum().to_numpy(dtype=float)) / 4.0
        acwr = acute7 / (chronic28 + 1e-6)

        # week-to-week ramp: last 7d vs prior 7d (both lagged)
        prev7 = pd.Series(acute7).shift(7).to_numpy(dtype=float)
        ramp_ratio = acute7 / (prev7 + 1e-6)

        # GARMIN RUNSAFE: Long run spike risk (categorical with specific risk multipliers)
        # Biggest single risk: increasing long run by >10% of max long run in previous 30 days
        # Risk categories from Garmin Runsafe study:
        #   Baseline (0-10%): Reference (no additional risk)
        #   Small Spike (>10-30%): 64% higher risk
        #   Moderate Spike (>30-100%): 52% higher risk
        #   Large Spike (>100%): 128% higher risk
        long_run_spike_risk = np.zeros(len(g), dtype=float)
        long_run_spike_category = np.zeros(len(g), dtype=int)  # 0=baseline, 1=small, 2=moderate, 3=large
        g_dates = pd.to_datetime(g["date"]).values
        g_session_type = g["session_type"].values
        g_km_total = g["km_total"].values
        
        for i, date in enumerate(g_dates):
            # Current day's long run distance (if it's a long run day)
            if i < len(g) and str(g_session_type[i]) == "long" and g_km_total[i] > 0:
                current_long_km = float(g_km_total[i])
                
                # Average long run in previous 30 days (excluding today, using lagged data)
                # Compare to average instead of max to avoid comparing spikes with spikes
                # Look back at previous days in this group
                prev_indices = [j for j in range(i) if 
                               (date - pd.to_datetime(g_dates[j])).days <= 30 and
                               (date - pd.to_datetime(g_dates[j])).days > 0 and
                               str(g_session_type[j]) == "long" and
                               g_km_total[j] > 0]
                
                # IMPROVED: More lenient detection - require at least 1 previous long run (was already this)
                # But also check if we have enough history, and if not, use a more lenient threshold
                if len(prev_indices) > 0:
                    prev_long_distances = [float(g_km_total[j]) for j in prev_indices]
                    # Use average instead of max to avoid spike-to-spike comparison
                    avg_prev_long = np.mean(prev_long_distances)
                    # Calculate percentage increase
                    if avg_prev_long > 0:
                        pct_increase = ((current_long_km / avg_prev_long) - 1.0) * 100.0
                        
                        # IMPROVED: Lower threshold for early detection (if few previous runs)
                        # If we have < 3 previous long runs, use 5% threshold instead of 10%
                        min_threshold = 5.0 if len(prev_indices) < 3 else 10.0
                        
                        # Categorize based on Garmin Runsafe study
                        # Risk multipliers: "X% higher risk" means (1 + X/100) × baseline
                        if pct_increase <= min_threshold:
                            # Baseline: 0-min_threshold% increase = Reference (no additional risk)
                            long_run_spike_category[i] = 0
                            long_run_spike_risk[i] = 0.0  # No additional risk
                        elif pct_increase <= 30.0:
                            # Small Spike: >min_threshold-30% = 64% higher risk = 1.64x baseline
                            long_run_spike_category[i] = 1
                            long_run_spike_risk[i] = 1.64  # 64% higher risk = 1.64x total (was 0.64 - BUG FIX)
                        elif pct_increase <= 100.0:
                            # Moderate Spike: >30-100% = 52% higher risk = 1.52x baseline
                            long_run_spike_category[i] = 2
                            long_run_spike_risk[i] = 1.52  # 52% higher risk = 1.52x total (was 0.52 - BUG FIX)
                        else:
                            # Large Spike: >100% = 128% higher risk = 2.28x baseline
                            long_run_spike_category[i] = 3
                            long_run_spike_risk[i] = 2.28  # 128% higher risk = 2.28x total (was 1.28 - BUG FIX)
                elif i > 0:
                    # IMPROVED: If no previous long runs but we have some history,
                    # compare to the most recent long run (if any exists)
                    prev_long_indices = [j for j in range(i) if 
                                        str(g_session_type[j]) == "long" and
                                        g_km_total[j] > 0]
                    if len(prev_long_indices) > 0:
                        # Use the most recent long run
                        most_recent_idx = prev_long_indices[-1]
                        most_recent_long = float(g_km_total[most_recent_idx])
                        if most_recent_long > 0:
                            pct_increase = ((current_long_km / most_recent_long) - 1.0) * 100.0
                            # Use 5% threshold for single comparison
                            if pct_increase > 5.0:
                                if pct_increase <= 30.0:
                                    long_run_spike_category[i] = 1
                                    long_run_spike_risk[i] = 1.64  # BUG FIX: was 0.64
                                elif pct_increase <= 100.0:
                                    long_run_spike_category[i] = 2
                                    long_run_spike_risk[i] = 1.52  # BUG FIX: was 0.52
                                else:
                                    long_run_spike_category[i] = 3
                                    long_run_spike_risk[i] = 2.28  # BUG FIX: was 1.28

        # HI share over last 7 days (lagged) - INTENSITY ZONES (Z3-4, Z5-T1-T2)
        # NOTE: Reduced weight (w_hi = 0.3) - intensity zones are WEAK in real data (1.06x-1.20x, non-significant)
        hi_km = (
            g["kms_z3_4"].to_numpy(dtype=float)
            + g["kms_z5_t1_t2"].to_numpy(dtype=float)
        )
        tot_km = g["km_total"].to_numpy(dtype=float)
        hi7 = pd.Series(hi_km).shift(1).rolling(7, min_periods=7).sum().to_numpy(dtype=float)
        km7 = pd.Series(tot_km).shift(1).rolling(7, min_periods=7).sum().to_numpy(dtype=float)
        # VERSION 4: Configurable HI share clipping
        hi_share_clip_max = getattr(cfg, 'hi_share_clip_max', 1.0)  # Default: 1.0
        hi_share7 = np.clip(hi7 / (km7 + 1e-6), 0.0, float(hi_share_clip_max))
        
        # NEW: SPRINTING RISK - #1 injury driver in real data (1.92x ratio, +91.9%, p<0.001)
        # Sprinting is the STRONGEST injury signal in real CC0 data
        # Real data shows sprinting at day -1 (before injury) is 1.92x higher than non-injury days
        # Calculate sprinting intensity: sprinting km per day (lagged)
        sprint_km = g["kms_sprinting"].to_numpy(dtype=float) if "kms_sprinting" in g.columns else np.zeros(len(g), dtype=float)
        # Sprinting in last 7 days (lagged) - cumulative effect
        sprint7 = pd.Series(sprint_km).shift(1).rolling(7, min_periods=7).sum().to_numpy(dtype=float)
        # Sprinting share: sprint km / total km (lagged) - proportion of training that's sprinting
        sprint_share7 = np.clip(sprint7 / (km7 + 1e-6), 0.0, 1.0)
        # Recent sprinting (last 1-2 days) - immediate spikes are most dangerous
        # Focus on day t-1 (day before injury) as real data shows strongest signal there
        sprint_recent_1d = pd.Series(sprint_km).shift(1).fillna(0.0).to_numpy(dtype=float)  # Day t-1 (before injury)
        sprint_recent_2d = pd.Series(sprint_km).shift(1).rolling(2, min_periods=1).sum().to_numpy(dtype=float)  # Last 2 days
        # STRENGTHENED: Sprinting risk should be very high when recent sprinting occurs
        # Real data: 1.92x ratio means ~2x more sprinting at injury vs non-injury
        # Scale sprinting risk to be proportional to sprinting amount (stronger signal)
        # Recent 1-day sprinting (immediate, most important): scale by 20x to make small values significant
        sprinting_risk_immediate = np.clip(sprint_recent_1d * 20.0, 0.0, 4.0)  # 0.2km sprinting = 4.0 risk
        # Recent 2-day sprinting (cumulative): scale by 10x
        sprinting_risk_recent = np.clip(sprint_recent_2d * 10.0, 0.0, 3.0)  # 0.3km sprinting = 3.0 risk
        # 7-day share (proportion): already 0-1, scale by 3x
        sprinting_risk_share = sprint_share7 * 3.0
        # STRENGTHENED: Combined sprinting risk - prioritize immediate spikes even more
        # Real data shows strongest signal at day -1 (day before injury), so immediate sprinting is critical
        # Increased immediate weight from 50% to 70% to match real data pattern
        sprinting_risk = 0.7 * sprinting_risk_immediate + 0.2 * sprinting_risk_recent + 0.1 * sprinting_risk_share
        sprinting_risk = np.clip(sprinting_risk, 0.0, 5.0)  # Increased cap from 4.0 to 5.0 to allow stronger signals
        
        # NEW: ABSOLUTE SPRINTING RISK - Direct additive effect on p_inj (similar to spike absolute risk)
        # Sprinting is the #1 injury driver in real data (1.92x ratio)
        # Need direct additive risk to ensure injuries happen on days AFTER sprinting sessions
        # Calculate absolute risk based on sprinting amount (km) on day t-1 (before injury)
        sprinting_absolute_risk = np.zeros(len(g), dtype=float)
        
        # PHASE 3: Check if user is advanced/elite and use reduced risk value
        is_advanced_or_elite = False
        if "profile" in g.columns:
            profile_str = str(g["profile"].iloc[0]).lower() if len(g) > 0 else "recreational"
            is_advanced_or_elite = profile_str in ["advanced", "elite"]
        elif "fitness" in g.columns:
            fitness_val = float(g["fitness"].iloc[0]) if len(g) > 0 else 0.5
            is_advanced_or_elite = fitness_val >= 0.7
        
        # Use reduced risk for advanced/elite, default for others
        if is_advanced_or_elite:
            sprinting_risk_per_km = getattr(cfg, 'sprinting_absolute_risk_per_km_advanced_elite', 0.062)  # PHASE 3: Reduced for advanced/elite
        else:
            sprinting_risk_per_km = getattr(cfg, 'sprinting_absolute_risk_per_km', 0.1381)  # Default for novice/recreational
        
        persistence_days = getattr(cfg, 'sprinting_risk_persistence_days', 3)  # PHASE 2: Increased to 3 days (from 2)
        
        for i in range(len(g)):
            # SPRINTING INJURY ASSOCIATION FIX: Strengthen association by increasing sprinting risk magnitude
            # Real CC0 shows injury days have 66.9% more sprinting (t-7), suggesting injuries occur when sprinting is present
            # Issue: Current sprinting_absolute_risk might be too low, so injuries occur but sprinting association is weak
            # Fix: Increase sprinting_absolute_risk_per_km to make sprinting a stronger driver
            # This will ensure that when sprinting happens, injuries are more likely, creating the association
            sprint_km_current = float(sprint_km[i]) if i < len(sprint_km) else 0.0
            if sprint_km_current > 0:
                # Calculate total risk for this sprinting session
                # Note: sprinting_risk_per_km is already increased in config (0.20 for advanced/elite)
                # No additional multiplier needed - the increased risk_per_km should be sufficient
                total_sprint_abs_risk = sprint_km_current * sprinting_risk_per_km
                
                # Apply same-day risk (day 0) - keep at 45% for balance
                same_day_risk_weight = getattr(cfg, 'same_day_risk_weight', 0.45)  # Default: 45% (optimal balance)
                same_day_risk = total_sprint_abs_risk * same_day_risk_weight
                sprinting_absolute_risk[i] += same_day_risk
                
                # Apply remaining persistent risk (distributed over next 3 days)
                persistent_risk_weight = 1.0 - same_day_risk_weight  # 55% (distributed)
                persistent_risk = total_sprint_abs_risk * persistent_risk_weight
                for days_after in range(1, persistence_days + 1):
                    if i + days_after < len(sprinting_absolute_risk):
                        # Distribute persistent risk evenly over 3 days
                        daily_persistent_risk = persistent_risk / float(persistence_days)
                        sprinting_absolute_risk[i + days_after] = max(
                            sprinting_absolute_risk[i + days_after],
                            daily_persistent_risk
                        )
            
            # Also check sprinting on day t-1 (legacy logic for backward compatibility)
            if i > 0:
                sprint_km_t_minus_1 = float(sprint_km[i-1]) if i-1 < len(sprint_km) else 0.0
                if sprint_km_t_minus_1 > 0 and sprint_km_current == 0:  # Only if no same-day sprinting
                    # Calculate absolute risk: proportional to sprinting amount
                    base_sprint_abs_risk = sprint_km_t_minus_1 * sprinting_risk_per_km
                    # Apply persistence: risk persists for persistence_days after sprinting session
                    # PHASE 3 (ADJUSTED): Stronger decay curve for better signal strength
                    for days_after in range(1, persistence_days + 1):
                        if i + days_after < len(sprinting_absolute_risk):
                            # Adjusted decay curve: stronger signals for better model detection
                            if days_after == 1:
                                decay_multiplier = 0.90  # Day 1: 90% risk (increased from 0.85)
                            elif days_after == 2:
                                decay_multiplier = 0.70  # Day 2: 70% risk (increased from 0.50)
                            else:
                                decay_multiplier = 0.50  # Day 3+: 50% risk (same)
                            decayed_risk = base_sprint_abs_risk * decay_multiplier
                            sprinting_absolute_risk[i + days_after] = max(
                                sprinting_absolute_risk[i + days_after],
                                decayed_risk
                            )
        
        # REMOVED: Cap on sprinting absolute risk to allow proportional scaling
        # Risk is now proportional to sprinting volume (0.50 per km)
        # Will be capped at sprinting_absolute_risk_clip_max (default 40%) when added to p_inj
        # sprinting_absolute_risk remains uncapped
        
        # NEW: Training load spikes (weekly spikes, not just long runs)
        # Compare current week's load to previous 4 weeks' average
        load_week = pd.Series(acute7).shift(1).rolling(7, min_periods=7).mean().to_numpy(dtype=float)  # Weekly average load
        load_week_prev4 = pd.Series(load_week).shift(7).rolling(28, min_periods=28).mean().to_numpy(dtype=float)  # Previous 4 weeks average
        training_load_spike_risk = np.zeros(len(g), dtype=float)
        for i in range(len(g)):
            if i >= 28 and load_week_prev4[i] > 0:  # Need at least 4 weeks of history
                pct_increase = ((load_week[i] / load_week_prev4[i]) - 1.0) * 100.0
                if pct_increase > 20.0:  # >20% increase in weekly load
                    # VERSION 4: Configurable training load spike clipping
                    training_load_spike_clip_max = getattr(cfg, 'training_load_spike_clip_max', 2.0)  # Default: 2.0
                    training_load_spike_risk[i] = np.clip(pct_increase / 50.0, 0.0, float(training_load_spike_clip_max))  # Scale: 20% = 0.4, 50% = 1.0, 100% = 2.0
        
        # NEW: Consecutive hard days (2+ hard sessions in a row)
        # EXPONENTIAL SCALING: Each extra day adds significantly more risk than the previous jump
        # Hard sessions: tempo, interval (not easy, long, rest)
        g_session_type = g["session_type"].values
        is_hard = np.array([str(s) in ["tempo", "interval"] for s in g_session_type], dtype=int)
        consecutive_hard_days = np.zeros(len(g), dtype=int)
        for i in range(1, len(g)):
            if is_hard[i] == 1:
                consecutive_hard_days[i] = consecutive_hard_days[i-1] + 1 if is_hard[i-1] == 1 else 1
            else:
                consecutive_hard_days[i] = 0
        # RECOMMENDATION 2: STRENGTHENED - Exponential scaling with steeper curve
        # Formula: risk = 1.75 ^ (days_over_threshold - 1) (increased from 1.5)
        # Day 2 (first over threshold, days_over_threshold=1): 1.75^0 = 1.0
        # Day 3 (days_over_threshold=2): 1.75^1 = 1.75 (75% more than day 2, was 50%)
        # Day 4 (days_over_threshold=3): 1.75^2 = 3.06 (75% more than day 3, was 50%)
        # Day 5 (days_over_threshold=4): 1.75^3 = 5.36 (75% more than day 4, was 50%)
        consecutive_hard_risk = np.zeros(len(g), dtype=float)
        threshold = 2  # Hardcoded: 2+ consecutive hard days is risky
        base_exponent = 1.75  # RECOMMENDATION 2: Increased from 1.5 to 1.75 for steeper risk curve
        for i in range(len(g)):
            if consecutive_hard_days[i] >= threshold:
                days_over_threshold = consecutive_hard_days[i] - threshold + 1
                # Exponential: base^exponent, where base=1.75 (strengthened)
                exponential_risk = base_exponent ** (days_over_threshold - 1)
                # VERSION 4: Configurable consecutive hard days clipping
                consecutive_hard_days_clip_max = getattr(cfg, 'consecutive_hard_days_clip_max', 10.0)  # Default: 10.0
                consecutive_hard_risk[i] = float(np.clip(exponential_risk, 0.0, float(consecutive_hard_days_clip_max)))  # Cap for safety
            else:
                consecutive_hard_risk[i] = 0.0
        
        # NEW: Insufficient recovery between hard sessions (48h rule)
        # Hard sessions should have at least 48h (2 days) between them
        insufficient_recovery_risk = np.zeros(len(g), dtype=float)
        last_hard_day = -999  # Days since last hard session
        for i in range(len(g)):
            if is_hard[i] == 1:
                if last_hard_day >= 0 and last_hard_day < 2:  # Hard session within 48h
                    # Risk increases as recovery time decreases
                    insufficient_recovery_risk[i] = (2.0 - last_hard_day) / 2.0  # 0 days = 1.0, 1 day = 0.5
                last_hard_day = 0
            else:
                if last_hard_day >= 0:
                    last_hard_day += 1
        
        # NEW: Training load distribution (too much in single session vs. spread out)
        # High risk if single session is >40% of weekly load
        load_distribution_risk = np.zeros(len(g), dtype=float)
        for i in range(len(g)):
            if i >= 7:  # Need at least 7 days of history
                current_load = g["training_load"].iloc[i] if i < len(g) else 0.0
                weekly_load = acute7[i] if i < len(acute7) else 0.0
                if weekly_load > 0:
                    single_session_share = current_load / weekly_load
                    if single_session_share > 0.40:  # >40% in single session
                        # VERSION 4: Configurable load distribution clipping
                        load_distribution_clip_max = getattr(cfg, 'load_distribution_clip_max', 2.0)  # Default: 2.0
                        load_distribution_risk[i] = np.clip((single_session_share - 0.40) / 0.30, 0.0, float(load_distribution_clip_max))  # 40% = 0, 50% = 0.33, 70% = 1.0
        
        # RECOMMENDATION 2: High intensity sessions per week risk (3+, 4+, 5+ sessions per week)
        # Count hard sessions (tempo, interval) in rolling 7-day window (lagged - exclude current day)
        # Use pandas rolling window with shift(1) for proper lagging
        is_hard_series = pd.Series(is_hard, index=g.index)
        # Shift by 1 to exclude current day, then calculate rolling sum
        hard_sessions_count_7d_series = is_hard_series.shift(1).fillna(0).rolling(7, min_periods=1).sum()
        hard_sessions_count_7d = hard_sessions_count_7d_series.fillna(0).astype(int).values
        
        # RECOMMENDATION 2: Exponential scaling for high intensity session count
        # Risk increases exponentially with session count (threshold at 3+ sessions)
        # Formula: risk = 1.6 ^ (sessions_over_threshold)
        # 3 sessions (at threshold): 1.6^0 = 1.0
        # 4 sessions: 1.6^1 = 1.6 (60% more than 3)
        # 5 sessions: 1.6^2 = 2.56 (60% more than 4)
        # 6 sessions: 1.6^3 = 4.10 (60% more than 5)
        high_intensity_count_risk = np.zeros(len(g), dtype=float)
        hi_threshold = 3  # Hardcoded: 3+ hard sessions per week is risky
        hi_base_exponent = 1.6  # Exponential base for high intensity sessions
        for i in range(len(g)):
            if hard_sessions_count_7d[i] >= hi_threshold:
                sessions_over_threshold = hard_sessions_count_7d[i] - hi_threshold
                # Exponential: base^exponent
                exponential_risk = hi_base_exponent ** sessions_over_threshold
                # Clip for safety (cap at 5.0, which is ~6 sessions per week)
                high_intensity_count_risk[i] = float(np.clip(exponential_risk, 0.0, 5.0))
            else:
                high_intensity_count_risk[i] = 0.0

        # --- Recovery / physiology deviations (lagged) ---
        sleep_z = _z_7_28_lagged(g["sleep_hours"].to_numpy(dtype=float))
        rhr_z = _z_7_28_lagged(g["rhr_bpm"].to_numpy(dtype=float))
        hrv_z = _z_7_28_lagged(g["hrv_ms"].to_numpy(dtype=float))
        stress_z = _z_7_28_lagged(g["stress_score"].to_numpy(dtype=float))

        # component transforms (directional)
        # VERSION 4: Configurable risk component clipping
        acwr_excess_clip_max = getattr(cfg, 'acwr_excess_clip_max', 2.5)  # Default: 2.5
        ramp_excess_clip_max = getattr(cfg, 'ramp_excess_clip_max', 2.5)  # Default: 2.5
        sleep_def_clip_max = getattr(cfg, 'sleep_def_clip_max', 3.0)  # Default: 3.0
        hrv_drop_clip_max = getattr(cfg, 'hrv_drop_clip_max', 3.0)  # Default: 3.0
        rhr_rise_clip_max = getattr(cfg, 'rhr_rise_clip_max', 3.0)  # Default: 3.0
        stress_rise_clip_max = getattr(cfg, 'stress_rise_clip_max', 3.0)  # Default: 3.0
        
        acwr_excess = np.clip(acwr - float(cfg.injury_acwr_thresh), 0.0, float(acwr_excess_clip_max))
        ramp_excess = np.clip(ramp_ratio - float(cfg.injury_ramp_thresh), 0.0, float(ramp_excess_clip_max))
        sleep_def = np.clip(-sleep_z, 0.0, float(sleep_def_clip_max))
        hrv_drop = np.clip(-hrv_z, 0.0, float(hrv_drop_clip_max))
        rhr_rise = np.clip(rhr_z, 0.0, float(rhr_rise_clip_max))
        stress_rise = np.clip(stress_z, 0.0, float(stress_rise_clip_max))

        # HYBRID: Apply fitness-based ACWR sensitivity
        acwr_sensitivity = _get_acwr_sensitivity(fitness, cfg)
        acwr_risk = float(cfg.w_acwr) * acwr_excess * acwr_sensitivity

        # raw risk (0..~4)
        # GARMIN RUNSAFE: Long run spike risk (biggest single risk factor)
        # FIXED: Apply spike risk multiplicatively instead of additively
        # This gives direct 1.64x, 1.52x, 2.28x injury rate multipliers
        # Interactions still apply but as multipliers on the spike multiplier
        has_spike = long_run_spike_risk > 0
        
        # Base spike multiplier (from Garmin Runsafe study)
        # 1.64 = 64% higher risk (small spike), 1.52 = 52% higher (moderate), 2.28 = 128% higher (large)
        # FIXED: Use long_run_spike_risk directly - it's already the correct multiplier (1.64, 1.52, 2.28)
        # For no-spike days, use 1.0 (no multiplier)
        spike_base_multiplier = np.where(has_spike, long_run_spike_risk, 1.0)
        
        # Add back interactions with smaller multipliers (to preserve base effect)
        # Since we're multiplicative, interactions should be modest to avoid over-amplification
        proneness_mult = np.where(has_spike, 1.0 + 0.1 * injury_proneness, 1.0)  # 1.0x to 1.1x
        fitness_mult = np.where(has_spike, 1.0 + 0.1 * (1.0 - fitness), 1.0)  # 1.0x to 1.1x
        sleep_mult = np.where(has_spike, 1.0 + 0.05 * np.clip(sleep_def / 3.0, 0.0, 1.0), 1.0)  # 1.0x to 1.05x
        acwr_mult = np.where(has_spike, 1.0 + 0.05 * np.clip(acwr_excess / 2.5, 0.0, 1.0), 1.0)  # 1.0x to 1.05x
        fatigue_proxy = np.clip(
            (acwr_risk + float(cfg.w_sleep) * sleep_def + float(cfg.w_hrv) * hrv_drop) / 3.0,
            0.0, 1.0
        )
        fatigue_mult = np.where(has_spike, 1.0 + 0.05 * fatigue_proxy, 1.0)  # 1.0x to 1.05x
        poor_recovery_proxy = np.clip((sleep_def + hrv_drop) / 6.0, 0.0, 1.0)
        recovery_mult = np.where(has_spike, 1.0 + 0.05 * poor_recovery_proxy, 1.0)  # 1.0x to 1.05x
        
        # Combined interaction multiplier (applied to spike_base_multiplier)
        # With small interactions, total multiplier should be close to base (1.64, 1.52, 2.28) but slightly higher
        spike_interaction_multiplier = proneness_mult * fitness_mult * sleep_mult * acwr_mult * fatigue_mult * recovery_mult
        spike_multiplier = spike_base_multiplier * spike_interaction_multiplier
        
        # NEW APPROACH: Use ABSOLUTE spike risk instead of multiplier
        # This adds a fixed probability to p_inj, regardless of base_hazard or risk_score
        # This should work better because it doesn't depend on risk_score being high
        # Calculate absolute risk based on spike category
        # PHASE 3: Check if user is advanced/elite and use reduced spike risk values
        is_advanced_or_elite_spike = False
        if "profile" in g.columns:
            profile_str = str(g["profile"].iloc[0]).lower() if len(g) > 0 else "recreational"
            is_advanced_or_elite_spike = profile_str in ["advanced", "elite"]
        elif "fitness" in g.columns:
            fitness_val = float(g["fitness"].iloc[0]) if len(g) > 0 else 0.5
            is_advanced_or_elite_spike = fitness_val >= 0.7
        
        # Use reduced risk values for advanced/elite, default for others
        if is_advanced_or_elite_spike:
            spike_risk_small = getattr(cfg, 'spike_absolute_risk_small_advanced_elite', 0.01934)  # PHASE 3: Reduced
            spike_risk_moderate = getattr(cfg, 'spike_absolute_risk_moderate_advanced_elite', 0.02210)  # PHASE 3: Reduced
            spike_risk_large = getattr(cfg, 'spike_absolute_risk_large_advanced_elite', 0.02763)  # PHASE 3: Reduced
        else:
            spike_risk_small = getattr(cfg, 'spike_absolute_risk_small', 0.03867)  # Default
            spike_risk_moderate = getattr(cfg, 'spike_absolute_risk_moderate', 0.04420)  # Default
            spike_risk_large = getattr(cfg, 'spike_absolute_risk_large', 0.05525)  # Default
        
        spike_absolute_risk = np.zeros(len(g), dtype=float)
        for i in range(len(g)):
            if long_run_spike_category[i] == 1:  # Small spike
                spike_absolute_risk[i] = float(spike_risk_small)
            elif long_run_spike_category[i] == 2:  # Moderate spike
                spike_absolute_risk[i] = float(spike_risk_moderate)
            elif long_run_spike_category[i] == 3:  # Large spike
                spike_absolute_risk[i] = float(spike_risk_large)
            else:
                spike_absolute_risk[i] = 0.0
        
        # PHASE 2: Spike persistence using config parameter (default: 3 days)
        # Day 0 (spike day): 100% strength
        # Day 1: 100% strength (full risk persists)
        # Days 2+: Exponential decay
        spike_persistence_days = getattr(cfg, 'spike_risk_persistence_days', 3)  # PHASE 2: Configurable persistence
        spike_absolute_risk_persistent = spike_absolute_risk.copy()
        decay_rate = 0.85  # 15% decay per day after day 1
        for i in range(len(spike_absolute_risk)):
            if spike_absolute_risk[i] > 0:  # If spike on day i
                base_risk = spike_absolute_risk[i]
                # Day 0 (spike day): already has full risk
                # Day 1: Full strength (100%)
                if i < len(spike_absolute_risk) - 1:
                    spike_absolute_risk_persistent[i+1] = max(spike_absolute_risk_persistent[i+1], base_risk)
                # Days 2+: Exponential decay up to persistence_days
                for days_after in range(2, spike_persistence_days + 1):
                    if i < len(spike_absolute_risk) - days_after:
                        decay_multiplier = decay_rate ** (days_after - 1)  # Day 2: 0.85, Day 3: 0.72, etc.
                        decayed_risk = base_risk * decay_multiplier
                        spike_absolute_risk_persistent[i+days_after] = max(
                            spike_absolute_risk_persistent[i+days_after], 
                            decayed_risk
                        )
        spike_absolute_risk = spike_absolute_risk_persistent
        
        # Store spike_multiplier for use in injury probability calculation
        # (We'll apply it multiplicatively to p_inj, not additively to risk_raw)
        
        # EXPERIMENT: Feature flag to control spike risk contribution to risk_raw
        # If spike_add_to_risk_raw=True: Spike risk is added to risk_raw (affects risk_score via smoothing)
        # If spike_add_to_risk_raw=False: Spike risk is ONLY additive to p_inj, NOT to risk_raw
        spike_additive_risk = np.zeros(len(g), dtype=float)
        if getattr(cfg, 'spike_add_to_risk_raw', True):  # Default: True (production behavior)
            # FIX: Add spike effect to risk_raw (additive) so it affects risk_score immediately
            # This addresses the timing issue where risk_score is lagged but spike is immediate
            # INCREASED: Make spike_additive_risk much stronger to overcome lagged risk_score
            # Convert spike multiplier to additive risk: spike_mult = 1.64 means add significant risk
            # Formula: additive_risk = (spike_mult - 1.0) * spike_risk_weight
            spike_risk_weight = 3.0  # INCREASED from 1.5 to 3.0 - make spike effect much stronger
            spike_additive_risk = np.where(has_spike, (long_run_spike_risk - 1.0) * spike_risk_weight, 0.0)
            # Small spike (1.64): adds 1.92, Moderate (1.52): adds 1.56, Large (2.28): adds 3.84
        
        risk_raw = (
            acwr_risk  # Already includes sensitivity multiplier
            + float(cfg.w_ramp) * ramp_excess
            + float(cfg.w_hi) * hi_share7  # REDUCED WEIGHT: Intensity zones are weak in real data (0.1x weight)
            # REMOVED: Sprinting from risk_raw - now handled via absolute risk (like spikes)
            # Sprinting absolute risk is added directly to p_inj, not through risk_raw
            # EXPERIMENT: Conditionally add spike effect to risk_raw based on feature flag
            + spike_additive_risk
            + float(cfg.w_sleep) * sleep_def
            + float(cfg.w_hrv) * hrv_drop
            + float(cfg.w_rhr) * rhr_rise
            + float(cfg.w_stress) * stress_rise
            # NEW: Additional validated risk factors
            + float(cfg.w_training_load_spike) * training_load_spike_risk
            + float(cfg.w_consecutive_hard_days) * consecutive_hard_risk
            + float(cfg.w_insufficient_recovery) * insufficient_recovery_risk
            + float(cfg.w_load_distribution) * load_distribution_risk
            + float(getattr(cfg, 'w_high_intensity_count', 1.2)) * high_intensity_count_risk  # RECOMMENDATION 2: High intensity sessions per week
        )
        
        # HYBRID: Absolute load risk for low-fitness runners
        # REMOVED: Absolute load (total km) is NON-SIGNIFICANT in real data (1.01x ratio, p=0.75)
        # Real injuries are driven by intensity spikes (sprinting) and relative load (ACWR), not absolute volume
        # Only apply if slope > 0 (allows disabling via config - currently set to 0.0)
        absolute_load_risk_clip_max = getattr(cfg, 'absolute_load_risk_clip_max', 2.0)  # Default: 2.0
        if cfg.absolute_load_risk_slope > 0 and fitness < cfg.absolute_load_risk_fitness_threshold:
            excess_load = np.clip(
                (acute7 - cfg.absolute_load_risk_threshold) / cfg.absolute_load_risk_threshold,
                0.0, float(absolute_load_risk_clip_max)
            )
            absolute_load_risk = cfg.absolute_load_risk_slope * excess_load
            risk_raw = risk_raw + absolute_load_risk
        
        risk_raw = np.where(np.isfinite(risk_raw), risk_raw, 0.0)
        # EXPERIMENT: Configurable risk_raw clipping
        # Production: 6.0 (clips risk_raw to 0.0-6.0)
        # Experiment: None or very high value for no clipping
        # This affects how high risk_score can get, which directly impacts p_inj
        risk_raw_clip_max = getattr(cfg, 'risk_raw_clip_max', 6.0)  # Default: 6.0 (production)
        if risk_raw_clip_max is not None and risk_raw_clip_max > 0:
            risk_raw = np.clip(risk_raw, 0.0, float(risk_raw_clip_max))
        # If risk_raw_clip_max is None or <= 0, no clipping is applied

        # persistent latent fatigue state (forecastable)
        # FIX: On spike days, use non-lagged ACWR to make risk_score respond immediately
        # This addresses the timing issue where lagged data makes risk_score low on spike days
        risk_score = np.zeros(len(g), dtype=float)
        alpha = float(np.clip(cfg.injury_fatigue_alpha, 0.0, 0.999))  # Now 0.6 for better responsiveness
        noise_sd = float(max(0.0, cfg.injury_risk_noise_sd))
        
        # Calculate non-lagged ACWR for spike days (use current day's data, not lagged)
        # Recalculate acute7 and chronic28 without shift(1) for spike days
        acute7_nonlagged = tl.rolling(7, min_periods=7).sum().to_numpy(dtype=float)  # No shift
        chronic28_nonlagged = (tl.rolling(28, min_periods=28).sum().to_numpy(dtype=float)) / 4.0  # No shift
        acwr_nonlagged = acute7_nonlagged / (chronic28_nonlagged + 1e-6)
        
        # Recalculate ACWR risk with non-lagged ACWR on spike days
        # VERSION 4: Use configurable ACWR excess clipping for non-lagged version too
        acwr_excess_nonlagged = np.clip(acwr_nonlagged - float(cfg.injury_acwr_thresh), 0.0, float(acwr_excess_clip_max))
        acwr_risk_nonlagged = float(cfg.w_acwr) * acwr_excess_nonlagged * acwr_sensitivity
        
        # Use non-lagged ACWR risk on spike days, regular on others
        risk_raw_spike_adjusted = risk_raw.copy()
        for t in range(len(g)):
            if has_spike[t]:
                # Replace ACWR component with non-lagged version
                # Remove old ACWR contribution and add new one
                risk_raw_spike_adjusted[t] = risk_raw[t] - acwr_risk[t] + acwr_risk_nonlagged[t]
        
        for t in range(len(g)):
            prev = risk_score[t - 1] if t > 0 else 0.0
            # Use spike-adjusted risk_raw on spike days, regular risk_raw otherwise
            raw = float(risk_raw_spike_adjusted[t]) if has_spike[t] else float(risk_raw[t])
            risk_score[t] = alpha * prev + (1.0 - alpha) * raw + float(rng.normal(0.0, noise_sd))
            if not np.isfinite(risk_score[t]):
                risk_score[t] = prev
            # EXPERIMENT: Configurable risk_score clipping
            # Production: 4.0 (clips risk_score to 0.0-4.0) - PRIMARY rate controller
            # Experiment: None or very high value for no clipping
            # This directly limits p_inj via: p_inj = base_hazard * (1.0 + 8.0 * risk_score) * factors
            risk_score_clip_max = getattr(cfg, 'risk_score_clip_max', 4.0)  # Default: 4.0 (production)
            if risk_score_clip_max is not None and risk_score_clip_max > 0:
                risk_score[t] = float(np.clip(risk_score[t], 0.0, float(risk_score_clip_max)))
            # If risk_score_clip_max is None or <= 0, no clipping is applied

        # illness baseline hazard with slight winter seasonality
        day_of_year = pd.to_datetime(g["date"]).dt.dayofyear.to_numpy()
        season = 0.25 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0) + 0.75  # ~0.5..1.0
        p_ill = np.clip(float(cfg.illness_hazard) * season, 0.0, 0.06)

        injury_onset = np.zeros(len(g), dtype=int)
        injury_ongoing = np.zeros(len(g), dtype=int)
        illness_onset = np.zeros(len(g), dtype=int)
        illness_ongoing = np.zeros(len(g), dtype=int)
        
        # NEW: Arrays to store injury severity and duration information
        injury_severity = np.full(len(g), np.nan, dtype=float)  # Severity at injury onset, NaN if no injury
        recovery_duration = np.full(len(g), np.nan, dtype=float)  # Recovery duration in days, NaN if no injury
        return_period_duration = np.full(len(g), np.nan, dtype=float)  # Return period duration in days, NaN if no injury
        days_since_recovery_end = np.full(len(g), 999, dtype=int)  # Days since recovery ended, 999 if not recovered yet
        
        # Track days since last injury ended (for post-injury risk)
        # Initialize to 999 (no recent injury)
        days_since_injury_end = 999

        # NEW: Calculate rest days in last 7 days for each day (used for rest day deficit calculation)
        # Rest day identified by: sessions == 0 OR km_total == 0 OR perceived_exertion == -0.01
        is_rest_day = np.zeros(len(g), dtype=bool)
        if "sessions" in g.columns and "km_total" in g.columns:
            sessions_arr = g["sessions"].fillna(0).values
            km_total_arr = g["km_total"].fillna(0).values
            perceived_exertion_arr = g.get("perceived_exertion", pd.Series([0] * len(g))).fillna(0).values
            is_rest_day = (sessions_arr == 0) | (km_total_arr == 0) | (perceived_exertion_arr == -0.01)
        
        # Calculate rest days in last 7 days window [t-6, t-1] for each day t
        rest_days_last_7 = np.zeros(len(g), dtype=int)
        for t in range(len(g)):
            # Look back 6 days (t-6 to t-1, excluding current day t)
            start_idx = max(0, t - 6)
            end_idx = max(0, t)  # t-1 inclusive (exclude t)
            if end_idx > start_idx:
                rest_days_last_7[t] = is_rest_day[start_idx:end_idx].sum()
            else:
                rest_days_last_7[t] = 0  # Not enough history
        
        # Calculate rest day deficit score for each day
        rest_day_deficit_scores = np.zeros(len(g), dtype=float)
        for t in range(len(g)):
            if t >= 7:  # Only calculate if we have enough history
                rest_day_deficit_scores[t] = _calculate_rest_day_deficit_score(
                    int(rest_days_last_7[t]),
                    athlete_rest_frequency
                )
            else:
                rest_day_deficit_scores[t] = 0.0  # Not enough history, no deficit
        
        # Calculate rest factor for each day (1.0 + deficit_score * weight)
        rest_factors = 1.0 + (rest_day_deficit_scores * cfg.injury_risk_rest_deficit_weight)
        # rest_factor: 1.0 (no deficit) to 1.3 (severe deficit, +30% risk)

        # PROFILE-BASED BASELINE RISK: Get profile-specific baseline injury hazard
        # U-shaped: novice (highest) -> recreational (lower) -> advanced (lowest) -> elite (higher)
        profile = str(g["profile"].iloc[0]) if "profile" in g.columns else "recreational"
        if profile == "novice":
            base_hazard = float(cfg.injury_hazard_novice)
        elif profile == "recreational":
            base_hazard = float(cfg.injury_hazard_recreational)
        elif profile == "advanced":
            base_hazard = float(cfg.injury_hazard_advanced)
        elif profile == "elite":
            base_hazard = float(cfg.injury_hazard_elite)
        else:
            base_hazard = float(cfg.injury_hazard_recreational)  # Default to recreational
        
        # PHASE 2: Risk factors contribute heavily to injury probability
        # Base hazard is profile-specific (U-shaped)
        # Risk factors multiply the base hazard significantly
        # Formula: p_inj = base_hazard * (1.0 + risk_multiplier * risk_score) + spike_absolute_risk + other_factors
        # risk_multiplier controls how much risk_score affects injury probability
        # Higher risk_multiplier = risk factors matter more, less randomness
        risk_multiplier = 8.0  # PHASE 2: Increased from 6.0 to 8.0 for stronger signal (was 3.5, then 6.0)
        # This means: risk_score of 0.5 -> 5x multiplier, risk_score of 2.0 -> 17x multiplier
        # This makes injuries strongly driven by risk factors, not just baseline hazard
            
        t = 0
        while t < len(g):
            # CRITICAL FIX: Check if already injured FIRST, before any other processing
            # This prevents overlapping injuries by skipping all injury generation logic
            # when already injured. Use explicit int() conversion to ensure proper comparison.
            current_injury_status = int(injury_ongoing[t])
            if current_injury_status == 1:
                # Already injured - skip this day entirely
                # Update days_since_injury_end tracking
                if t > 0:
                    prev_status = int(injury_ongoing[t-1])
                    if prev_status == 1 and current_injury_status == 0:
                        # Injury just ended today (shouldn't happen if we're here, but check anyway)
                        days_since_injury_end = 0
                    elif current_injury_status == 1:
                        # Currently injured, reset counter
                        days_since_injury_end = 999
                t += 1
                continue
            
            # Track days since injury ended (only if not currently injured)
            if t > 0:
                if injury_ongoing[t-1] == 1 and injury_ongoing[t] == 0:
                    # Injury just ended today
                    days_since_injury_end = 0
                elif injury_ongoing[t] == 0 and days_since_injury_end < 999:
                    # Not currently injured, continue counting
                    days_since_injury_end += 1
                elif injury_ongoing[t] == 0:
                    # Not injured, but counter was reset (new injury just occurred)
                    days_since_injury_end = 999
            
            # Skip if already ill (prevents overlapping illnesses)
            if illness_ongoing[t] == 1:
                t += 1
                continue

            # illness generation (only if not injured and not already ill)
            if rng.random() < float(p_ill[t]):
                dur = _sample_positive_int(cfg.illness_dur_mean, cfg.illness_dur_sd, rng, 2, 14)
                illness_onset[t] = 1
                # Set illness_ongoing for the duration - ensure we don't overwrite existing injuries
                end_idx = min(len(g), t + dur)
                for i in range(t, end_idx):
                    if injury_ongoing[i] == 0:  # Only set if not injured
                        illness_ongoing[i] = 1

            # injury hazard increases with persistent fatigue; elevated if ill
            ill_factor = 1.35 if illness_ongoing[t] == 1 else 1.0
            # HYBRID: Apply user-level injury proneness (0.5 = baseline, 0 = 0.5x risk, 1 = 1.5x risk)
            proneness_factor = 0.5 + injury_proneness  # Maps 0->0.5x, 0.5->1.0x, 1->1.5x
            # HYBRID: Apply user-level injury resilience (reduces baseline risk)
            # High resilience (1.0) = 0.85x risk, low resilience (0.0) = 1.15x risk
            resilience_factor = 1.15 - 0.3 * injury_resilience  # Maps 0->1.15x, 0.5->1.0x, 1->0.85x
            
            # POST-INJURY ELEVATED RISK: Athletes are at much higher risk after returning from injury
            # Risk is highest immediately after return, then decays over time
            post_injury_factor = 1.0  # Baseline
            if days_since_injury_end < getattr(cfg, "post_injury_risk_decay_days", 45):
                # Calculate risk multiplier that decays over time
                # Highest risk in first acute_days, then exponential decay
                acute_days = getattr(cfg, "post_injury_risk_acute_days", 7)
                decay_days = getattr(cfg, "post_injury_risk_decay_days", 45)
                max_mult = getattr(cfg, "post_injury_risk_max_multiplier", 2.5)
                
                if days_since_injury_end < acute_days:
                    # Acute phase: maximum risk
                    post_injury_factor = max_mult
                else:
                    # Decay phase: exponential decay from max_mult to 1.0
                    decay_progress = (days_since_injury_end - acute_days) / (decay_days - acute_days)
                    decay_progress = np.clip(decay_progress, 0.0, 1.0)
                    # Exponential decay: starts at max_mult, ends at 1.0
                    post_injury_factor = 1.0 + (max_mult - 1.0) * np.exp(-3.0 * decay_progress)
            
            # Calculate injury probability (only if not injured and past warmup)
            if t < int(getattr(cfg, "injury_warmup_days", 28)):
                p_inj = 0.0
            else:
                # VALIDATION: Injuries are generated based on:
                # 1. Profile-specific base hazard (U-shaped: novice > recreational > advanced < elite)
                # 2. Risk score (from multiple risk factors, multiplied by risk_multiplier=8.0)
                # 3. User-level proneness/resilience factors
                # 4. Post-injury elevated risk (decays over time)
                # 5. Illness factor (1.35x if ill)
                # 6. Spike absolute risk (additive, from long run spikes)
                # This ensures injuries are NOT random - they're strongly driven by risk factors
                
                # Calculate base probability from risk factors
                # NEW: Apply rest factor (rest day frequency deficit increases risk)
                rest_factor = float(rest_factors[t]) if t < len(rest_factors) else 1.0
                p_inj_base = base_hazard * (1.0 + risk_multiplier * float(risk_score[t])) * ill_factor * proneness_factor * resilience_factor * post_injury_factor * rest_factor
                
                # Get absolute spike risk (additive, not multiplicative)
                # This ensures spike days have higher injury probability regardless of other factors
                spike_abs_risk = float(spike_absolute_risk[t]) if t < len(spike_absolute_risk) else 0.0
                
                # Get absolute sprinting risk (additive, similar to spike risk)
                # Sprinting is the #1 injury driver in real data (1.92x ratio)
                # Direct additive risk ensures injuries happen on days after sprinting sessions
                sprint_abs_risk = float(sprinting_absolute_risk[t]) if t < len(sprinting_absolute_risk) else 0.0
                
                # SEPARATE CAPPING: Cap each component individually before adding
                # This allows different risk sources to have different maximum contributions
                # Base risk cap (normal risk factors: ACWR, ramp, sleep, HRV, etc.)
                p_inj_base_clip_max = getattr(cfg, 'p_inj_base_clip_max', 0.15)  # Default: 15%
                if p_inj_base_clip_max is not None and p_inj_base_clip_max > 0:
                    p_inj_base = float(np.clip(p_inj_base, 0.0, float(p_inj_base_clip_max)))
                
                # Spike absolute risk cap (long run spikes)
                spike_abs_risk_clip_max = getattr(cfg, 'spike_absolute_risk_clip_max', 0.40)  # Default: 40%
                if spike_abs_risk_clip_max is not None and spike_abs_risk_clip_max > 0:
                    spike_abs_risk = float(np.clip(spike_abs_risk, 0.0, float(spike_abs_risk_clip_max)))
                
                # Sprinting absolute risk cap
                sprint_abs_risk_clip_max = getattr(cfg, 'sprinting_absolute_risk_clip_max', 0.40)  # Default: 40%
                if sprint_abs_risk_clip_max is not None and sprint_abs_risk_clip_max > 0:
                    sprint_abs_risk = float(np.clip(sprint_abs_risk, 0.0, float(sprint_abs_risk_clip_max)))
                
                # Add all components together
                p_inj = p_inj_base + spike_abs_risk + sprint_abs_risk
                
                # DEBUG: Track spike risk for spike days and day+1/day+2 after spike
                if has_spike[t] or spike_abs_risk > 0:
                    debug_info = {
                        'user_id': int(g["user_id"].iloc[0]),
                        'day': t,
                        'date': str(g["date"].iloc[t]) if t < len(g) else '',
                        'spike_category': int(long_run_spike_category[t]) if t < len(long_run_spike_category) else 0,
                        'spike_absolute_risk': spike_abs_risk,
                        'p_inj_base': p_inj_base,
                        'p_inj_total': p_inj,
                        'risk_score': float(risk_score[t]),
                        'risk_raw': float(risk_raw[t]) if t < len(risk_raw) else 0.0,
                        'acwr': float(acwr[t]) if t < len(acwr) else 0.0,
                        'p_inj_after_clip': 0.0,  # Will be updated after clip
                        'injury_occurred': 0,  # Will be updated after injury check
                    }
                    # Store in a list for later analysis (we'll write to file at end)
                    generate_events._debug_spike_info.append(debug_info)
            
            # Optional final p_inj clipping (safety net after component caps)
            # This is an ADDITIONAL safety limit on top of the component caps
            # Set p_inj_clip_max to None or <= 0 to disable (use component caps only)
            # Default: None (no final clipping, component caps are sufficient)
            p_inj_clip_max = getattr(cfg, 'p_inj_clip_max', None)  # Default: None (use component caps only)
            if p_inj_clip_max is not None and p_inj_clip_max > 0:
                p_inj = float(np.clip(p_inj, 0.0, float(p_inj_clip_max)))
            # If p_inj_clip_max is None or <= 0, no final clipping is applied (component caps are used)
            
            # DEBUG: Update p_inj_after_clip in debug info (for all spike days, not just first 50)
            if has_spike[t] and hasattr(generate_events, '_debug_spike_info'):
                # Find the most recent debug entry for this user and day
                user_id = int(g["user_id"].iloc[0])
                for debug_entry in reversed(generate_events._debug_spike_info):
                    if debug_entry.get('user_id') == user_id and debug_entry.get('day') == t:
                        debug_entry['p_inj_after_clip'] = p_inj
                        break

            # Generate injury if probability threshold met
            # CRITICAL: Triple-check we're not already injured (safety check)
            # Use explicit int comparison to avoid numpy boolean issues
            # Re-check injury_ongoing[t] right before generating to catch any race conditions
            final_check = int(injury_ongoing[t])
            if final_check == 1:
                # Already injured - skip injury generation (this should have been caught earlier)
                t += 1
                continue
            
            if final_check == 0 and rng.random() < p_inj:
                # NEW: Calculate injury severity based on risk factors at injury onset
                # Get risk factor values at injury onset (day t)
                acwr_val = float(acwr[t]) if t < len(acwr) else 1.0
                spike_cat = int(long_run_spike_category[t]) if t < len(long_run_spike_category) else 0
                training_load_val = float(acute7[t]) if t < len(acute7) else 0.0
                baseline_load_val = float(chronic28[t] * 4.0) if t < len(chronic28) else training_load_val  # Approximate baseline
                consecutive_hard_val = int(consecutive_hard_days[t]) if t < len(consecutive_hard_days) else 0
                
                # Calculate recovery index (approximate from sleep/HRV if available)
                recovery_index_val = 0.7  # Default moderate recovery
                if "sleep_hours" in g.columns and "hrv_ms" in g.columns and t < len(g):
                    sleep_h = g["sleep_hours"].iloc[t] if not pd.isna(g["sleep_hours"].iloc[t]) else 7.0
                    hrv_ms = g["hrv_ms"].iloc[t] if not pd.isna(g["hrv_ms"].iloc[t]) else 65.0
                    # Normalize sleep around 7-8h (optimal)
                    sleep_norm = np.clip((sleep_h - 6.0) / 2.0, -1, 1)
                    # Normalize HRV around 60-70ms (typical good value)
                    hrv_norm = np.clip((hrv_ms - 50.0) / 20.0, -1, 1)
                    recovery_index_val = float(np.clip((sleep_norm * hrv_norm + 1.0) / 2.0, 0.0, 1.0))
                
                # Get rest day deficit score at injury onset
                rest_deficit_score = float(rest_day_deficit_scores[t]) if t < len(rest_day_deficit_scores) else 0.0
                
                # Calculate injury severity (1.0-10.0)
                severity = _calculate_injury_severity(
                    acwr=acwr_val,
                    spike_category=spike_cat,
                    training_load=training_load_val,
                    baseline_load=baseline_load_val,
                    consecutive_hard_days=consecutive_hard_val,
                    recovery_index=recovery_index_val,
                    rest_day_deficit_score=rest_deficit_score,
                    cfg=cfg
                )
                
                # Calculate recovery duration from severity (with fitness/resilience adjustments)
                calculated_recovery_duration = _get_recovery_duration_from_severity(
                    severity=severity,
                    fitness=fitness,
                    injury_resilience=injury_resilience,
                    cfg=cfg
                )
                
                # Calculate return period duration from recovery duration
                calculated_return_period_duration = _get_return_period_from_recovery(
                    recovery_duration=calculated_recovery_duration,
                    cfg=cfg
                )
                
                # CRITICAL: Check ALL days in the recovery duration BEFORE setting anything
                # This prevents any possibility of overlaps
                end_idx = min(len(g), t + calculated_recovery_duration)
                can_generate = True
                overlap_at_day = None
                for i in range(t, end_idx):
                    if int(injury_ongoing[i]) == 1:
                        # Overlap detected - cannot generate this injury
                        can_generate = False
                        overlap_at_day = i
                        break
                
                if not can_generate:
                    # Cannot generate injury due to overlap - skip
                    t += 1
                    continue
                
                # Safe to generate - set injury_onset and injury_ongoing
                injury_onset[t] = 1
                for i in range(t, end_idx):
                    injury_ongoing[i] = 1
                
                # CRITICAL FIX: Final overlap check after setting
                # If the previous day was injured AND we just created a new injury on current day,
                # that means the previous injury was still ongoing when we started the new one.
                # This is an overlap and we need to rollback.
                if t > 0 and int(injury_ongoing[t-1]) == 1 and int(injury_onset[t]) == 1:
                    # Overlap detected: previous injury is still ongoing but we started a new one
                    # Rollback this injury
                    injury_onset[t] = 0
                    # Unset injury_ongoing for all days we just set
                    for day_idx in range(t, end_idx):
                        injury_ongoing[day_idx] = 0
                    # Skip to next day
                    t += 1
                    continue
                
                # Store severity and duration information for this injury
                # Store at injury onset day
                injury_severity[t] = severity
                recovery_duration[t] = float(calculated_recovery_duration)
                return_period_duration[t] = float(calculated_return_period_duration)
                
                # Propagate duration to all days in recovery period AND return period (for gradual return logic)
                # Recovery period: t to t + recovery_duration - 1
                # Return period: t + recovery_duration to t + recovery_duration + return_period_duration - 1
                recovery_end_idx = min(len(g), t + int(calculated_recovery_duration))
                return_end_idx = min(len(g), recovery_end_idx + int(calculated_return_period_duration))
                
                # Store during recovery period (injury_ongoing = 1)
                for i in range(t, recovery_end_idx):
                    recovery_duration[i] = float(calculated_recovery_duration)
                    return_period_duration[i] = float(calculated_return_period_duration)
                
                # Also store return_period_duration during return period (so gradual return logic can use it)
                # This allows gradual return logic to know the return period duration even after recovery ends
                for i in range(recovery_end_idx, return_end_idx):
                    # Store return_period_duration so gradual return can calculate progress
                    # Note: recovery_duration not needed after recovery ends, but we keep it for consistency
                    return_period_duration[i] = float(calculated_return_period_duration)
                
                # Reset days_since_injury_end when new injury occurs
                days_since_injury_end = 999
                
                # DEBUG: Mark injury occurred in debug info
                if has_spike[t] and hasattr(generate_events, '_debug_spike_info'):
                    # Find the most recent debug entry for this day
                    for debug_entry in reversed(generate_events._debug_spike_info):
                        if debug_entry.get('user_id') == int(g["user_id"].iloc[0]) and debug_entry.get('day') == t:
                            debug_entry['injury_occurred'] = 1
                            break
            
            t += 1

        # NEW: Calculate days_since_recovery_end for each day (used for gradual return-to-training)
        # This tracks how many days have passed since recovery ended (recovery_end = injury_ongoing changes from 1 to 0)
        for i in range(len(g)):
            if i == 0:
                days_since_recovery_end[i] = 999  # First day, no history
            else:
                if injury_ongoing[i-1] == 1 and injury_ongoing[i] == 0:
                    # Recovery just ended today
                    days_since_recovery_end[i] = 0
                elif injury_ongoing[i] == 0 and days_since_recovery_end[i-1] < 999:
                    # Not currently injured, continue counting from previous day
                    days_since_recovery_end[i] = days_since_recovery_end[i-1] + 1
                elif injury_ongoing[i] == 1:
                    # Currently injured, reset counter
                    days_since_recovery_end[i] = 999
                else:
                    # Not injured, but counter was reset (new injury just occurred)
                    days_since_recovery_end[i] = 999

        g["illness_onset"] = illness_onset
        g["illness_ongoing"] = illness_ongoing
        g["injury_onset"] = injury_onset
        g["injury_ongoing"] = injury_ongoing
        
        # NEW: Add injury severity and duration fields to daily data
        g["injury_severity"] = pd.Series(injury_severity, index=g.index)
        g["recovery_duration"] = pd.Series(recovery_duration, index=g.index)
        g["return_period_duration"] = pd.Series(return_period_duration, index=g.index)
        g["days_since_recovery_end"] = pd.Series(days_since_recovery_end, index=g.index).astype(int)

        # Forward labels are onset-based: onset occurs in the next 7 days (t+1..t+7).
        onset = injury_onset.astype(int)
        injury_next_7d = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            j0 = i + 1
            j1 = min(len(g), i + 8)
            injury_next_7d[i] = 1 if onset[j0:j1].sum() > 0 else 0
        g["injury_next_7d"] = injury_next_7d

        # same for illness
        onset_i = illness_onset.astype(int)
        illness_next_7d = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            j0 = i + 1
            j1 = min(len(g), i + 8)
            illness_next_7d[i] = 1 if onset_i[j0:j1].sum() > 0 else 0
        g["illness_next_7d"] = illness_next_7d
        
        # GARMIN RUNSAFE: Add long run spike risk to daily output
        # FIXED: Convert to Series to ensure proper alignment and handle NaN values
        g["long_run_spike_risk"] = pd.Series(long_run_spike_risk, index=g.index).fillna(0.0)
        g["long_run_spike_category"] = pd.Series(long_run_spike_category, index=g.index).fillna(0)
        
        # NEW: Add spike absolute risk to daily output (so model can learn from it)
        # This is the additive risk value that directly affects injury probability
        # Model will learn: when spike_absolute_risk > 0, predict much higher risk
        # Note: This is the UNCAPPED value (before capping at spike_absolute_risk_clip_max)
        # The model will learn the relationship, including the cap, from the training data
        g["spike_absolute_risk"] = pd.Series(spike_absolute_risk, index=g.index).fillna(0.0)
        
        # NEW: Add sprinting absolute risk to daily output (so model can learn from it)
        # This is the additive risk value that directly affects injury probability
        # Model will learn: when sprinting_absolute_risk > 0, predict much higher risk
        # Note: This is the UNCAPPED value (before capping at sprinting_absolute_risk_clip_max)
        # The model will learn the relationship, including the cap, from the training data
        g["sprinting_absolute_risk"] = pd.Series(sprinting_absolute_risk, index=g.index).fillna(0.0)
        
        # CRITICAL: Add binary indicator for ANY spike event (makes it easier for model to learn)
        # This ensures the model can easily identify days with spike events
        g["has_long_run_spike"] = (g["long_run_spike_category"] > 0).astype(int)
        
        # Also add indicator for spike in last 7 days (captures day+1, day+2 effects)
        # This helps model learn that injuries often occur 1-3 days after spike
        g["had_spike_last_7d"] = g.groupby("user_id")["has_long_run_spike"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).sum().fillna(0) > 0
        ).astype(int)
        
        # RECOMMENDATION 1 & 2: Store consecutive hard days and risk in daily output
        # This allows the model to learn from consecutive hard days patterns
        g["consecutive_hard_days"] = pd.Series(consecutive_hard_days, index=g.index).fillna(0).astype(int)
        g["consecutive_hard_risk"] = pd.Series(consecutive_hard_risk, index=g.index).fillna(0.0).astype(float)
        
        # Also add binary indicators for consecutive hard days thresholds
        g["has_2plus_consecutive_hard"] = (g["consecutive_hard_days"] >= 2).astype(int)
        g["has_3plus_consecutive_hard"] = (g["consecutive_hard_days"] >= 3).astype(int)
        
        # RECOMMENDATION 1 & 2: Store high intensity session count and risk in daily output
        g["hard_sessions_count_7d"] = pd.Series(hard_sessions_count_7d, index=g.index).fillna(0).astype(int)
        g["high_intensity_count_risk"] = pd.Series(high_intensity_count_risk, index=g.index).fillna(0.0).astype(float)
        
        # Also add binary indicators for high intensity session thresholds
        g["has_3plus_hard_sessions_7d"] = (g["hard_sessions_count_7d"] >= 3).astype(int)
        g["has_4plus_hard_sessions_7d"] = (g["hard_sessions_count_7d"] >= 4).astype(int)
        g["has_5plus_hard_sessions_7d"] = (g["hard_sessions_count_7d"] >= 5).astype(int)

        # Keep fitness and injury_proneness (needed for model features)
        # Don't drop - these are useful features for the model
        # Note: fitness and injury_proneness are kept in output for model training

        out_rows.append(g)

    # DEBUG: Write spike_multiplier debug info to file (after all users processed)
    if hasattr(generate_events, '_debug_spike_info') and len(generate_events._debug_spike_info) > 0:
        import json
        import os
        # Write to current directory
        debug_file = "spike_multiplier_debug.json"
        try:
            with open(debug_file, 'w') as f:
                json.dump(generate_events._debug_spike_info, f, indent=2)
            # Clear for next run
            generate_events._debug_spike_info = []
        except Exception as e:
            pass  # Don't fail if we can't write debug file

    return pd.concat(out_rows, ignore_index=True)

