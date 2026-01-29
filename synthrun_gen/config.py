from __future__ import annotations

"""Generator configuration for SynthRun/StrideWise - HYBRID VERSION.

This hybrid combines StrideWise's forecastable patterns with profile-based realism.

REPLACE: synthrun_gen/config.py with this file
"""

from dataclasses import dataclass, asdict, field
from typing import Literal, Dict, Any, List
import json


OutputFormat = Literal["csv", "parquet", "both"]


@dataclass
class GeneratorConfig:
    # ---- dataset shape ----
    seed: int = 42
    n_users: int = 500
    start_date: str = "2023-01-01"
    n_days: int = 365
    # Warmup period: generate data BEFORE start_date to provide history for rolling features and spike detection
    # This ensures users have previous long runs, rolling averages, etc. from day 1
    warmup_days: int = 60  # Generate 60 days of history before start_date (covers 30-day windows + buffer)

    # ---- base training profile (weekly km) ----
    base_km_mean: float = 35.0
    base_km_sd: float = 15.0
    long_run_frac_mean: float = 0.28
    long_run_frac_sd: float = 0.06

    # ---- coherent week-to-week planning dynamics ----
    weekly_ramp_sd: float = 0.06        # week-to-week volatility (~6%)
    weekly_scale_min: float = 0.55
    weekly_scale_max: float = 1.55
    cutback_every_weeks: int = 4        # every Nth week tends to be easier
    cutback_scale: float = 0.75
    camp_week_prob: float = 0.04        # occasional higher-volume "camp" week
    camp_scale: float = 1.20
    long_run_spike_prob: float = 0.40   # probability of spike (high frequency - ensures spikes are common)
    long_run_spike_mean_above_safe: float = 0.35  # mean increase above 5% safe threshold (increased to generate more large spikes)
    long_run_spike_sd_above_safe: float = 0.20     # std dev of increase above safe threshold (increased to generate more large spikes)
    long_run_spike_max: float = 2.0    # maximum spike multiplier (100% increase = doubling)

    # ---- injury hazard shaping (forecastable / persistent risk) ----
    injury_acwr_thresh: float = 0.95  # LOWERED: Was 1.10, but mean ACWR is ~0.99, so threshold was too high
    # With threshold at 0.95, ACWR can contribute to risk on more days, creating stronger risk-based signal
    injury_ramp_thresh: float = 1.15
    injury_fatigue_alpha: float = 0.6  # OPTIMIZED: Balance between persistence and responsiveness
    # With alpha=0.6, 40% of current day's risk_raw is incorporated (vs 25% with alpha=0.75)
    # This allows risk_score to respond faster to spikes while maintaining some persistence
    # Good compromise: Still forecastable but more responsive to immediate risks
    injury_risk_noise_sd: float = 0.03  # realism; too high reduces predictability

    # weights for risk components
    # ADJUSTED BASED ON REAL DATA ANALYSIS: Sprinting is #1 driver (1.92x), intensity zones are weak (1.06x-1.20x, non-significant)
    w_acwr: float = 1.2  # ACWR is primary risk factor (relative load)
    w_ramp: float = 0.6  # Rapid load increases are high risk
    w_hi: float = 0.025  # PHASE 2: INCREASED from 0.015 to 0.025 to strengthen day-1 Z5 signal
    # Model shows z5_day1 is #2 most important feature (8.78 importance)
    # Current signal: +29.1% difference, target: +50-100% difference
    # Increased weight to create stronger intensity zone signals on day 1
    # Real data shows intensity zones are WEAK injury drivers (1.06x-1.20x, non-significant, p>0.1)
    # CONSISTENCY ANALYSIS: Intensity zones show high variation across seeds (CV 11-15%, Std 0.128-0.187)
    # Mean ratios: Z3-4 ~1.734x, Z5-T1-T2 ~1.699x (consistent across seeds: 1.4x-1.9x range)
    # Root cause: Indirect correlation (injuries happen after hard sessions, which have intensity zones)
    # Strategy: ACCEPT REALISTIC CORRELATION - ~1.5x-1.8x ratio is realistic given indirect correlation
    # High variation (CV 11-15%) suggests exact tuning to 1.0x-1.2x is not possible
    # Target: Accept ~1.5x-1.8x ratio (realistic indirect correlation, cannot be tuned further)
    # Iteration 4: Consistency analysis confirmed high variation - accepting realistic correlation
    w_sprinting: float = 8.0  # INCREASED from 4.0 to 8.0: Sprinting is the #1 injury driver in real data (1.92x ratio, +91.9%) - STRONGEST SIGNAL
    # Real data shows sprinting at day -1 is 1.92x higher at injury vs non-injury (strongest signal)
    # Need very strong weight (8.0) to make sprinting the dominant risk factor and match real data ratio
    w_sleep: float = 0.5  # Sleep deficit is important
    w_hrv: float = 0.45  # HRV drop indicates poor recovery
    w_rhr: float = 0.45  # RHR rise indicates stress
    w_stress: float = 0.6  # Stress significantly affects injury risk
    w_long_run_spike: float = 2.0  # PHASE 2: Optimal value (reverted from Phase 3)
    # Long run spikes are high risk
    # Phase 3 showed that increasing this further (to 2.5) decreased performance
    # Calculation accounting for persistent fatigue state (alpha=0.9):
    # For 1.64x injury rate with baseline risk=2.0, need risk_raw~5.0
    # Spike contribution = 5.0 - 2.0 = 3.0
    # With interactions (1.5x), spike_base_risk = 3.0/1.5 = 2.0
    # For small spike (1.64), w_long_run_spike = 2.0/1.64 ≈ 1.2, using 2.0 for safety margin
    
    # NEW: Additional validated risk factor weights
    # INCREASED: Strengthened new risk factor weights
    w_training_load_spike: float = 0.8  # Increased from 0.30 - weekly spikes are high risk
    w_consecutive_hard_days: float = 1.5  # PHASE 2: Optimal value (reverted from Phase 3)
    # Consecutive hard days are very risky
    # Phase 3 showed that increasing this further (to 2.0) decreased performance
    w_insufficient_recovery: float = 0.9  # Increased from 0.35 - insufficient recovery is very risky
    w_load_distribution: float = 0.5  # Increased from 0.20 - load concentration matters
    w_high_intensity_count: float = 1.2  # RECOMMENDATION 2: High intensity sessions per week (3+, 4+, 5+) are risky

    # ---- HYBRID: Profile-based injury risk (fitness-dependent) ----
    # ACWR sensitivity multiplier by fitness level (0=novice, 1=elite)
    # Lower fitness = more sensitive to high ACWR
    acwr_sensitivity_low_fitness: float = 1.5   # fitness < 0.4 (novice-like)
    acwr_sensitivity_mid_fitness: float = 1.2    # fitness 0.4-0.7 (recreational)
    acwr_sensitivity_high_fitness: float = 1.0   # fitness 0.7-0.85 (advanced)
    acwr_sensitivity_elite_fitness: float = 0.8  # fitness > 0.85 (elite)
    
    # Absolute load risk: REDUCED - Total km is NON-SIGNIFICANT in real data (1.01x ratio, p=0.75)
    # Real injuries are driven by intensity spikes (sprinting) and relative load (ACWR), not absolute volume
    absolute_load_risk_threshold: float = 500.0  # TRIMP units (~50km/week)
    absolute_load_risk_fitness_threshold: float = 0.4  # Only applies if fitness < this
    absolute_load_risk_slope: float = 0.0  # REDUCED from 0.1 to 0.0 - absolute load (total km) is NON-SIGNIFICANT in real data (1.01x ratio, p=0.75)
    # Real injuries are driven by intensity spikes (sprinting) and relative load (ACWR), not absolute volume

    # ---- wear compliance ----
    wear_rate_mean: float = 0.88
    wear_rate_sd: float = 0.10

    # ---- sensor missingness when device worn ----
    miss_hrv: float = 0.07
    miss_rhr: float = 0.03
    miss_sleep: float = 0.05
    miss_stress: float = 0.08
    miss_resp: float = 0.10
    miss_temp: float = 0.10

    # ---- illness/injury process ----
    illness_hazard: float = 0.003
    # PROFILE-BASED BASELINE INJURY RISK (U-shaped: novice high, recreational lower, advanced lowest, elite higher)
    # REDUCED: Target injury rate ~2.00% (down from 3.62%)
    # Reduction factor: 0.5525 (44.8% reduction) - maintains proportional relationships
    # FURTHER REDUCED for advanced/elite: Match real CC0 data (7.88 injuries/athlete vs 11.63-13.00 in synthetic)
    # Advanced: 0.00403 → 0.00273 (32.2% reduction, 0.678x)
    # Elite: 0.00536 → 0.00325 (39.4% reduction, 0.606x)
    # These are daily baseline probabilities that will be modified by risk factors
    # Formula: p_inj = base_hazard * (1.0 + 8.0 * risk_score) * other_factors + absolute_risks
    # For risk_score=0.8 (typical): p_inj = base_hazard * 7.4 * other_factors + absolute_risks
    injury_hazard_novice: float = 0.00624      # Reduced from 0.0113 (× 0.5525) - Target ~2.00% injury rate
    injury_hazard_recreational: float = 0.00492  # Reduced from 0.0089 (× 0.5525) - Target ~2.00% injury rate
    injury_hazard_advanced: float = 0.00115    # REVERTED: Back to recommendations implementation values
    injury_hazard_elite: float = 0.00138       # REVERTED: Back to recommendations implementation values
    
    # ABSOLUTE SPIKE RISK: Add fixed probability instead of multiplying
    # GARMIN RUNSAFE: This is the BIGGEST SINGLE FACTOR in injury causation
    # REDUCED: Target injury rate ~2.00% (down from 3.62%)
    # Reduction factor: 0.5525 (44.8% reduction) - maintains proportional relationships
    # Values are daily probabilities that will be added to p_inj
    spike_absolute_risk_small: float = 0.04331    # INCREASED: Strengthen spike-injury association for ALL profiles (+12% from 0.03867) to reach AUC ≥0.70
    spike_absolute_risk_moderate: float = 0.04950  # INCREASED: Strengthen spike-injury association for ALL profiles (+12% from 0.04420) to reach AUC ≥0.70
    spike_absolute_risk_large: float = 0.06188    # INCREASED: Strengthen spike-injury association for ALL profiles (+12% from 0.05525) to reach AUC ≥0.70
    # PHASE 3 (BEST CONFIG): Optimal balance found through testing
    # Best result: 15% reduction gave ROC AUC 0.5884 (94.3% of target), PR AUC 0.0264 (98.5% of target)
    # This maintains strong signal while keeping associations in +50-100% range
    spike_absolute_risk_small_advanced_elite: float = 0.02950  # REVERTED: Back to recommendations implementation values
    spike_absolute_risk_moderate_advanced_elite: float = 0.03370  # REVERTED: Back to recommendations implementation values
    spike_absolute_risk_large_advanced_elite: float = 0.04210  # REVERTED: Back to recommendations implementation values
    
    # NEW: ABSOLUTE SPRINTING RISK - Direct additive effect on p_inj (like spikes)
    # Sprinting is the #1 injury driver in real data (1.92x ratio, +91.9%)
    # Real data shows sprinting at day -1 is 1.92x higher at injury vs non-injury
    # Need direct additive risk to ensure injuries happen on days after sprinting
    # Sprinting absolute risk is proportional to sprinting amount (km)
    # Formula: sprinting_abs_risk = sprinting_km * sprinting_risk_per_km
    # REDUCED: Target injury rate ~2.00% (down from 3.62%)
    # FURTHER REDUCED: Halve sprinting-related injuries (long run spikes should be #1 driver)
    # Reduction: 0.2762 → 0.1381 (50% reduction to halve sprinting injuries)
    # With 1.0 km sprinting = 13.81% risk (reduced from 27.62%), capped at 40% (sprinting_absolute_risk_clip_max)
    # 0.1 km = 1.38% risk, 1.0 km = 13.81% risk (capped at 13.81%), 5.0 km = 69.05% risk (capped at 40%)
    sprinting_absolute_risk_per_km: float = 0.1381  # Default for all profiles - 13.81% per km (1.38% per 0.1 km)
    # PHASE 3 (BEST CONFIG): Optimal balance found through testing
    # Best result: 0.13 gave ROC AUC 0.5884 (94.3% of target), PR AUC 0.0264 (98.5% of target)
    # This maintains strong signal while keeping associations in +50-100% range
    sprinting_absolute_risk_per_km_advanced_elite: float = 0.060  # REVERTED: Back to recommendations implementation values
    # FITNESS-LEVEL DEPENDENT SPRINTING FREQUENCY: Novice/recreational runners are less likely to do sprinting
    # This reduces sprinting-related injuries by reducing the chance they do sprinting at all
    # Probability that novice/recreational runners include sprinting in tempo/interval sessions
    sprinting_probability_novice_rec: float = 0.50  # 50% chance (vs 100% for advanced/elite) - reduces sprinting frequency
    # PHASE 2: Increased persistence to create more distributed risk timing
    # Current: 2 days persistence → strongest on day 6
    # Target: More distributed across days 0-6
    # Increase to 3 days to spread risk across more days
    sprinting_risk_persistence_days: int = 3  # Risk persists for 3 days after sprinting session (increased from 2)
    
    # SPRINTING INJURY ASSOCIATION FIX: Multiplier to strengthen sprinting-injury association
    # Real CC0 shows injury days have 66.9% more sprinting (t-7), suggesting strong association
    # Increase this multiplier to make sprinting a stronger injury driver
    # Default: 2.0x (increases sprinting_absolute_risk by 100% to strengthen association)
    # This ensures that when sprinting is present, injuries are much more likely
    sprinting_risk_association_multiplier: float = 2.0
    
    # PHASE 2: NEW: Spike risk persistence (similar to sprinting)
    # Controls how long spike risk persists after long run spike
    # Should create more distributed associations across days
    spike_risk_persistence_days: int = 3  # Risk persists for 3 days after spike (default: same as sprinting)
    
    # EXPERIMENT: Feature flag to control spike risk contribution
    # If True: Spike risk is added to both risk_raw (affects risk_score) AND p_inj (additive)
    # If False: Spike risk is ONLY added to p_inj (additive), NOT to risk_raw
    spike_add_to_risk_raw: bool = True  # Default: True (production behavior)
    
    # VERSION 4: Unbounded risk clipping (defaults to no clipping)
    # Risk raw clipping parameter
    # Controls maximum value for risk_raw before smoothing into risk_score
    # Default: None (no clipping, unbounded)
    # Previous production: 6.0 (clips risk_raw to 0.0-6.0)
    risk_raw_clip_max: float = None  # Default: None (no clipping, unbounded)
    
    # Risk score clipping parameter
    # Controls maximum value for risk_score before calculating p_inj
    # Default: None (no clipping, unbounded)
    # Previous production: 4.0 (clips risk_score to 0.0-4.0)
    # This is the PRIMARY rate controller - directly limits p_inj
    # Formula: p_inj = base_hazard * (1.0 + 8.0 * risk_score) * factors + spike_absolute_risk
    risk_score_clip_max: float = None  # Default: None (no clipping, unbounded)
    
    # p_inj clipping parameter - SEPARATE CAPS FOR EACH COMPONENT
    # Controls maximum value for p_inj (final injury probability)
    # Default: None (no clipping, unbounded)
    # Previous production: 0.15 (15% maximum daily injury probability)
    # This is the FINAL safety limit - prevents unrealistic injury probabilities
    # Formula: p_inj = base_hazard * (1.0 + 8.0 * risk_score) * factors + spike_absolute_risk
    p_inj_clip_max: float = None  # Default: None (no clipping, unbounded)
    
    # SEPARATE CAPPING: Cap each p_inj component individually before adding
    # This allows different risk sources to have different maximum contributions
    # Base risk cap (normal risk factors: ACWR, ramp, sleep, HRV, etc.)
    p_inj_base_clip_max: float = 0.15  # Default: 15%
    # Spike absolute risk cap (long run spikes)
    spike_absolute_risk_clip_max: float = 0.40  # Default: 40%
    # Sprinting absolute risk cap
    sprinting_absolute_risk_clip_max: float = 0.40  # Default: 40%
    
    # VERSION 4: Configurable risk component clipping
    # All risk components are now configurable (previously hardcoded)
    
    # ACWR excess clipping
    # Limits ACWR excess contribution to risk_raw
    # Previous: Hardcoded to 2.5
    acwr_excess_clip_max: float = 2.5  # Default: 2.5
    
    # Ramp excess clipping
    # Limits ramp excess contribution to risk_raw
    # Previous: Hardcoded to 2.5
    ramp_excess_clip_max: float = 2.5  # Default: 2.5
    
    # Sleep deficit clipping
    # Limits sleep deficit contribution to risk_raw
    # Previous: Hardcoded to 3.0
    sleep_def_clip_max: float = 3.0  # Default: 3.0
    
    # HRV drop clipping
    # Limits HRV drop contribution to risk_raw
    # Previous: Hardcoded to 3.0
    hrv_drop_clip_max: float = 3.0  # Default: 3.0
    
    # RHR rise clipping
    # Limits RHR rise contribution to risk_raw
    # Previous: Hardcoded to 3.0
    rhr_rise_clip_max: float = 3.0  # Default: 3.0
    
    # Stress rise clipping
    # Limits stress rise contribution to risk_raw
    # Previous: Hardcoded to 3.0
    stress_rise_clip_max: float = 3.0  # Default: 3.0
    
    # Training load spike clipping
    # Limits training load spike contribution to risk_raw
    # Previous: Hardcoded to 2.0
    training_load_spike_clip_max: float = 2.0  # Default: 2.0
    
    # Consecutive hard days clipping
    # Limits consecutive hard days contribution to risk_raw
    # Previous: Hardcoded to 10.0
    consecutive_hard_days_clip_max: float = 10.0  # Default: 10.0
    
    # Load distribution clipping
    # Limits load distribution contribution to risk_raw
    # Previous: Hardcoded to 2.0
    load_distribution_clip_max: float = 2.0  # Default: 2.0
    
    # Absolute load risk clipping
    # Limits absolute load risk contribution to risk_raw
    # Previous: Hardcoded to 2.0
    absolute_load_risk_clip_max: float = 2.0  # Default: 2.0
    
    # HI share clipping
    # Limits high intensity share (0.0-1.0, already a proportion)
    # Previous: Hardcoded to 1.0
    hi_share_clip_max: float = 1.0  # Default: 1.0
    
    # Legacy single hazard (kept for backward compatibility, but will be overridden by profile-based)
    injury_hazard: float = 0.0015  # Default (will be replaced by profile-specific values)
    # With ACWR threshold lowered to 0.95, risk_score should be higher, so base hazard adjusted accordingly
    # Expected: risk_score typically 0.8-2.5 with increased weights, so p_inj = 0.0085 * (1.0 + 3.5 * risk_score)
    # Low risk (0.8): 0.0085 * 3.8 = 3.23%
    # High risk (2.5): 0.0085 * 9.75 = 8.29%
    # Average risk (1.5): 0.0085 * 6.25 = 5.31%
    # But with clipping at 0.08 (8%), this should give ~1.5-2.0% overall injury rate

    # Prevent injury onsets during first N days to avoid unstable early rollups.
    injury_warmup_days: int = 28
    
    # ---- post-injury elevated risk (re-injury risk) ----
    # Athletes are at much higher risk immediately after returning from injury
    # Risk decays over time as they adapt back to training
    post_injury_risk_max_multiplier: float = 5.0  # FURTHER STRENGTHENED: Increased from 3.5 to 5.0 (400% higher risk)
    post_injury_risk_decay_days: int = 45  # Days for risk to decay to baseline (typically 30-60 days)
    post_injury_risk_acute_days: int = 7  # Days of highest risk immediately after return (typically 3-14 days)

    # event durations (days) - HYBRID: fitness-dependent injury duration
    illness_dur_mean: float = 4.0
    illness_dur_sd: float = 1.5
    # Base injury duration (will be adjusted by fitness and severity)
    injury_dur_mean_base: float = 14.0  # For mid-fitness (0.5-0.7)
    injury_dur_sd: float = 4.0
    # Fitness adjustments (days added/subtracted from base)
    injury_dur_low_fitness_add: float = 4.0   # fitness < 0.4: +4 days
    injury_dur_mid_low_add: float = 2.0       # fitness 0.4-0.5: +2 days
    injury_dur_high_subtract: float = 2.0     # fitness 0.7-0.85: -2 days
    injury_dur_elite_subtract: float = 4.0    # fitness > 0.85: -4 days
    
    # ---- Injury Severity Modeling (NEW) ----
    # Severity scale: 1-10 (1=mild, 10=worst)
    # Base severity (mild injury even with low risk)
    injury_severity_base: float = 3.0
    # Risk factor contributions to severity (each contributes 0-2 points)
    injury_severity_acwr_weight: float = 2.0  # ACWR > 1.0 adds to severity
    injury_severity_spike_weight: float = 1.5  # Spike size matters
    injury_severity_load_weight: float = 1.5  # Training load above baseline
    injury_severity_hard_days_weight: float = 1.5  # Consecutive hard days
    injury_severity_recovery_weight: float = 1.0  # Poor recovery
    injury_severity_rest_deficit_weight: float = 0.5  # Rest day deficit
    
    # Recovery duration mapping (severity → recovery days)
    # Formula: recovery_days = base_days * (severity_factor ^ exponent)
    injury_recovery_base_days: float = 10.0  # Base for severity 5
    injury_recovery_exponent: float = 1.8  # Exponential curve
    injury_recovery_severity_1_days: float = 2.0  # Severity 1: 2 days
    injury_recovery_severity_5_days: float = 14.0  # Severity 5: 14 days
    injury_recovery_severity_10_days: float = 90.0  # Severity 10: 90 days (3 months)
    injury_recovery_min_days: int = 2  # Minimum recovery duration
    injury_recovery_max_days: int = 120  # Maximum recovery duration (4 months cap)
    
    # Gradual return-to-training period (proportional to recovery duration)
    injury_return_min_days: int = 1  # Minimum return period (severity 1)
    injury_return_max_days: int = 28  # Maximum return period (4 weeks, severity 10)
    # Return period mapping (recovery_days → return_days)
    # Piecewise linear: Recovery [2,7,14,30,60,90] → Return [1,3,10,17,24,28]
    injury_return_recovery_points: List[float] = field(default_factory=lambda: [2, 7, 14, 30, 60, 90])
    injury_return_period_points: List[float] = field(default_factory=lambda: [1, 3, 10, 17, 24, 28])
    # Volume ramp-up parameters (proportional to return period)
    injury_return_volume_start_short: float = 0.5  # Start at 50% volume (short return)
    injury_return_volume_start_moderate: float = 0.4  # Start at 40% volume (moderate return)
    injury_return_volume_start_long: float = 0.25  # Start at 25% volume (long return)
    injury_return_intensity_delay_ratio: float = 0.3  # Intensity starts after 30% of return period
    injury_return_ramp_steepness_moderate: float = 3.0  # Sigmoid curve steepness (moderate return)
    injury_return_ramp_steepness_long: float = 2.5  # Sigmoid curve steepness (long return)
    injury_return_week_delay_long: int = 7  # Fixed 1 week delay for intensity (long return)
    
    # ---- Rest Day Frequency as Injury Risk Factor (NEW) ----
    # Rest day deficit contributes to injury probability and severity
    injury_risk_rest_deficit_weight: float = 0.3  # +30% risk at max deficit (rest_factor = 1.0 + deficit_score * 0.3)
    injury_severity_rest_deficit_weight: float = 0.5  # +0.5 points to severity at max deficit
    
    # ---- Post-Workout Rest Day Probability (NEW) ----
    # FIX: Reduced post-workout rest probabilities to match real CC0 ~27% overall rest frequency
    # Current: 39.22% rest days (target: ~27%)
    # Weekly plan: ~21.4% rest days
    # Post-workout adds: ~18% extra → need to reduce by ~40% to get to ~27% total
    # Strategy: Reduce all probabilities by ~40% (from 23% to ~14%, etc.)
    post_workout_rest_prob_high_intensity: float = 0.14  # Reduced from 0.23 to 0.14 (14% after high intensity)
    post_workout_rest_prob_long_run: float = 0.09  # Reduced from 0.15 to 0.09 (9% after long run)
    post_workout_rest_prob_extremely_long: float = 0.18  # Reduced from 0.30 to 0.18 (18% after extremely long run)
    post_workout_rest_prob_hard_and_long: float = 0.21  # Reduced from 0.35 to 0.21 (21% after hard + long session)
    post_workout_rest_prob_extremely_long_and_hard: float = 0.24  # Reduced from 0.40 to 0.24 (24% after extremely long + hard session)
    
    # Long run thresholds (as percentage of weekly volume, profile-specific)
    long_run_threshold_pct: float = 0.15  # 15% of weekly volume = long run
    extremely_long_run_threshold_pct: float = 0.20  # 20% of weekly volume = extremely long run
    # Profile-specific thresholds (optional, if we want to vary by profile)
    # For now, use same percentage for all profiles, but could vary:
    # novice_long_run_pct: float = 0.12
    # elite_long_run_pct: float = 0.18

    # ---- physiology baselines / dynamics ----
    rhr_mean: float = 55.0
    rhr_sd: float = 6.0
    hrv_mean: float = 65.0
    hrv_sd: float = 18.0

    # mean-reversion strength for daily RHR/HRV (0..1)
    mr_kappa_rhr: float = 0.25
    mr_kappa_hrv: float = 0.20

    # scaling for load effects on physio
    rhr_load_slope: float = 0.035
    hrv_load_slope: float = -0.45

    # ---- HR modelling ----
    hr_drift_per_hour: float = 4.0
    illness_hr_bump: float = 6.0
    temp_illness_bump: float = 0.35

    # ---- HYBRID: Pace calculation ----
    # Sex-based VO2max adjustment for pace (women typically 8-10 points lower)
    vo2max_sex_adjustment_female: float = -8.0
    # Pace formula: calibrated to match target paces
    # Novice (VO2max=40): easy ~7:00, Advanced (VO2max=55): easy ~4:30
    pace_formula_intercept: float = 12.412
    pace_formula_slope: float = 0.1513

    # ---- output ----
    out_format: OutputFormat = "csv"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "GeneratorConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return GeneratorConfig(**data)

