"""Feature engineering from CC0 day data to map to main_model features."""
from __future__ import annotations
import numpy as np
import pandas as pd

def engineer_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from day data to map to main_model's important features.
    
    Creates:
    - ACWR approximation
    - Spike risk detection
    - Hard day features
    - Load distribution features
    - Intensity features
    - Recovery indicators
    """
    df = df.copy()
    
    # Helper to get day values
    def get_day_values(base_col: str) -> np.ndarray:
        """Get values for days 0-6 (suffixes: '', '.1', '.2', '.3', '.4', '.5', '.6')."""
        values = []
        for i in range(7):
            suffix = '' if i == 0 else f'.{i}'
            col = f"{base_col}{suffix}"
            if col in df.columns:
                val = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values
            else:
                val = np.zeros(len(df))
            values.append(val)
        return np.array(values).T  # Shape: (n_rows, 7)
    
    # Get day values (CC0 uses specific column names)
    # Try multiple naming conventions (real CC0 uses "total km" with space)
    if any('kms.' in c for c in df.columns):
        kms = get_day_values('kms')
    elif any('total_km' in c for c in df.columns):
        kms = get_day_values('total_km')
    elif any('total km' in c for c in df.columns):
        # Real CC0 data uses "total km" with space
        kms = get_day_values('total km')
    else:
        raise ValueError(f"Could not find kms columns. Available columns: {[c for c in df.columns if 'km' in c.lower()][:10]}")
    
    if any('sessions.' in c for c in df.columns):
        sessions = get_day_values('sessions')
    elif any('nr_sessions' in c for c in df.columns):
        sessions = get_day_values('nr_sessions')
    elif any('nr. sessions' in c for c in df.columns):
        # Real CC0 data uses "nr. sessions" with space and period
        sessions = get_day_values('nr. sessions')
    else:
        sessions = np.zeros((len(df), 7))  # Default to zeros
    
    if any('kms_z3_4.' in c for c in df.columns):
        kms_z3_4 = get_day_values('kms_z3_4')
    elif any('km Z3-4' in c for c in df.columns):
        # Real CC0 data uses "km Z3-4" with space and dash
        kms_z3_4 = get_day_values('km Z3-4')
    else:
        kms_z3_4 = np.zeros((len(df), 7))
    
    if any('kms_z5_t1_t2.' in c for c in df.columns):
        kms_z5_t1_t2 = get_day_values('kms_z5_t1_t2')
    elif any('km Z5-T1-T2' in c for c in df.columns):
        # Real CC0 data uses "km Z5-T1-T2" with space and dashes
        kms_z5_t1_t2 = get_day_values('km Z5-T1-T2')
    else:
        kms_z5_t1_t2 = np.zeros((len(df), 7))
    
    if any('kms_sprinting.' in c for c in df.columns):
        kms_sprinting = get_day_values('kms_sprinting')
    elif any('km sprinting' in c for c in df.columns):
        # Real CC0 data uses "km sprinting" with space
        kms_sprinting = get_day_values('km sprinting')
    else:
        kms_sprinting = np.zeros((len(df), 7))
    
    # 1. ACWR Approximation - IMPROVED
    # Multiple ACWR approximations using different time windows
    
    eps = 1e-6
    
    # Acute = last 7 days
    acute_load_7d = kms.sum(axis=1)
    
    # Chronic estimates using different approaches
    # Approach 1: Use earlier days (3-6) as baseline, scale to 28-day
    chronic_4d = kms[:, 3:7].sum(axis=1)
    chronic_estimate_28d_v1 = chronic_4d * (28.0 / 4.0)
    acwr_v1 = acute_load_7d / (chronic_estimate_28d_v1 + eps)
    
    # Approach 2: Use weighted average (more weight to recent)
    # Days 0-2 (recent) vs days 3-6 (earlier)
    recent_3d = kms[:, 0:3].sum(axis=1)
    earlier_4d = kms[:, 3:7].sum(axis=1)
    # Weighted: 60% recent, 40% earlier (scaled to 28-day)
    chronic_weighted = (recent_3d * 0.6 + earlier_4d * 0.4) * (28.0 / 7.0)
    acwr_v2 = acute_load_7d / (chronic_weighted + eps)
    
    # Approach 3: Use mean of all 7 days as baseline (assumes stability)
    mean_7d = kms.mean(axis=1)
    chronic_mean_28d = mean_7d * 28.0
    acwr_v3 = acute_load_7d / (chronic_mean_28d + eps)
    
    # Use the best approximation (v1 - earlier days as baseline)
    acwr_approx = np.clip(acwr_v1, 0.0, 5.0)
    df['acwr_approx'] = acwr_approx.astype('float32')
    
    # Also add alternative ACWR calculations
    df['acwr_weighted'] = np.clip(acwr_v2, 0.0, 5.0).astype('float32')
    df['acwr_mean'] = np.clip(acwr_v3, 0.0, 5.0).astype('float32')
    
    # ACWR trend (increasing vs decreasing)
    df['acwr_trend'] = (acwr_v1 - acwr_v3).astype('float32')  # Compare v1 to mean-based
    
    # ACWR excess (above 1.5 threshold)
    df['acwr_excess'] = (acwr_approx > 1.5).astype('float32')
    df['acwr_high'] = (acwr_approx > 1.2).astype('float32')
    
    # IMPROVED: ACWR trajectory (rate of change) to match main_model feature (importance rank 23)
    # Approximate as change in ACWR: compare recent 3-day ACWR vs earlier 4-day ACWR
    recent_3d_acwr = recent_3d / (chronic_estimate_28d_v1 + eps)
    earlier_4d_chronic = earlier_4d * (28.0 / 4.0)
    earlier_4d_acwr = earlier_4d / (earlier_4d_chronic + eps)
    df['acwr_trajectory'] = np.clip(recent_3d_acwr - earlier_4d_acwr, -2.0, 2.0).astype('float32')
    
    # 2. Spike Risk Detection - IMPROVED to match main_model features
    # Main Model uses: had_spike_last_7d, spike_in_last_7d (importance ranks 20-21)
    # Need to create features that match main_model's spike detection
    
    max_daily_kms = kms.max(axis=1)
    mean_daily_kms = kms.mean(axis=1)
    median_daily_kms = np.median(kms, axis=1)
    std_daily_kms = kms.std(axis=1)
    
    # Method 1: Max vs mean (original)
    spike_risk_v1 = (max_daily_kms - mean_daily_kms) / (mean_daily_kms + eps)
    
    # Method 2: Max vs median (more robust to outliers)
    spike_risk_v2 = (max_daily_kms - median_daily_kms) / (median_daily_kms + eps)
    
    # Method 3: Z-score approach (how many stds above mean)
    spike_zscore = (max_daily_kms - mean_daily_kms) / (std_daily_kms + eps)
    
    # Method 4: Recent spike (day 0 vs days 1-6 average)
    recent_day = kms[:, 0]
    other_days_mean = kms[:, 1:7].mean(axis=1)
    spike_recent = (recent_day - other_days_mean) / (other_days_mean + eps)
    
    # IMPROVED: Match main_model's spike features (had_spike_last_7d, spike_in_last_7d)
    # Main Model's spike detection is based on long-run spikes (>10% increase vs 30-day max)
    # We approximate this using day 0 (most recent) vs average of previous 6 days
    # If day 0 is >10% above average, consider it a spike
    
    # Thresholds matching main_model's spike categories:
    # Small: >10-30%, Moderate: >30-100%, Large: >100%
    spike_small = (spike_recent > 0.10) & (spike_recent <= 0.30)
    spike_moderate = (spike_recent > 0.30) & (spike_recent <= 1.00)
    spike_large = (spike_recent > 1.00)
    
    # Match main_model feature: had_spike_last_7d (any spike in last 7 days)
    df['had_spike_last_7d'] = ((spike_recent > 0.10) | (spike_risk_v1 > 0.10)).astype('int8')
    
    # Match main_model feature: spike_in_last_7d (binary indicator)
    df['spike_in_last_7d'] = (spike_recent > 0.10).astype('int8')
    
    # Additional spike features
    df['spike_risk'] = np.clip(spike_risk_v1, 0.0, 10.0).astype('float32')
    df['spike_risk_median'] = np.clip(spike_risk_v2, 0.0, 10.0).astype('float32')
    df['spike_zscore'] = np.clip(spike_zscore, 0.0, 10.0).astype('float32')
    df['spike_recent'] = np.clip(spike_recent, 0.0, 10.0).astype('float32')
    
    # Binary spike indicators (multiple thresholds) matching main_model categories
    df['has_spike'] = (spike_risk_v1 > 0.10).astype('int8')  # >10% increase (matches main_model small spike threshold)
    df['has_small_spike'] = spike_small.astype('int8')  # 10-30% increase
    df['has_moderate_spike'] = spike_moderate.astype('int8')  # 30-100% increase
    df['has_large_spike'] = spike_large.astype('int8')  # >100% increase
    df['has_very_large_spike'] = (spike_risk_v1 > 1.0).astype('int8')  # >100% increase (alternative)
    df['has_recent_spike'] = (spike_recent > 0.10).astype('int8')  # Recent day spike >10%
    
    # Spike magnitude (matching main_model's spike risk calculation)
    df['spike_magnitude'] = np.clip(spike_risk_v1, 0.0, 5.0).astype('float32')
    
    # 3. Hard Day Features
    # Hard day = sprint kms > 0 OR high Z5 kms (>5km threshold)
    hard_day_threshold = 5.0
    hard_days = ((kms_sprinting > 0) | (kms_z5_t1_t2 > hard_day_threshold)).astype(int)
    df['hard_day_count'] = hard_days.sum(axis=1).astype('int8')
    df['hard_day_frequency'] = (df['hard_day_count'] / 7.0).astype('float32')
    
    # Consecutive hard days
    def max_consecutive(arr: np.ndarray) -> np.ndarray:
        """Find max consecutive True values per row."""
        result = np.zeros(arr.shape[0], dtype=int)
        for i in range(arr.shape[0]):
            row = arr[i]
            max_cons = 0
            current = 0
            for val in row:
                if val:
                    current += 1
                    max_cons = max(max_cons, current)
                else:
                    current = 0
            result[i] = max_cons
        return result
    
    df['consecutive_hard_days'] = max_consecutive(hard_days).astype('int8')
    
    # High intensity days (Z5 or sprint)
    high_intensity_days = ((kms_z5_t1_t2 > 3.0) | (kms_sprinting > 0)).astype(int)
    df['high_intensity_day_count'] = high_intensity_days.sum(axis=1).astype('int8')
    
    # 4. Load Distribution Features - IMPROVED
    # Variance and std of daily kms
    df['load_variance'] = np.var(kms, axis=1, ddof=0).astype('float32')
    df['load_std'] = std_daily_kms.astype('float32')
    df['load_cv'] = (std_daily_kms / (mean_daily_kms + eps)).astype('float32')  # Coefficient of variation
    
    # Training monotony (inverse variance - low variance = high monotony)
    df['load_monotony'] = (1.0 / (1.0 + df['load_variance'])).astype('float32')
    
    # Load range (max - min)
    df['load_range'] = (max_daily_kms - kms.min(axis=1)).astype('float32')
    df['load_range_ratio'] = (df['load_range'] / (mean_daily_kms + eps)).astype('float32')
    
    # Rest day count and patterns
    rest_days = ((sessions == 0) & (kms == 0)).astype(int)
    df['rest_day_count'] = rest_days.sum(axis=1).astype('int8')
    df['rest_day_frequency'] = (df['rest_day_count'] / 7.0).astype('float32')
    
    # Consecutive rest days
    def max_consecutive_rest(arr: np.ndarray) -> np.ndarray:
        """Find max consecutive rest days per row."""
        result = np.zeros(arr.shape[0], dtype=int)
        for i in range(arr.shape[0]):
            row = arr[i]
            max_cons = 0
            current = 0
            for val in row:
                if val:
                    current += 1
                    max_cons = max(max_cons, current)
                else:
                    current = 0
            result[i] = max_cons
        return result
    
    df['consecutive_rest_days'] = max_consecutive_rest(rest_days).astype('int8')
    
    # Load balance (how evenly distributed)
    # Lower variance = more balanced
    df['load_balance'] = (1.0 / (1.0 + df['load_variance'])).astype('float32')
    
    # 5. Intensity Features
    total_kms_7d = kms.sum(axis=1)
    z5_kms_7d = kms_z5_t1_t2.sum(axis=1)
    z3_4_kms_7d = kms_z3_4.sum(axis=1)
    sprint_kms_7d = kms_sprinting.sum(axis=1)
    
    # Intensity shares
    df['z5_share'] = np.clip(z5_kms_7d / (total_kms_7d + eps), 0.0, 1.0).astype('float32')
    df['z3_4_share'] = np.clip(z3_4_kms_7d / (total_kms_7d + eps), 0.0, 1.0).astype('float32')
    df['sprint_share'] = np.clip(sprint_kms_7d / (total_kms_7d + eps), 0.0, 1.0).astype('float32')
    df['high_intensity_share'] = np.clip((z5_kms_7d + sprint_kms_7d) / (total_kms_7d + eps), 0.0, 1.0).astype('float32')
    
    # High intensity indicator
    df['high_intensity_week'] = (df['high_intensity_share'] > 0.2).astype('int8')
    
    # 6. Recovery Indicators - MOVED to section 11 (Enhanced Recovery Signal Features)
    # Recovery features are now handled in section 11 with proper column name handling
    
    # 7. Additional Derived Features - IMPROVED
    # Total sessions
    df['total_sessions_7d'] = sessions.sum(axis=1).astype('int8')
    
    # Average daily kms
    df['avg_daily_kms'] = mean_daily_kms.astype('float32')
    df['median_daily_kms'] = median_daily_kms.astype('float32')
    
    # Max daily kms
    df['max_daily_kms'] = max_daily_kms.astype('float32')
    df['min_daily_kms'] = kms.min(axis=1).astype('float32')
    
    # Load increase (day 0 vs day 6)
    df['load_increase'] = (kms[:, 0] - kms[:, 6]).astype('float32')
    df['load_increase_pct'] = ((kms[:, 0] - kms[:, 6]) / (kms[:, 6] + eps)).astype('float32')
    
    # Load trend (slope over 7 days)
    x = np.arange(7)
    load_trends = []
    for i in range(len(df)):
        y = kms[i, :]
        if np.sum(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0
        load_trends.append(slope)
    df['load_trend'] = np.array(load_trends).astype('float32')
    
    # Recent vs earlier load (for trend detection)
    df['load_recent_vs_earlier'] = (recent_3d - earlier_4d).astype('float32')
    df['load_recent_vs_earlier_pct'] = ((recent_3d - earlier_4d) / (earlier_4d + eps)).astype('float32')
    
    # 8. Interaction Features (matching main_model's important interactions)
    # ACWR × High Intensity
    df['acwr_x_high_intensity'] = (df['acwr_approx'] * df['high_intensity_share']).astype('float32')
    
    # Spike × ACWR
    df['spike_x_acwr'] = (df['spike_risk'] * df['acwr_approx']).astype('float32')
    
    # Spike × High Intensity
    df['spike_x_high_intensity'] = (df['spike_risk'] * df['high_intensity_share']).astype('float32')
    
    # ACWR × Load Variance (high ACWR + low variance = risky)
    df['acwr_x_low_variance'] = (df['acwr_approx'] * (1.0 - df['load_variance'] / (df['load_variance'].max() + eps))).astype('float32')
    
    # Hard Days × ACWR
    df['hard_days_x_acwr'] = (df['hard_day_count'] * df['acwr_approx']).astype('float32')
    
    # Consecutive Hard × ACWR
    df['consecutive_hard_x_acwr'] = (df['consecutive_hard_days'] * df['acwr_approx']).astype('float32')
    
    # 9. Pattern Features
    # Days with increasing load
    increasing_days = np.zeros(len(df), dtype=int)
    for i in range(6):
        increasing_days += (kms[:, i] < kms[:, i+1]).astype(int)
    df['increasing_load_days'] = increasing_days.astype('int8')
    
    # Days with decreasing load
    decreasing_days = np.zeros(len(df), dtype=int)
    for i in range(6):
        decreasing_days += (kms[:, i] > kms[:, i+1]).astype(int)
    df['decreasing_load_days'] = decreasing_days.astype('int8')
    
    # Alternating pattern (high-low-high or low-high-low)
    alternating = np.zeros(len(df), dtype=int)
    for i in range(5):
        # Check if pattern is high-low-high or low-high-low (vectorized)
        high_low_high = (kms[:, i] > kms[:, i+1]) & (kms[:, i+1] < kms[:, i+2])
        low_high_low = (kms[:, i] < kms[:, i+1]) & (kms[:, i+1] > kms[:, i+2])
        alternating += (high_low_high | low_high_low).astype(int)
    df['alternating_pattern'] = alternating.astype('int8')
    
    # 10. POST-INJURY PATTERN FEATURES (Signal-Based Alignment)
    # Based on investigation: Post-injury periods show 25-28% lower volume, 34-59% lower intensity
    # These features detect post-injury patterns from training data
    
    # Volume reduction features (compare recent vs earlier periods)
    # Recent 3 days vs earlier 4 days (within 7-day window)
    # Use the kms array that was already extracted (handles different column name formats)
    recent_3d_volume = kms[:, 0:3].sum(axis=1)
    earlier_4d_volume = kms[:, 3:7].sum(axis=1)
    
    # Volume reduction (negative = reduction)
    df['volume_reduction_7d'] = ((recent_3d_volume - earlier_4d_volume) / (earlier_4d_volume + eps)).astype('float32')
    
    # Low volume period indicator (recent volume < 70% of earlier volume)
    df['low_volume_period'] = (recent_3d_volume < (earlier_4d_volume * 0.7)).astype('int8')
    
    # Gradual return indicator (volume increasing over time after low period)
    # Check if volume is increasing (positive trend) after a low period
    volume_trend_positive = (df['load_trend'] > 0).astype(int)
    df['gradual_return'] = (df['low_volume_period'].astype(int) & volume_trend_positive).astype('int8')
    
    # Intensity reduction features
    recent_3d_intensity = kms_z5_t1_t2[:, 0:3].sum(axis=1) + kms_sprinting[:, 0:3].sum(axis=1)
    earlier_4d_intensity = kms_z5_t1_t2[:, 3:7].sum(axis=1) + kms_sprinting[:, 3:7].sum(axis=1)
    
    # Intensity reduction (negative = reduction)
    df['intensity_reduction_7d'] = ((recent_3d_intensity - earlier_4d_intensity) / (earlier_4d_intensity + eps)).astype('float32')
    
    # Low intensity period indicator (recent intensity < 50% of earlier intensity)
    df['low_intensity_period'] = (recent_3d_intensity < (earlier_4d_intensity * 0.5)).astype('int8')
    
    # Post-injury pattern score (composite indicator)
    # Combines volume reduction, intensity reduction, and gradual return
    post_injury_score = (
        (df['low_volume_period'].astype(float) * 0.4) +
        (df['low_intensity_period'].astype(float) * 0.4) +
        (df['gradual_return'].astype(float) * 0.2)
    )
    df['post_injury_pattern_score'] = np.clip(post_injury_score, 0.0, 1.0).astype('float32')
    
    # ENHANCED: Additional post-injury features to better match main_model
    # Days since last injury approximation (inverse relationship with pattern score)
    # Higher pattern score = more recent injury (fewer days since)
    df['post_injury_days_approx'] = np.clip((1.0 - post_injury_score) * 30.0, 0.0, 30.0).astype('float32')
    df['post_injury_days_inverse'] = np.clip(1.0 / (df['post_injury_days_approx'] + 1.0), 0.0, 1.0).astype('float32')
    df['post_injury_bin'] = (post_injury_score > 0.5).astype('int8')
    df['post_injury_decay_exp'] = np.exp(-df['post_injury_days_approx'] / 7.0).astype('float32')
    df['post_injury_acute'] = (df['post_injury_days_approx'] < 7.0).astype('int8')
    
    # ENHANCED: Recovery features to better match main_model
    # Recovery mean (from perceived recovery if available)
    if 'perceived recovery' in df.columns or 'perceived_recovery' in df.columns:
        recovery_col = 'perceived recovery' if 'perceived recovery' in df.columns else 'perceived_recovery'
        recovery_values = get_day_values(recovery_col.replace(' ', '_').replace('.', ''))
        df['recovery_mean'] = recovery_values.mean(axis=1).astype('float32')
        df['recovery_min'] = recovery_values.min(axis=1).astype('float32')
        df['recovery_max'] = recovery_values.max(axis=1).astype('float32')
        df['recovery_trend'] = (recovery_values[:, 0] - recovery_values[:, 6]).astype('float32')
    
    # ENHANCED: ACWR clipped variants to match main_model
    df['acwr_clipped'] = np.clip(df['acwr_approx'], 0.0, 2.0).astype('float32')
    df['acwr_clipped_lag2'] = df['acwr_clipped'].shift(2).fillna(1.0).astype('float32')
    df['acwr_clipped_lag3'] = df['acwr_clipped'].shift(3).fillna(1.0).astype('float32')
    df['acwr_clipped_lag7'] = df['acwr_clipped'].shift(7).fillna(1.0).astype('float32')
    
    # ENHANCED: Training monotony (matching main_model feature rank 37)
    df['training_monotony'] = df['load_monotony'].copy()  # Already computed above
    
    # ENHANCED: Load monotony (matching main_model feature rank 67)
    df['load_monotony_enhanced'] = (1.0 / (1.0 + df['load_variance'])).astype('float32')
    
    # Post-injury pattern binary (strong indicator)
    df['post_injury_pattern'] = (df['post_injury_pattern_score'] > 0.5).astype('int8')
    
    # 11. ENHANCED RECOVERY SIGNAL FEATURES (Signal-Based Alignment)
    # Handle both "perceived_recovery" (underscore) and "perceived recovery" (space) column names
    recovery_col_name = None
    for col_pattern in ['perceived_recovery', 'perceived recovery']:
        if any(col_pattern in c for c in df.columns):
            recovery_col_name = col_pattern
            break
    
    if recovery_col_name:
        perceived_recovery = get_day_values(recovery_col_name)
        df['recovery_mean'] = np.mean(perceived_recovery, axis=1).astype('float32')
        df['recovery_min'] = np.min(perceived_recovery, axis=1).astype('float32')
        df['recovery_max'] = np.max(perceived_recovery, axis=1).astype('float32')
        df['recovery_std'] = np.std(perceived_recovery, axis=1).astype('float32')
        df['recovery_trend'] = (perceived_recovery[:, 0] - perceived_recovery[:, 6]).astype('float32')  # improving if positive
        df['recovery_range'] = (df['recovery_max'] - df['recovery_min']).astype('float32')
        
        # Recovery thresholds (normalized to 0-1 scale if needed)
        # CC0 perceived_recovery is typically 0-10, but may be normalized
        recovery_mean_normalized = df['recovery_mean'] / 10.0 if df['recovery_mean'].max() > 1.0 else df['recovery_mean']
        
        # Poor recovery indicators (multiple thresholds)
        df['poor_recovery'] = (recovery_mean_normalized < 0.3).astype('int8')  # < 3.0 on 0-10 scale
        df['very_poor_recovery'] = (recovery_mean_normalized < 0.2).astype('int8')  # < 2.0 on 0-10 scale
        df['recovery_improving'] = (df['recovery_trend'] > 0.1).astype('int8')
        df['recovery_declining'] = (df['recovery_trend'] < -0.1).astype('int8')
        df['recovery_stable'] = (np.abs(df['recovery_trend']) < 0.1).astype('int8')
        
        # Recovery × ACWR interaction (poor recovery + high ACWR = very risky)
        df['poor_recovery_x_acwr'] = (df['poor_recovery'].astype(float) * df['acwr_approx']).astype('float32')
        
        # Recovery × Spike interaction
        df['poor_recovery_x_spike'] = (df['poor_recovery'].astype(float) * df['spike_risk']).astype('float32')
        
        # Recovery × Hard Days
        df['poor_recovery_x_hard_days'] = (df['poor_recovery'].astype(float) * df['hard_day_count']).astype('float32')
        
        # Recovery × Post-Injury Pattern (poor recovery + post-injury pattern = very risky)
        df['poor_recovery_x_post_injury'] = (df['poor_recovery'].astype(float) * df['post_injury_pattern_score']).astype('float32')
    
    # 12. POST-INJURY × LOAD PATTERN INTERACTIONS (Signal-Based Alignment)
    # Post-injury + high ACWR = very risky
    df['post_injury_x_acwr'] = (df['post_injury_pattern_score'] * df['acwr_approx']).astype('float32')
    
    # Post-injury + spike = very risky
    df['post_injury_x_spike'] = (df['post_injury_pattern_score'] * df['spike_risk']).astype('float32')
    
    # Post-injury + high intensity = very risky
    df['post_injury_x_high_intensity'] = (df['post_injury_pattern_score'] * df['high_intensity_share']).astype('float32')
    
    return df
