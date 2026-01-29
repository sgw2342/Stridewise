from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import asdict
from .config import GeneratorConfig
from .utils import clip01

def generate_users(cfg: GeneratorConfig, rng: np.random.Generator, elite_only: bool = False) -> pd.DataFrame:
    """Generate users with HYBRID enhancements:
    - VO2max-based pace calculation with proper sex differences
    - Profile category based on fitness (for reference)
    - VO2max estimate for pace calculation
    """
    n = cfg.n_users
    user_id = np.arange(1, n+1, dtype=int)

    # demographics
    sex = rng.choice(["F", "M"], size=n, p=[0.48, 0.52])
    age = np.clip(rng.normal(38, 10, size=n), 18, 70).round().astype(int)
    height_cm = np.clip(rng.normal(173, 9, size=n), 150, 200).round(1)
    weight_kg = np.clip(rng.normal(70, 12, size=n), 45, 110).round(1)

    # base training volume (km/week) - FIXED: Generate based on realistic profile distribution
    if elite_only:
        # For elite-only mode, generate competitive runners (advanced/elite)
        # Target: Advanced 80%, Elite 20% (to match competitive runners)
        # Advanced: 80-120 km/week, Elite: 80-120 km/week (same volume, differentiated by fitness)
        elite_profile_probs = [0.80, 0.20]  # [advanced, elite]
        elite_profiles_assigned = rng.choice(["advanced", "elite"], size=n, p=elite_profile_probs)
        
        base_km_week = np.zeros(n)
        fitness = np.zeros(n)
        for i, prof in enumerate(elite_profiles_assigned):
            if prof == "advanced":
                # Advanced: REDUCED to match real CC0 (was 80-150, now 60-110 km/week)
                # Real CC0 shows ~7.0 km/day average, synthetic shows ~9.3 km/day (+31%)
                # Need to reduce by ~24.5%: 115 * 0.755 = 87 km/week
                # New range: 60-110 km/week (mean 85.0)
                midpoint = (60.0 + 110.0) / 2.0  # 85.0
                sd = (110.0 - 60.0) / 4.0  # 12.5
                base_km_week[i] = np.clip(rng.normal(midpoint, sd), 60.0, 110.0)
                fitness[i] = clip01(rng.normal(0.77, 0.06))
                fitness[i] = np.clip(fitness[i], 0.70, 0.85)
            else:  # elite
                # Elite: REDUCED to match real CC0 (was 80-200, now 70-130 km/week)
                # Real CC0 shows ~7.0 km/day average, synthetic shows ~9.3 km/day (+31%)
                # Need to reduce by ~24.5%: 140 * 0.755 = 106 km/week
                # New range: 70-130 km/week (mean 100.0)
                midpoint = (70.0 + 130.0) / 2.0  # 100.0
                sd = (130.0 - 70.0) / 4.0  # 15.0
                base_km_week[i] = np.clip(rng.normal(midpoint, sd), 70.0, 130.0)
                fitness[i] = clip01(rng.normal(0.92, 0.05))
                fitness[i] = np.clip(fitness[i], 0.85, 1.0)
        
        base_km_week = base_km_week.round(2)
        profiles_assigned = elite_profiles_assigned
    else:
        # FIXED: Generate base_km_week based on realistic profile distribution
        # Target distribution: Novice 35%, Recreational 40%, Advanced 20%, Elite 5%
        # Volume ranges (all use normal distribution with mean at midpoint):
        #   Novice: 8-30 km/week (midpoint: 19.0, SD: 5.5)
        #   Recreational: 25-100 km/week (midpoint: 62.5, SD: 18.75)
        #   Advanced: 80-150 km/week (midpoint: 115.0, SD: 17.5)
        #   Elite: 80-200 km/week (midpoint: 140.0, SD: 30.0)
        profile_probs = [0.35, 0.40, 0.20, 0.05]  # [novice, recreational, advanced, elite]
        profiles_assigned = rng.choice(["novice", "recreational", "advanced", "elite"], 
                                      size=n, p=profile_probs)
        
        base_km_week = np.zeros(n)
        for i, prof in enumerate(profiles_assigned):
            if prof == "novice":
                # Novice: 8-30 km/week
                # Use normal distribution with mean at midpoint (19.0) of 8-30 range
                # SD chosen so ±2σ covers most of range: (30-8)/4 = 5.5
                midpoint = (8.0 + 30.0) / 2.0  # 19.0
                sd = (30.0 - 8.0) / 4.0  # 5.5
                base_km_week[i] = np.clip(rng.normal(midpoint, sd), 8.0, 30.0)
            elif prof == "recreational":
                # Recreational: 25-100 km/week (should be largest group at 40%)
                # Mean at midpoint (62.5) of 25-100 range
                # SD chosen so ±2σ covers most of range: (100-25)/4 = 18.75
                midpoint = (25.0 + 100.0) / 2.0  # 62.5
                sd = (100.0 - 25.0) / 4.0  # 18.75
                base_km_week[i] = np.clip(rng.normal(midpoint, sd), 25.0, 100.0)
            elif prof == "advanced":
                # Advanced: REDUCED to match real CC0 (was 80-150, now 60-110 km/week)
                # Real CC0 shows ~7.0 km/day average, synthetic shows ~9.3 km/day (+31%)
                # Need to reduce by ~24.5%: 115 * 0.755 = 87 km/week
                # New range: 60-110 km/week (mean 85.0)
                midpoint = (60.0 + 110.0) / 2.0  # 85.0
                sd = (110.0 - 60.0) / 4.0  # 12.5
                base_km_week[i] = np.clip(rng.normal(midpoint, sd), 60.0, 110.0)
            else:  # elite
                # Elite: REDUCED to match real CC0 (was 80-200, now 70-130 km/week)
                # Real CC0 shows ~7.0 km/day average, synthetic shows ~9.3 km/day (+31%)
                # Need to reduce by ~24.5%: 140 * 0.755 = 106 km/week
                # New range: 70-130 km/week (mean 100.0)
                midpoint = (70.0 + 130.0) / 2.0  # 100.0
                sd = (130.0 - 70.0) / 4.0  # 15.0
                base_km_week[i] = np.clip(rng.normal(midpoint, sd), 70.0, 130.0)
        
        base_km_week = base_km_week.round(2)
    
    long_run_frac = clip01(rng.normal(cfg.long_run_frac_mean, cfg.long_run_frac_sd, size=n))

    # FIXED: Set fitness based on profile (not the other way around)
    # Fitness is now derived from profile to match training volume expectations
    fitness = np.zeros(n)
    for i, prof in enumerate(profiles_assigned):
        if prof == "novice":
            # Novice: fitness 0.2-0.4 (lower end)
            fitness[i] = clip01(rng.normal(0.30, 0.08))
        elif prof == "recreational":
            # Recreational: fitness 0.4-0.65 (mid range)
            fitness[i] = clip01(rng.normal(0.52, 0.10))
        elif prof == "advanced":
            # Advanced: fitness 0.7-0.85 (high but not elite)
            fitness[i] = clip01(rng.normal(0.77, 0.06))
        else:  # elite
            # Elite: fitness 0.85-1.0 (highest)
            fitness[i] = clip01(rng.normal(0.92, 0.05))
    
    # Ensure fitness values are in correct ranges
    for i, prof in enumerate(profiles_assigned):
        if prof == "novice":
            fitness[i] = np.clip(fitness[i], 0.15, 0.40)
        elif prof == "recreational":
            fitness[i] = np.clip(fitness[i], 0.40, 0.70)
        elif prof == "advanced":
            fitness[i] = np.clip(fitness[i], 0.70, 0.85)
        else:  # elite
            fitness[i] = np.clip(fitness[i], 0.85, 1.0)

    # HYBRID: Estimate VO2max from fitness and base training volume
    # VO2max ranges: ~35-40 (novice) to ~60-70 (elite)
    # Map fitness (0..1) to VO2max (35..70)
    vo2max_base = 35.0 + fitness * 35.0  # Maps fitness 0->35, 1->70
    # Add individual variation
    vo2max = np.clip(vo2max_base + rng.normal(0, 3.0, size=n), 30.0, 75.0)
    
    # HYBRID: Apply sex-based VO2max adjustment (women typically 8-10 points lower)
    vo2max_adjusted = vo2max.copy()
    vo2max_adjusted[sex == "F"] += cfg.vo2max_sex_adjustment_female
    vo2max_adjusted = np.clip(vo2max_adjusted, 28.0, 70.0)

    # HYBRID: Pace calculation using VO2max-based formula (calibrated to target paces)
    # Formula: base_pace = 12.412 - 0.1513 * vo2max
    # Then easy pace = base_pace * 1.1
    base_pace_minpkm = cfg.pace_formula_intercept - cfg.pace_formula_slope * vo2max_adjusted
    base_pace_minpkm = np.clip(base_pace_minpkm, 3.0, 9.0)
    # Easy pace is ~10% slower than base pace
    pace_easy_minpkm = base_pace_minpkm * 1.1
    # Add small individual variation
    pace_easy_minpkm = np.clip(
        pace_easy_minpkm + rng.normal(0, 0.15, size=n),
        3.5, 9.5
    )

    # physiology baselines (fitness affects RHR/HRV)
    rhr_base = np.clip(rng.normal(cfg.rhr_mean, cfg.rhr_sd, size=n) - 7*fitness, 38, 78)
    hrv_base = np.clip(rng.normal(cfg.hrv_mean, cfg.hrv_sd, size=n) + 25*fitness, 20, 160)

    # HRmax: common formula with noise; ensure plausible range
    hrmax = np.clip(208.0 - 0.7*age + rng.normal(0, 6, size=n), 165, 210)
    hrr = np.clip(hrmax - rhr_base, 80, 170)

    # wear compliance
    wear_rate = clip01(rng.normal(cfg.wear_rate_mean, cfg.wear_rate_sd, size=n))

    # HYBRID: User-level injury proneness/resilience
    # FIXED: Increased variation to allow some athletes to have 0 injuries (matching real CC0)
    # This creates a learnable user-level signal for injury risk
    # Some users are inherently more prone to injury (higher baseline risk)
    # Some users are more resilient (lower baseline risk, may have 0 injuries)
    # This is independent of fitness but may correlate with it
    # INCREASED variation: std 0.25 (was 0.15) to create wider range, including very resilient athletes
    injury_proneness = clip01(rng.normal(0.4, 0.25, size=n))  # 0 = very resilient, 1 = very prone
    # Add slight correlation with fitness (lower fitness = slightly more prone)
    injury_proneness = clip01(injury_proneness - 0.1 * (1.0 - fitness) + rng.normal(0, 0.08, size=n))  # Increased noise
    # Ensure some athletes have very low proneness (0.0-0.1 range) to allow 0 injuries
    for i in range(n):
        if injury_proneness[i] < 0.1 and rng.random() < 0.15:  # 15% chance to have very low proneness
            injury_proneness[i] = rng.uniform(0.0, 0.1)  # Very resilient athletes
    
    # HYBRID: User-level injury resilience (separate from proneness)
    # Resilience affects recovery time and baseline risk reduction
    # Independent of proneness - some people are prone but resilient (quick recovery)
    # Some are not prone but not resilient (slow recovery if injured)
    injury_resilience = clip01(rng.normal(0.5, 0.15, size=n))  # 0 = low resilience, 1 = high resilience
    # Add slight correlation with fitness (higher fitness = slightly more resilient)
    injury_resilience = clip01(injury_resilience + 0.1 * fitness + rng.normal(0, 0.05, size=n))
    
    # NEW: User-level baseline rest day frequency (for rest day deficit calculation)
    # FIXED: Increased variation to match real CC0 (Range: 7.4%-100%, Std: 17.9%)
    # Rest day frequency: 0.0 = never rest, 1.0 = always rest
    # Real CC0: Mean 43.4%, Median 42.4%, Range 7.4%-100%, Std 17.9%
    # Target: ~43-44% overall mean, wider range (7-100%), higher std (~18%)
    # Varies by profile: Novice/Recreational higher, Advanced/Elite lower
    rest_day_frequency = np.zeros(n)
    # FIX: Reduce rest day frequency to match real CC0 (~27% vs current ~54%)
    # Target: Reduce all means by ~50% to get from 54% to 27%
    for i, prof in enumerate(profiles_assigned):
        if prof == "novice":
            # Novice: Reduced from 0.48 to 0.24 (50% reduction to match real CC0)
            base_freq = clip01(rng.normal(0.24, 0.08))  # Reduced mean and std
            rest_day_frequency[i] = np.clip(base_freq, 0.10, 0.40)
            # Allow some outliers: 10% chance of very low (7-15%) or very high (45-60%)
            if rng.random() < 0.10:
                if rng.random() < 0.5:
                    rest_day_frequency[i] = rng.uniform(0.07, 0.15)  # Very low rest
                else:
                    rest_day_frequency[i] = rng.uniform(0.45, 0.60)  # Very high rest
        elif prof == "recreational":
            # Recreational: Reduced from 0.43 to 0.22 (50% reduction to match real CC0)
            base_freq = clip01(rng.normal(0.22, 0.07))  # Reduced mean and std
            rest_day_frequency[i] = np.clip(base_freq, 0.12, 0.35)
            # Allow some outliers: 8% chance of very low (7-18%) or very high (40-55%)
            if rng.random() < 0.08:
                if rng.random() < 0.5:
                    rest_day_frequency[i] = rng.uniform(0.07, 0.18)  # Very low rest
                else:
                    rest_day_frequency[i] = rng.uniform(0.40, 0.55)  # Very high rest
        elif prof == "advanced":
            # Advanced: Reduced from 0.40 to 0.20 (50% reduction to match real CC0)
            # NOTE: Further reduction to 0.15 reduced performance (0.6225 → 0.6194)
            # Keeping 0.20 as it produces better AUC
            base_freq = clip01(rng.normal(0.20, 0.07))  # Mean 0.20, std 0.07
            rest_day_frequency[i] = np.clip(base_freq, 0.12, 0.32)
            # Allow some outliers: 8% chance of very low (7-20%) or very high (35-50%)
            if rng.random() < 0.08:
                if rng.random() < 0.5:
                    rest_day_frequency[i] = rng.uniform(0.07, 0.20)  # Very low rest
                else:
                    rest_day_frequency[i] = rng.uniform(0.35, 0.50)  # Very high rest
        else:  # elite
            # Elite: Reduced from 0.35 to 0.18 (50% reduction to match real CC0)
            # NOTE: Further reduction to 0.13 reduced performance (0.6225 → 0.6194)
            # Keeping 0.18 as it produces better AUC
            base_freq = clip01(rng.normal(0.18, 0.06))  # Mean 0.18, std 0.06
            rest_day_frequency[i] = np.clip(base_freq, 0.10, 0.30)
            # Allow some outliers: 10% chance of very low (7-18%) or very high (32-45%)
            if rng.random() < 0.10:
                if rng.random() < 0.6:  # Elite more likely to have very low rest
                    rest_day_frequency[i] = rng.uniform(0.07, 0.18)  # Very low rest
                else:
                    rest_day_frequency[i] = rng.uniform(0.32, 0.45)  # Very high rest
    
    rest_day_frequency = rest_day_frequency.round(3)

    # NEW: Add never_sprints flag for advanced/elite profiles
    # Real CC0 shows 27% of users never sprint
    # For advanced/elite profiles, assign 27% to never sprint
    never_sprints = np.zeros(n, dtype=bool)
    for i, prof in enumerate(profiles_assigned):
        if prof in ["advanced", "elite"]:
            # 27% of advanced/elite users never sprint
            if rng.random() < 0.27:
                never_sprints[i] = True
        # Novice/recreational already have low sprinting probability, so no need to add never_sprints flag
    
    # SPRINTING INJURY ASSOCIATION FIX: Make athletes who sprint more have higher injury proneness
    # Real CC0 shows injury days have 66.9% more sprinting on t-7, suggesting strong user-level effect
    # Athletes who sprint more frequently are more injury-prone overall
    # This creates the association across the whole week (t-7 to t-1)
    # Apply BEFORE the very low proneness check to ensure it's not overridden
    # OPTIMIZED: Boost of 0.35-0.50 gave best result (+37.6%, 56.2% of target)
    # v2 (0.50-0.70) and v3 (0.40-0.55) made it worse, so reverting to v1
    # To reach +66.9%, we may need additional pattern-level fixes (sprinting creates patterns)
    for i in range(n):
        if not never_sprints[i] and profiles_assigned[i] in ["advanced", "elite"]:
            # Athletes who sprint (not never_sprints) have higher injury proneness
            # Increase proneness by 0.35-0.50 to create strong user-level association
            # This ensures athletes who sprint more are more injury-prone overall
            # Best result: +37.6% (56.2% of target +66.9%)
            injury_proneness[i] = clip01(injury_proneness[i] + rng.uniform(0.35, 0.50))

    # FIXED: Profile is already assigned based on training volume
    # Use the profiles_assigned array that was created earlier
    profile = profiles_assigned

    users = pd.DataFrame({
        "user_id": user_id,
        "sex": sex,
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "base_km_week": base_km_week.round(2),
        "long_run_frac": long_run_frac.round(3),
        "fitness": fitness.round(3),
        "vo2max": vo2max.round(1),  # Base VO2max (before sex adjustment)
        "vo2max_adjusted": vo2max_adjusted.round(1),  # After sex adjustment (for pace)
        "profile": profile,  # Reference category
        "rhr_base": rhr_base.round(2),
        "hrv_base": hrv_base.round(2),
        "hrmax": hrmax.round(1),
        "hrr": hrr.round(1),
        "wear_rate": wear_rate.round(3),
        "pace_easy_minpkm": pace_easy_minpkm.round(3),
        "injury_proneness": injury_proneness.round(3),  # User-level injury proneness (0=resilient, 1=prone)
        "injury_resilience": injury_resilience.round(3),  # User-level injury resilience (0=low, 1=high) - affects recovery and baseline risk
        "rest_day_frequency": rest_day_frequency,  # User-level baseline rest day frequency (0=never, 1=always) - for rest day deficit calculation
        "never_sprints": never_sprints.astype(int),  # User-level flag: 1 = never sprints, 0 = may sprint (for advanced/elite only)
    })

    # stable key types (prevents merge bugs)
    users["user_id"] = users["user_id"].astype("int64")
    return users

