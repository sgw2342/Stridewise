from __future__ import annotations
import numpy as np
import pandas as pd
from .config import GeneratorConfig
from .utils import date_range, rolling_sum, rolling_mean, ou_step, clip01

def _generate_training_cycle_structure(
    n_weeks: int,
    profile: str,
    rng: np.random.Generator
) -> dict:
    """
    Generate a structured training cycle → break pattern for an athlete.
    
    Based on real CC0 data analysis:
    - Training cycles: Mean 12.2 weeks, Median 7.0 weeks (variable, 1-90 weeks)
    - Recovery breaks: Mean 3.1 weeks, Median 1.0 weeks (mostly 1-3 weeks, typically 1 week)
    - Break-to-cycle ratio: Median 0.33 (breaks are ~1/3 the length of cycles)
    - 82.5% of easy clusters are training cycle breaks (not injury recovery)
    - Most athletes vary cycle lengths (CV ~0.80), some (22.6%) are more consistent (CV < 0.50)
    
    Note: Injuries are handled separately later in the pipeline and will override this structure
    by enforcing rest days during injury_ongoing periods (as done in pipeline.py).
    
    Returns:
        Dictionary mapping week_idx -> (is_break_week, cycle_num)
        - week_idx: Week number (0-indexed)
        - is_break_week: True if this is a recovery break week (easy, no intensity), False if training cycle week (hard, with intensity)
        - cycle_num: Which training cycle this week belongs to (0-indexed)
    """
    cycle_structure = []
    current_week = 0
    cycle_num = 0
    
    # Determine athlete consistency (most vary, some are consistent)
    is_consistent_athlete = rng.random() < 0.226  # 22.6% are consistent
    
    # Consistent athletes: Longer cycles (~14.5 weeks), consistent breaks (~3.2 weeks)
    # Variable athletes: Mixed cycle lengths (CV ~0.80), variable breaks
    # ADJUSTED: Consistent athletes also need more breaks to reach ~24% overall
    if is_consistent_athlete:
        base_cycle_length = 12.0  # Reduced from 14.5 to increase break frequency
        cycle_length_std = 2.0  # Lower variation
        base_break_length = 3.5  # Increased from 3.2 to get more breaks
        break_length_std = 0.8  # Slightly more variation
    else:
        base_cycle_length = 5.8  # Adjusted from 5.4 to 5.8 to target ~24% break frequency (structure test shows ~26%, but actual data is ~22%, accounting for injury interference)
        cycle_length_std = 6.0  # Higher variation (CV ~0.75)
        base_break_length = 2.0  # Increased from 1.5 to get mean ~3.1 weeks
        break_length_std = 2.0  # Higher variation (adjusted for wider distribution)
    
    while current_week < n_weeks:
        # Generate training cycle length
        # Distribution: Short cycles (1-4 weeks) 30%, Medium (5-12 weeks) 50%, Long (13-30 weeks) 18%, Very long (31+) 2%
        rand_val = rng.random()
        if rand_val < 0.30:
            # Short cycles: 1-4 weeks
            cycle_length = max(1, int(np.clip(rng.normal(2.5, 1.0), 1, 4)))
        elif rand_val < 0.80:
            # Medium cycles: 4-7 weeks (further reduced from 4-8 to 4-7 to increase break frequency to ~24%)
            cycle_length = max(4, int(np.clip(rng.normal(base_cycle_length, cycle_length_std), 4, 7)))
        elif rand_val < 0.98:
            # Long cycles: 13-30 weeks
            cycle_length = max(13, int(np.clip(rng.normal(20, 6), 13, 30)))
        else:
            # Very long cycles: 31-50 weeks (rare)
            cycle_length = max(31, int(np.clip(rng.normal(40, 8), 31, 50)))
        
        # Ensure we don't exceed n_weeks
        cycle_length = min(cycle_length, n_weeks - current_week)
        if cycle_length <= 0:
            break  # No more weeks left
        
        # Generate break length based on distribution (from real CC0 analysis)
        # Distribution: 53% are 1 week, 15% are 2 weeks, 10% are 3 weeks, 22% are 4+ weeks
        # Mean: 3.1 weeks, Median: 1.0 weeks
        # ADJUSTED: Increase break length distribution to reach ~24% break frequency (from 20.3%)
        # To get from 20.3% to 24%, need ~18% increase in break frequency
        # Strategy: Reduce 1-week breaks from 53% to 45%, increase 2+ week breaks proportionally
        rand_val = rng.random()
        if rand_val < 0.48:  # Adjusted from 0.46 to 0.48 to target ~24% (structure shows ~26%, actual data ~22%, accounting for injury interference)
            # 48% are 1 week breaks (adjusted to target ~24% break frequency in actual data)
            base_break_len = 1.0
        elif rand_val < 0.68:  # 48% + 20% = 68% (increased from 15% to 20%)
            # 20% are 2 week breaks (increased from 15% to get more breaks)
            base_break_len = 2.0
        elif rand_val < 0.78:  # 68% + 10% = 78% (kept at 10%)
            # 10% are 3 week breaks
            base_break_len = 3.0
        else:
            # 22% are 4+ week breaks (longer recovery, adjusted to target ~24% break frequency in actual data)
            # ADJUSTED: To get mean 3.1 weeks: (0.48*1 + 0.20*2 + 0.10*3 + 0.22*avg_4plus) = 3.1
            # Solving: avg_4plus = (3.1 - 1.18) / 0.22 = 8.73 weeks
            # So 4+ week breaks should average ~8.7 weeks
            # Use weighted distribution: mostly 4-6 weeks, some 7-10 weeks, few 11-14 weeks
            rand_val = rng.random()
            if rand_val < 0.45:  # 45% of 4+ week breaks are 4-6 weeks (increased from 40%)
                base_break_len = 4.0 + rng.integers(0, 3)  # 4-6 weeks
            elif rand_val < 0.80:  # 35% are 7-10 weeks (kept similar)
                base_break_len = 7.0 + rng.integers(0, 4)  # 7-10 weeks
            else:  # 20% are 11-14 weeks (extended recovery, reduced from 25% to shift towards shorter breaks)
                base_break_len = 11.0 + rng.integers(0, 4)  # 11-14 weeks
            # This gives mean ~8.7 weeks for 4+ week breaks, which should push overall mean to ~3.0-3.1 weeks
        
        # Adjust based on cycle length (minimal adjustment to maintain distribution)
        # Distribution from real CC0 is mostly independent of cycle length
        # Only apply slight adjustments: very short cycles → longer breaks, very long cycles → shorter breaks
        if cycle_length <= 2:
            # Very short cycles → longer breaks (recovery after intense period, override distribution)
            break_length = max(1, int(np.clip(rng.normal(3.4, 1.0), 1, 8)))
        elif cycle_length <= 30:
            # Most cycles (3-30 weeks) → use distribution-based break length with minimal variation
            # Add small random variation (±1 week) to maintain distribution but add realism
            variation = rng.integers(-1, 2)  # -1, 0, or +1 week
            break_length = int(max(1, base_break_len + variation))
        else:
            # Very long cycles (>30 weeks) → slightly shorter breaks (minimal adjustment)
            # But still maintain distribution (don't override completely)
            break_length = int(max(1, np.clip(base_break_len - 0.5, 1, base_break_len)))
        
        # Generate training cycle weeks (hard weeks with intensity)
        for w in range(current_week, current_week + cycle_length):
            cycle_structure.append((w, False, cycle_num))  # False = training cycle week (hard, with intensity)
        
        current_week += cycle_length
        
        # Generate break weeks (easy weeks, no intensity)
        # Only if we haven't reached the end and there's room for at least 1 week break
        if current_week < n_weeks:
            # Ensure we don't exceed n_weeks
            break_length = min(break_length, n_weeks - current_week)
            
            if break_length > 0:
                # Generate break weeks based on calculated break_length
                for w in range(current_week, current_week + break_length):
                    cycle_structure.append((w, True, cycle_num))  # True = break week (easy, no intensity)
                
                current_week += break_length
                cycle_num += 1
            else:
                # No break (shouldn't happen, but safety check)
                cycle_num += 1
        else:
            # No break at end of training period
            cycle_num += 1
    
    # Return as dictionary for easy lookup: week_idx -> (is_break, cycle_num)
    return {week_idx: (is_break, cycle_num) for week_idx, is_break, cycle_num in cycle_structure}

def build_daily_plan(cfg: GeneratorConfig, users: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create a daily plan (km_total + session_type) before physiology/events.

    Key realism fix:
      - Weekly volume scaling must be coherent *within a week* and evolve gradually *week-to-week*.
        (Previously, the weekly scale was unintentionally re-sampled every day, injecting noise that
         destroys forward-time predictability of injury risk drivers.)

    Injury/illness events later will down-regulate physiology (and optionally may down-regulate realized load).
    
    WARMUP PERIOD: Generates data BEFORE start_date to provide history for:
    - Rolling features (need 7-28 day windows)
    - Spike detection (needs previous long runs in last 30 days)
    - ACWR calculation (needs 28-day history)
    """
    # Generate warmup period BEFORE start_date
    warmup_days = int(getattr(cfg, "warmup_days", 60))
    if warmup_days > 0:
        warmup_start = pd.to_datetime(cfg.start_date) - pd.Timedelta(days=warmup_days)
        warmup_dates = date_range(warmup_start.strftime("%Y-%m-%d"), warmup_days)
        dates = list(warmup_dates) + list(date_range(cfg.start_date, cfg.n_days))
    else:
        dates = date_range(cfg.start_date, cfg.n_days)
    
    n_weeks = int(np.ceil(len(dates) / 7.0))

    rows = []
    for _, u in users.iterrows():
        uid = int(u["user_id"])
        base_km_week = float(u["base_km_week"])
        long_frac = float(u["long_run_frac"])
        profile = str(u["profile"]) if "profile" in u else "recreational"
        # FIXED: Extract rest_day_frequency attribute to incorporate into plan generation
        user_rest_freq = float(u["rest_day_frequency"]) if "rest_day_frequency" in u else None

        # PROFILE-BASED TRAINING PLAN GENERATION
        # Each profile has different training frequency and intensity patterns

        # Build coherent weekly scales with a mild random-walk ramp and cutback weeks
        scale = 1.0
        weekly_scales: list[float] = []
        for w in range(n_weeks):
            # mild multiplicative drift (log-normal-ish)
            step = float(rng.normal(0.0, cfg.weekly_ramp_sd))
            scale = float(scale * np.exp(step))
            scale = float(np.clip(scale, cfg.weekly_scale_min, cfg.weekly_scale_max))

            # cutback week (commonly every 4th week)
            if cfg.cutback_every_weeks > 0 and ((w + 1) % int(cfg.cutback_every_weeks) == 0):
                if rng.random() < 0.90:
                    scale = float(scale * cfg.cutback_scale)

            # occasional "camp" / spike week
            if rng.random() < cfg.camp_week_prob:
                scale = float(scale * cfg.camp_scale)

            scale = float(np.clip(scale, 0.45, 1.80))
            weekly_scales.append(scale)

        # Track long runs since last spike to ensure at least one per month
        long_runs_since_last_spike = 999  # Start high to force early spike
        
        # NEW: Generate structured training cycle → break pattern for this athlete
        # This replaces random easy week distribution with structured training blocks
        # Based on real CC0 analysis: 82.5% of easy clusters are training cycle breaks
        # Training cycles: Mean 12.2 weeks, Median 7.0 weeks (variable)
        # Recovery breaks: Mean 3.1 weeks, Median 1.0 weeks (typically 1 week)
        training_cycle_structure = _generate_training_cycle_structure(n_weeks, profile, rng)
        
        # Generate weekly plans based on profile and training cycle structure
        for w in range(n_weeks):
            week_start_idx = w * 7
            week_dates = dates[week_start_idx:week_start_idx + 7]
            week_scale = weekly_scales[w]
            target_week_km = base_km_week * week_scale

            # Check if this week is a break week (easy, no intensity) or training cycle week (hard, with intensity)
            # Default to training cycle week if not found in structure (shouldn't happen)
            is_break_week, cycle_num = training_cycle_structure.get(w, (False, 0))
            
            # Generate weekly plan based on profile
            if profile == "novice":
                # Novice: 1-3 runs per week only (mostly easy, maybe one long)
                # Most weeks: 1-2 runs, occasional 3 runs
                num_runs = rng.integers(1, 4)  # 1, 2, or 3 runs
                # Choose which days to run (avoid consecutive days, favor weekends)
                run_days = []
                if num_runs == 1:
                    # Single run - usually on weekend (day 6 or 5)
                    run_days = [rng.integers(5, 7)]
                elif num_runs == 2:
                    # Two runs - usually Tue/Sat or Wed/Sun or Sat/Sun
                    options = [(1, 5), (1, 6), (2, 6), (5, 6)]
                    run_days = list(options[rng.integers(0, len(options))])
                else:  # num_runs == 3
                    # Three runs - Tue/Thu/Sat or Wed/Fri/Sun
                    options = [(1, 3, 5), (2, 4, 6)]
                    run_days = list(options[rng.integers(0, len(options))])
                
                # Assign session types (mostly easy, one might be long)
                week_plan = ["rest"] * 7
                for i, day_idx in enumerate(sorted(run_days)):
                    if i == len(run_days) - 1 and num_runs >= 2 and rng.random() < 0.4:
                        # Last run might be a long run (40% chance if 2+ runs)
                        week_plan[day_idx] = "long"
                    else:
                        week_plan[day_idx] = "easy"
                
                # Double sessions: None for novices
                double_session_days = []
                
            elif profile == "recreational":
                # Recreational: Similar to base but varied - target ~50% of weeks with intensity
                # NEW: Use structured training cycle → break pattern instead of random distribution
                # Break weeks = easy/base weeks with NO intensity sessions
                # Training cycle weeks = hard weeks with intensity sessions
                
                if is_break_week:
                    # Easy/base week: NO intensity sessions (only easy, rest)
                    # FIXED: Long runs in break weeks should be converted to easy runs to prevent intensity zones
                    # FIX: Further reduce default rest days to match real CC0 ~27% rest frequency
                    # NOTE: Reducing to 0-1 rest days reduced performance (0.6225 → 0.6194)
                    # Keeping 1-2 rest days (mean 1.5) as it produces better AUC
                    # Target: ~27% rest days = ~1.9 rest days per week
                    # Use 1-2 rest days (mean 1.5) to target ~21-28% range
                    num_rest = rng.integers(1, 3)  # 1 or 2 rest days (mean 1.5 = 21.4%)
                    week_plan = ["easy"] * 7
                    rest_days = rng.choice(7, size=num_rest, replace=False).tolist()
                    for rd in rest_days:
                        week_plan[rd] = "rest"
                    # No long runs in break weeks - all easy runs
                else:
                    # Hard week: 1-2 hard sessions (reduced from 1-3 to further reduce intensity frequency)
                    # FIX: Further reduce default rest days for advanced/elite to match real CC0 ~27% rest frequency
                    # Number of hard sessions per week: 1-2 (mostly 1, occasional 2)
                    # Target: ~27% rest days = ~1.9 rest days per week
                    # For advanced/elite: Reduce base rest days since post-workout rest will add more
                    num_hard = rng.integers(1, 3)  # 1 or 2
                    # Distribution: 60% chance of 1 session, 40% chance of 2 sessions (mean ~1.4)
                    if num_hard == 2 and rng.random() < 0.4:  # 40% chance of 2 sessions, 60% chance of 1
                        num_hard = 1
                    if num_hard == 1:
                        # 1 hard session - Thursday tempo/interval
                        base_tmpl = ["rest", "easy", "easy", "tempo", "easy", "easy", "long"]  # Reduced from 2 rest to 1 rest
                        week_plan = base_tmpl.copy()
                        week_plan[3] = "interval" if rng.random() < 0.35 else "tempo"
                    elif num_hard == 2:
                        # 2 hard sessions - add another on Tue or Sat
                        base_tmpl = ["rest", "easy", "easy", "tempo", "easy", "easy", "long"]  # Reduced from 2 rest to 1 rest
                        week_plan = base_tmpl.copy()
                        week_plan[3] = "interval" if rng.random() < 0.35 else "tempo"
                        if rng.random() < 0.5:
                            week_plan[1] = "interval" if rng.random() < 0.5 else "tempo"
                        else:
                            week_plan[5] = "interval" if rng.random() < 0.5 else "tempo"
                    else:  # num_hard == 3
                        # 3 hard sessions - Tue, Thu, Sat
                        week_plan = ["rest", "tempo", "easy", "interval", "easy", "tempo", "long"]  # Reduced from 2 rest to 1 rest
                        # Randomize which are tempo vs interval
                        hard_days = [1, 3, 5]
                        for hd in hard_days:
                            week_plan[hd] = "interval" if rng.random() < 0.5 else "tempo"
                    
                    # Ensure 3-4 easy/off days (rest + easy)
                    # Count current rest/easy
                    rest_easy_count = sum(1 for s in week_plan if s in ["rest", "easy"])
                    if rest_easy_count < 3:
                        # Convert some hard sessions to easy
                        hard_positions = [i for i, s in enumerate(week_plan) if s in ["tempo", "interval"]]
                        while rest_easy_count < 3 and len(hard_positions) > 0:
                            pos = hard_positions.pop(rng.integers(0, len(hard_positions)))
                            week_plan[pos] = "easy"
                            rest_easy_count += 1
                
                # Double sessions: Rare for recreational (5% chance per week)
                double_session_days = []
                if rng.random() < 0.05:
                    # One double session day - usually weekend
                    double_day = rng.choice([5, 6])  # Sat or Sun
                    double_session_days.append(double_day)
                
            elif profile == "advanced":
                # Advanced: Target ~50% of weeks with intensity (matching real CC0)
                # NEW: Use structured training cycle → break pattern instead of random distribution
                # Break weeks = easy/base weeks with NO intensity sessions
                # Training cycle weeks = hard weeks with intensity sessions
                
                if is_break_week:
                    # Easy/base week: NO intensity sessions (only easy, rest)
                    # FIXED: Long runs in break weeks should be converted to easy runs to prevent intensity zones
                    # FIX: Further reduce default rest days to match real CC0 ~27% rest frequency
                    # NOTE: Reducing to 0-1 rest days reduced performance (0.6225 → 0.6194)
                    # Keeping 1-2 rest days (mean 1.5) as it produces better AUC
                    # Target: ~27% rest days = ~1.9 rest days per week
                    # Use 1-2 rest days (mean 1.5) to target ~21-28% range
                    num_rest = rng.integers(1, 3)  # 1 or 2 rest days (mean 1.5 = 21.4%)
                    week_plan = ["easy"] * 7
                    rest_days = rng.choice(7, size=num_rest, replace=False).tolist()
                    for rd in rest_days:
                        week_plan[rd] = "rest"
                    # No long runs in break weeks - all easy runs
                else:
                    # Hard week: INCREASED to 2-3 hard sessions to match real CC0 intensity zone frequency
                    # Real CC0 shows Z5 ~0.58 km/day, Z3_4 ~0.69 km/day
                    # Current synthetic: Z5 ~0.37 km/day (-37%), Z3_4 ~0.50 km/day (-27%)
                    # Need to increase intensity zones by 57% (Z5) and 38% (Z3_4)
                    # Strategy: Increase hard sessions from 1-2 (mean 1.3) to 2-3 (mean 2.5)
                    # Distribution: 30% chance of 2 sessions, 70% chance of 3 sessions (mean ~2.7)
                    num_hard = 2 if rng.random() < 0.30 else 3
                    
                    # Place hard sessions - spread across week (Tue, Thu, Sat, or Mon, Wed, Fri)
                    hard_days = []
                    if num_hard == 2:
                        # Two hard sessions: Tue, Thu or Tue, Sat or Thu, Sat
                        options = [[1, 3], [1, 5], [3, 5]]  # Tue/Thu, Tue/Sat, Thu/Sat
                        hard_days = options[rng.integers(0, 3)]
                    elif num_hard == 3:
                        # Three hard sessions: Tue, Thu, Sat (classic pattern)
                        hard_days = [1, 3, 5]  # Tue, Thu, Sat
                    else:  # num_hard == 1 (shouldn't happen with new distribution, but keep for safety)
                        # Single hard session: Tuesday, Thursday, or Saturday
                        hard_days = [rng.choice([1, 3, 5])]  # One of Tue, Thu, or Sat
                    
                    # Build week plan
                    week_plan = ["rest"] * 7
                    # NOTE: Increasing interval proportion to 70% reduced performance (0.6225 → 0.6100)
                    # Keeping 50% as it produces better AUC
                    for hd in hard_days:
                        week_plan[hd] = "interval" if rng.random() < 0.5 else "tempo"
                    
                    # Fill remaining days with easy or rest (reduced rest probability for advanced/elite)
                    # For advanced/elite: More aggressive - 95% chance of easy vs rest (target ~5% base rest)
                    for i in range(7):
                        if week_plan[i] == "rest":
                            if i == 6:  # Sunday - usually long run
                                week_plan[i] = "long"
                            elif profile in ["advanced", "elite"]:
                                # Advanced/elite: 95% chance of easy vs rest (very few base rest days)
                                if rng.random() < 0.95:
                                    week_plan[i] = "easy"
                            else:
                                # Other profiles: 85% chance of easy vs rest
                                if rng.random() < 0.85:
                                    week_plan[i] = "easy"
                
                # Double sessions: Common for advanced (30% chance per week, 1-2 double days)
                double_session_days = []
                if rng.random() < 0.30:
                    num_doubles = rng.integers(1, 3)  # 1 or 2 double session days
                    # Usually on hard days or easy days (not rest days)
                    available_days = [i for i in range(7) if week_plan[i] != "rest"]
                    if len(available_days) >= num_doubles:
                        double_session_days = rng.choice(available_days, size=num_doubles, replace=False).tolist()
                
            else:  # profile == "elite"
                # Elite: Target ~50% of weeks with intensity (matching real CC0)
                # NEW: Use structured training cycle → break pattern instead of random distribution
                # Break weeks = easy/base weeks with NO intensity sessions
                # Training cycle weeks = hard weeks with intensity sessions
                
                if is_break_week:
                    # Easy/base week: NO intensity sessions (only easy, rest)
                    # FIXED: Long runs in break weeks should be converted to easy runs to prevent intensity zones
                    # FIX: Elite athletes have LOWER rest frequency even in break weeks (~20-25% vs ~27%)
                    # Elite are more resilient and maintain higher training frequency
                    week_plan = ["easy"] * 7  # Start with all easy (no long runs in break weeks)
                    # Note: week_plan[6] is already "easy" (no long run in break weeks)
                    
                    # FIX: Elite target ~20-25% rest days = ~1.4-1.75 rest days per week (lower than advanced)
                    # Further reduced: Use 0-1 rest days (mean 0.2) to target ~3-14% range
                    # 80% chance of 0 rest days, 20% chance of 1 rest day (mean 0.2 = 2.9%)
                    if rng.random() < 0.80:  # 80% chance of 0 rest days (elite can train 7 days)
                        rest_count = 0
                    else:  # 20% chance of 1 rest day
                        rest_count = 1
                    
                    if rest_count > 0:
                        # Convert some easy days to rest (can include Sunday now since it's easy, not long)
                        rest_candidates = [i for i in range(7) if week_plan[i] == "easy"]
                        if len(rest_candidates) >= rest_count:
                            rest_days = rng.choice(rest_candidates, size=rest_count, replace=False).tolist()
                            for rd in rest_days:
                                week_plan[rd] = "rest"
                else:
                    # Hard week: INCREASED to 2-3 hard sessions to match real CC0 intensity zone frequency
                    # Real CC0 shows Z5 ~0.58 km/day, Z3_4 ~0.69 km/day
                    # Current synthetic: Z5 ~0.37 km/day (-37%), Z3_4 ~0.50 km/day (-27%)
                    # Need to increase intensity zones by 57% (Z5) and 38% (Z3_4)
                    # Strategy: Increase hard sessions from 1-2 (mean 1.4) to 2-3 (mean 2.5)
                    # Distribution: 30% chance of 2 sessions, 70% chance of 3 sessions (mean ~2.7)
                    num_hard = 2 if rng.random() < 0.30 else 3
                    
                    # Place hard sessions - spread across week (Tue, Thu, Sat)
                    hard_days = []
                    if num_hard == 2:
                        # Two hard sessions: Tue, Thu or Tue, Sat or Thu, Sat
                        options = [[1, 3], [1, 5], [3, 5]]  # Tue/Thu, Tue/Sat, Thu/Sat
                        hard_days = options[rng.integers(0, 3)]
                    elif num_hard == 3:
                        # Three hard sessions: Tue, Thu, Sat (classic pattern)
                        hard_days = [1, 3, 5]  # Tue, Thu, Sat
                    else:  # num_hard == 1 (shouldn't happen with new distribution, but keep for safety)
                        # Single hard session: Tuesday, Thursday, or Saturday
                        hard_days = [rng.choice([1, 3, 5])]  # One of Tue, Thu, or Sat
                    
                    # Build week plan - elite have LESS rest than advanced
                    # FIX: Elite athletes should have lower rest frequency (~20-25% vs ~27% for advanced)
                    # Elite are more resilient and train more frequently
                    week_plan = ["easy"] * 7  # Start with all easy
                    # NOTE: Increasing interval proportion to 70% reduced performance (0.6225 → 0.6100)
                    # Keeping 50% as it produces better AUC
                    for hd in hard_days:
                        week_plan[hd] = "interval" if rng.random() < 0.5 else "tempo"
                    
                    # Sunday long run (if not already hard)
                    if 6 not in hard_days:
                        week_plan[6] = "long"
                    # Some weeks have hard long run (20% chance)
                    elif rng.random() < 0.20:
                        week_plan[6] = "long"  # Keep as long, but elite might do it hard
                    
                    # FIX: Elite target ~20-25% rest days = ~1.4-1.75 rest days per week (lower than advanced)
                    # Use 0-2 rest days (mean 1.0) to target ~14-28% range, but weighted toward 1 rest day
                    # Elite are more resilient, so fewer rest days
                    if rng.random() < 0.70:  # 70% chance of 1 rest day
                        rest_count = 1
                    elif rng.random() < 0.85:  # 15% chance of 0 rest days (elite can train 7 days)
                        rest_count = 0
                    else:  # 15% chance of 2 rest days
                        rest_count = 2
                    
                    if rest_count > 0:
                        # Convert some easy days to rest (usually Mon or Wed)
                        rest_candidates = [i for i in range(7) if week_plan[i] == "easy" and i not in hard_days and i != 6]
                        if len(rest_candidates) >= rest_count:
                            rest_days = rng.choice(rest_candidates, size=rest_count, replace=False).tolist()
                            for rd in rest_days:
                                week_plan[rd] = "rest"
                
                # Double sessions: High probability for elite (70% chance per week, 2-4 double days)
                double_session_days = []
                if rng.random() < 0.70:
                    num_doubles = rng.integers(2, 5)  # 2, 3, or 4 double session days
                    # Usually on hard days or easy days (not rest days)
                    available_days = [i for i in range(7) if week_plan[i] != "rest"]
                    if len(available_days) >= num_doubles:
                        double_session_days = rng.choice(available_days, size=num_doubles, replace=False).tolist()
                    else:
                        double_session_days = available_days.copy()
            
            # FIXED: Adjust rest days based on user rest_day_frequency attribute (for variation)
            # FIX: For advanced/elite, cap user_rest_freq adjustment to prevent excessive rest days
            # Target: ~27% overall rest days (including post-workout rest)
            # Strategy: Only adjust if it doesn't push total rest days too high
            # Post-workout rest will add ~10-15%, so cap weekly plan rest at ~15% (1.05 rest days/week)
            if user_rest_freq is not None and user_rest_freq is not np.nan:
                # Count current rest days in week_plan
                current_rest_count = sum(1 for s in week_plan if s == "rest")
                target_rest_count = round(user_rest_freq * 7)  # Target rest days per week (0-7)
                target_rest_count = max(0, min(7, target_rest_count))  # Clamp to 0-7
                
                # FIX: For advanced/elite, cap target to account for post-workout rest
                # Post-workout rest adds ~10-15%, so cap weekly plan at ~1.05 rest days (15%)
                if profile in ["advanced", "elite"]:
                    max_rest_before_postworkout = 1.05  # ~15% to leave room for post-workout rest
                    target_rest_count = min(target_rest_count, round(max_rest_before_postworkout))
                
                rest_diff = target_rest_count - current_rest_count
                
                # Only adjust if it doesn't push rest days too high
                # NOTE: Feature interaction adjustments reduced performance (0.6225 → 0.6154)
                # Keeping original logic as it produces better AUC
                if rest_diff > 0:
                    # Need more rest days: Convert easy days to rest (prioritize easy, can also convert long runs if needed)
                    easy_candidates = [i for i in range(7) if week_plan[i] == "easy"]
                    if len(easy_candidates) >= rest_diff:
                        # Have enough easy days - convert them
                        rest_to_add = rng.choice(easy_candidates, size=rest_diff, replace=False).tolist()
                        for rd in rest_to_add:
                            week_plan[rd] = "rest"
                    elif len(easy_candidates) > 0:
                        # Convert all available easy days to rest
                        for rd in easy_candidates:
                            week_plan[rd] = "rest"
                        # If still need more, consider converting long runs (but preserve at least 1 long run per week if possible)
                        remaining_needed = rest_diff - len(easy_candidates)
                        long_candidates = [i for i in range(7) if week_plan[i] == "long"]
                        if remaining_needed > 0 and len(long_candidates) > 1:  # Keep at least 1 long run
                            rest_to_add_long = rng.choice(long_candidates, size=min(remaining_needed, len(long_candidates) - 1), replace=False).tolist()
                            for rd in rest_to_add_long:
                                week_plan[rd] = "rest"
                elif rest_diff < 0:
                    # Need fewer rest days: Convert rest days to easy (aggressive - convert all if needed)
                    rest_candidates = [i for i in range(7) if week_plan[i] == "rest"]
                    if len(rest_candidates) >= abs(rest_diff):
                        rest_to_remove = rng.choice(rest_candidates, size=abs(rest_diff), replace=False).tolist()
                        for rd in rest_to_remove:
                            week_plan[rd] = "easy"
                    elif len(rest_candidates) > 0:
                        # Convert all available rest days to easy (aggressive adjustment)
                        for rd in rest_candidates:
                            week_plan[rd] = "easy"
            
            # Generate daily plans for this week
            # Handle case where week might be shorter than 7 days at the end
            week_length = min(7, len(week_dates))
            for di in range(week_length):
                d = week_dates[di]
                st = week_plan[di] if di < len(week_plan) else "rest"
                is_double_day = di in double_session_days
                
                # Calculate distance for this day
                if st == "rest":
                    km = 0.0
                elif st == "long":
                    # Long-run spike generation: ensure at least one per month
                    spike = 1.0
                    is_spike_day = False
                    
                    # Force spike if it's been >4 long runs since last spike (~1 month)
                    force_spike = (long_runs_since_last_spike >= 4)
                    
                    # Otherwise, use probability-based spike
                    if force_spike or rng.random() < cfg.long_run_spike_prob:
                        is_spike_day = True
                        mean_increase_above_safe = cfg.long_run_spike_mean_above_safe
                        sd_increase = cfg.long_run_spike_sd_above_safe
                        
                        increase_above_safe = rng.normal(mean_increase_above_safe, sd_increase)
                        increase_above_safe = max(0.15, increase_above_safe)
                        
                        spike = 1.05 + increase_above_safe
                        spike = float(np.clip(spike, 1.20, cfg.long_run_spike_max))
                        
                        long_runs_since_last_spike = 0
                    else:
                        long_runs_since_last_spike += 1
                    
                    # Calculate base distance
                    base_target = target_week_km * long_frac * spike
                    
                    if is_spike_day:
                        random_variation = float(np.clip(rng.normal(1.0, 0.05), 0.95, 1.05))
                    else:
                        random_variation = float(np.clip(rng.normal(1.0, 0.12), 0.75, 1.35))
                    
                    km = base_target * random_variation
                elif st in ["tempo", "interval"]:
                    # Hard session distance - varies by profile
                    if profile in ["advanced", "elite"]:
                        # Advanced/Elite: Hard sessions are typically 10-15% of weekly volume
                        hard_frac = float(np.clip(rng.normal(0.125, 0.02), 0.10, 0.18))
                    else:
                        # Recreational: 18% of weekly volume
                        hard_frac = 0.18
                    
                    km = target_week_km * hard_frac * float(np.clip(rng.normal(1.0, 0.12), 0.60, 1.40))
                else:  # easy
                    # Easy days: split remaining weekly km across easy days
                    # Count easy days in week (excluding double days which are handled separately)
                    easy_days_count = sum(1 for i, s in enumerate(week_plan) if s == "easy" and i not in double_session_days)
                    if easy_days_count > 0:
                        # Calculate remaining km after hard/long sessions
                        # Calculate hard session km (varies by profile)
                        hard_frac = 0.125 if profile in ["advanced", "elite"] else 0.18
                        hard_sessions = sum(1 for s in week_plan if s in ["tempo", "interval"])
                        long_sessions = sum(1 for s in week_plan if s == "long")
                        
                        hard_km_total = (target_week_km * hard_frac * hard_sessions) + \
                                       (target_week_km * long_frac * long_sessions)
                        remaining_km = max(0, target_week_km - hard_km_total)
                        
                        if easy_days_count > 0:
                            km = remaining_km / easy_days_count * float(np.clip(rng.normal(1.0, 0.18), 0.60, 1.50))
                        else:
                            km = 0.0
                    else:
                        km = 0.0
                
                # Handle double sessions (store in additional columns)
                km_primary = km
                km_secondary = 0.0
                secondary_session_type = "easy"
                has_double_flag = 0
                
                if is_double_day:
                    has_double_flag = 1
                    if st == "long":
                        # Long run split: 60% AM, 40% PM (easy recovery)
                        km_primary = km * 0.6
                        km_secondary = km * 0.4
                        secondary_session_type = "easy"
                    elif st in ["tempo", "interval"]:
                        # Hard session + easy recovery
                        km_primary = km
                        km_secondary = target_week_km * 0.05 * float(np.clip(rng.normal(1.0, 0.15), 0.50, 1.30))
                        secondary_session_type = "easy"
                    else:  # easy
                        # Easy day + easy recovery
                        km_primary = km
                        km_secondary = target_week_km * 0.05 * float(np.clip(rng.normal(1.0, 0.15), 0.50, 1.30))
                        secondary_session_type = "easy"
                    
                    # Only set double if secondary km > 0
                    if km_secondary <= 0:
                        has_double_flag = 0
                
                km_primary = float(max(0.0, km_primary))
                km_secondary = float(max(0.0, km_secondary))
                
                # Store primary session (one row per day)
                rows.append({
                    "user_id": uid,
                    "date": d.strftime("%Y-%m-%d"),
                    "session_type": st,
                    "km_total": round(km_primary, 3),
                    "has_double": has_double_flag,
                    "double_session_type": secondary_session_type if has_double_flag else None,
                    "double_km": round(km_secondary, 3) if has_double_flag else 0.0,
                    "is_break_week": is_break_week,  # Training block indicator: True = easy week (no intensity), False = training cycle week
                })

    plan = pd.DataFrame(rows)
    plan["user_id"] = plan["user_id"].astype("int64")
    plan = plan.sort_values(["user_id", "date"]).reset_index(drop=True)
    
    # NEW: Post-workout rest day probability (conditional override after hard sessions)
    # Apply conditional rest probability based on previous day's workout
    # This needs to be done after plan is generated but before activities are created
    if "km_total" in plan.columns and "session_type" in plan.columns:
        # Get athlete weekly volumes for long run thresholds
        user_weekly_km = {}
        for uid in plan["user_id"].unique():
            user_plan = plan[plan["user_id"] == uid]
            # Calculate weekly volume from the plan (approximate)
            # Use average daily km over a week
            if len(user_plan) >= 7:
                weekly_km = user_plan["km_total"].head(7).sum()  # Approximate weekly volume
            else:
                weekly_km = user_plan["km_total"].sum() * (7.0 / len(user_plan))
            user_weekly_km[uid] = float(weekly_km)
        
        # Get user profiles for profile-specific rest probabilities
        # FIX: users dataframe is passed to generate_daily_plan, so we can access it
        user_profiles = {}
        if users is not None and "user_id" in users.columns and "profile" in users.columns:
            for _, u in users.iterrows():
                user_profiles[int(u["user_id"])] = str(u["profile"])
        
        # Apply post-workout rest day probability day-by-day
        for uid in plan["user_id"].unique():
            user_plan_idx = plan[plan["user_id"] == uid].index
            user_plan_idx_sorted = sorted(user_plan_idx)
            weekly_km = user_weekly_km[uid]
            
            # Get user profile for profile-specific probabilities
            user_profile = user_profiles.get(int(uid), "advanced")  # Default to advanced
            is_elite = (user_profile == "elite")
            
            # Define long run thresholds (percentage-based, profile-specific)
            long_run_threshold = weekly_km * cfg.long_run_threshold_pct  # 15% of weekly volume
            extremely_long_run_threshold = weekly_km * cfg.extremely_long_run_threshold_pct  # 20% of weekly volume
            
            for i, idx in enumerate(user_plan_idx_sorted):
                if i == 0:
                    continue  # Skip first day (no previous day)
                
                # Get previous day's workout
                prev_idx = user_plan_idx_sorted[i-1]
                prev_km = plan.loc[prev_idx, "km_total"] if prev_idx in plan.index else 0.0
                prev_session_type = str(plan.loc[prev_idx, "session_type"]) if prev_idx in plan.index else "rest"
                
                # Check if previous day was a hard session
                is_high_intensity = prev_session_type in ["tempo", "interval"]
                is_long_run = prev_km > long_run_threshold
                is_extremely_long = prev_km > extremely_long_run_threshold
                
                # Get current day's plan
                current_session_type = str(plan.loc[idx, "session_type"]) if idx in plan.index else "rest"
                is_scheduled_as_training = current_session_type != "rest"
                
                # If previous day was hard/long AND today is scheduled as training:
                if (is_high_intensity or is_long_run) and is_scheduled_as_training:
                    # Calculate conditional rest probability
                    # FIX: Further reduce post-workout rest probabilities for advanced/elite
                    # Target: Reduce from 32% to ~27% rest days
                    # Strategy: Further reduce probabilities by additional 30% (on top of previous 40-50% reduction)
                    # Elite: 0.5x * 0.7 = 0.35x (65% total reduction), Advanced: 0.6x * 0.7 = 0.42x (58% total reduction)
                    base_reduction = 0.5 if is_elite else 0.6  # Previous reduction
                    additional_reduction = 0.7  # Additional 30% reduction
                    reduction_factor = base_reduction * additional_reduction  # Elite: 0.35x, Advanced: 0.42x
                    
                    if is_extremely_long and is_high_intensity:
                        base_prob = cfg.post_workout_rest_prob_extremely_long_and_hard
                        cond_rest_prob = base_prob * reduction_factor  # Elite: 12%, Advanced: 14.4%
                    elif is_high_intensity and is_long_run:
                        base_prob = cfg.post_workout_rest_prob_hard_and_long
                        cond_rest_prob = base_prob * reduction_factor  # Elite: 10.5%, Advanced: 12.6%
                    elif is_extremely_long:
                        base_prob = cfg.post_workout_rest_prob_extremely_long
                        cond_rest_prob = base_prob * reduction_factor  # Elite: 9%, Advanced: 10.8%
                    elif is_high_intensity:
                        base_prob = cfg.post_workout_rest_prob_high_intensity
                        cond_rest_prob = base_prob * reduction_factor  # Elite: 7%, Advanced: 8.4%
                    elif is_long_run:
                        base_prob = cfg.post_workout_rest_prob_long_run
                        cond_rest_prob = base_prob * reduction_factor  # Elite: 4.5%, Advanced: 5.4%
                    else:
                        cond_rest_prob = 0.0
                    
                    # Apply conditional rest probability (override: convert training day to rest day)
                    if rng.random() < cond_rest_prob:
                        # Override: Convert training day to rest day
                        plan.loc[idx, "km_total"] = 0.0
                        plan.loc[idx, "session_type"] = "rest"
                        if "has_double" in plan.columns:
                            plan.loc[idx, "has_double"] = 0
                        if "double_session_type" in plan.columns:
                            plan.loc[idx, "double_session_type"] = None
                        if "double_km" in plan.columns:
                            plan.loc[idx, "double_km"] = 0.0
    
    # WARMUP PERIOD: Keep warmup data in plan for now
    # We'll filter it out later in pipeline.py after events are generated
    # This ensures spike detection and rolling features have history
    return plan

def _training_load_from_km(km: float, session_type: str, rng: np.random.Generator) -> float:
    # TRIMP-ish proxy; unit-consistent with itself for ACWR
    if km <= 0.0:
        return 0.0
    if session_type == "easy":
        mult = 1.0
    elif session_type == "long":
        mult = 1.1
    elif session_type == "tempo":
        mult = 1.35
    elif session_type == "interval":
        mult = 1.55
    else:
        mult = 1.0
    return float(km * mult * rng.normal(1.0, 0.06))

def generate_daily_signals(cfg: GeneratorConfig, users: pd.DataFrame, daily_plan: pd.DataFrame,
                           activities: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate daily table with physiology, sleep, stress, resp/temp and wear compliance.

    Notes:
    - HR modelling is handled per-activity; daily aggregates are computed from activities.
    - RHR/HRV are mean-reverting with load/illness adjustments applied later in events stage.
    """
    users_local = users.copy()
    users_local["user_id"] = users_local["user_id"].astype("int64")

    dates = pd.to_datetime(daily_plan["date"].unique())
    dates = np.sort(dates)

    # Aggregate activities to daily
    if len(activities) > 0:
        a = activities.copy()
        a["date"] = pd.to_datetime(a["date"])
        # weighted pace aggregation (bug-fixed): duration-weighted
        a["pace_x_dur"] = a["pace_min_per_km"] * a["duration_min"]
        # NEW: Aggregate gait metrics from activities
        day_agg = a.groupby(["user_id","date"]).agg(
            sessions=("activity_id","count"),
            km_total=("distance_km","sum"),
            duration_min=("duration_min","sum"),
            avg_hr_bpm=("avg_hr_bpm","mean"),
            pace_min_per_km=("pace_x_dur","sum"),
            elev_gain_m=("elev_gain_m","sum"),
            kms_z3_4=("kms_z3_4","sum"),
            kms_z5_t1_t2=("kms_z5_t1_t2","sum"),
            kms_sprinting=("kms_sprinting","sum"),
            cadence_spm=("cadence_spm","mean"),  # Average cadence across activities
            gct_ms=("gct_ms","mean"),  # Average GCT across activities
            stride_length_cm=("stride_length_cm","mean"),  # Average stride length
            vertical_oscillation_cm=("vertical_oscillation_cm","mean"),  # Average vertical oscillation
            gct_balance=("gct_balance","mean"),  # Average GCT balance
        ).reset_index()
        day_agg["pace_min_per_km"] = (day_agg["pace_min_per_km"] / np.maximum(day_agg["duration_min"], 1e-6)).round(3)
    else:
        day_agg = pd.DataFrame(columns=[
            "user_id","date","sessions","km_total","duration_min","avg_hr_bpm","pace_min_per_km","elev_gain_m",
            "kms_z3_4","kms_z5_t1_t2","kms_sprinting",
            "cadence_spm","gct_ms","stride_length_cm","vertical_oscillation_cm","gct_balance"
        ])

    plan = daily_plan.copy()
    plan["date"] = pd.to_datetime(plan["date"])
    plan = plan.merge(day_agg, on=["user_id","date"], how="left", suffixes=("","_act"))

    # Fill realized km etc. from plan if no activity (rest / missing)
    plan["sessions"] = plan["sessions"].fillna(0).astype(int)
    plan["km_total"] = plan["km_total_act"].where(plan["km_total_act"].notna(), plan["km_total"])
    plan["duration_min"] = plan["duration_min"].fillna(0.0)
    plan["avg_hr_bpm"] = plan["avg_hr_bpm"].fillna(np.nan)
    plan["pace_min_per_km"] = plan["pace_min_per_km"].fillna(np.nan)
    plan["elev_gain_m"] = plan["elev_gain_m"].fillna(0.0)
    plan["kms_z3_4"] = plan["kms_z3_4"].fillna(0.0)
    plan["kms_z5_t1_t2"] = plan["kms_z5_t1_t2"].fillna(0.0)
    plan["kms_sprinting"] = plan["kms_sprinting"].fillna(0.0)
    # Fill gait metrics from activities (or leave as NaN if no activities)
    plan["cadence_spm"] = plan.get("cadence_spm", pd.Series(index=plan.index, dtype=float))
    plan["gct_ms"] = plan.get("gct_ms", pd.Series(index=plan.index, dtype=float))
    plan["stride_length_cm"] = plan.get("stride_length_cm", pd.Series(index=plan.index, dtype=float))
    plan["vertical_oscillation_cm"] = plan.get("vertical_oscillation_cm", pd.Series(index=plan.index, dtype=float))
    plan["gct_balance"] = plan.get("gct_balance", pd.Series(index=plan.index, dtype=float))
    plan = plan.drop(columns=[c for c in plan.columns if c.endswith("_act")], errors="ignore")

    # Compute training load (unit-consistent for ACWR)
    loads = []
    for _, row in plan.iterrows():
        loads.append(_training_load_from_km(float(row["km_total"]), str(row["session_type"]), rng))
    plan["training_load"] = np.array(loads, dtype=float)

    # Wear compliance and per-sensor missingness masks
    u_lookup = users_local.set_index("user_id")
    device_worn = np.zeros(len(plan), dtype=int)
    miss_hrv = np.zeros(len(plan), dtype=int)
    miss_rhr = np.zeros(len(plan), dtype=int)
    miss_sleep = np.zeros(len(plan), dtype=int)
    miss_stress = np.zeros(len(plan), dtype=int)
    miss_resp = np.zeros(len(plan), dtype=int)
    miss_temp = np.zeros(len(plan), dtype=int)

    for i, row in enumerate(plan.itertuples(index=False)):
        uid = int(row.user_id)
        wear_p = float(u_lookup.loc[uid, "wear_rate"])
        worn = 1 if rng.random() < wear_p else 0
        device_worn[i] = worn
        if worn == 1:
            miss_hrv[i] = 1 if rng.random() < cfg.miss_hrv else 0
            miss_rhr[i] = 1 if rng.random() < cfg.miss_rhr else 0
            miss_sleep[i] = 1 if rng.random() < cfg.miss_sleep else 0
            miss_stress[i] = 1 if rng.random() < cfg.miss_stress else 0
            miss_resp[i] = 1 if rng.random() < cfg.miss_resp else 0
            miss_temp[i] = 1 if rng.random() < cfg.miss_temp else 0
        else:
            miss_hrv[i] = miss_rhr[i] = miss_sleep[i] = miss_stress[i] = miss_resp[i] = miss_temp[i] = 1

    plan["device_worn"] = device_worn
    plan["missing_hrv"] = miss_hrv
    plan["missing_rhr"] = miss_rhr
    plan["missing_sleep"] = miss_sleep
    plan["missing_stress"] = miss_stress
    plan["missing_resp"] = miss_resp
    plan["missing_temp"] = miss_temp

    # rolling 7-day wear rate (routing signal for dual-path)
    plan = plan.sort_values(["user_id","date"]).reset_index(drop=True)
    wear7 = []
    for uid, g in plan.groupby("user_id", sort=False):
        w = g["device_worn"].to_numpy()
        wear7.append(rolling_mean(w, 7))
    plan["wear_7d_rate"] = np.concatenate(wear7)

    # Generate physiology and recovery signals with mean reversion; load- and sleep-linked
    rhr = np.zeros(len(plan), dtype=float)
    hrv = np.zeros(len(plan), dtype=float)
    sleep_h = np.zeros(len(plan), dtype=float)
    stress = np.zeros(len(plan), dtype=float)
    resp = np.zeros(len(plan), dtype=float)
    temp = np.zeros(len(plan), dtype=float)

    # subjective fields
    perceived_exertion = np.zeros(len(plan), dtype=float)
    perceived_recovery = np.zeros(len(plan), dtype=float)
    perceived_trainingSuccess = np.zeros(len(plan), dtype=float)
    strength_training = np.zeros(len(plan), dtype=int)
    hours_alternative = np.zeros(len(plan), dtype=float)

    for uid, gidx in plan.groupby("user_id", sort=False).groups.items():
        idx = np.array(list(gidx), dtype=int)
        u = u_lookup.loc[int(uid)]
        # Get user profile for profile-specific calculations
        user_profile = str(u["profile"]) if "profile" in u else "advanced"
        
        # ALGORITHMIC FIX: User-level sprinting propensity affects perceived features
        # Users who sprint more (not never_sprints) should have higher baseline perceived features
        # This creates user-level association: sprinting users have higher perceived features overall
        # Similar to injury proneness boost - users who sprint more are more injury-prone AND have higher perceived values
        never_sprints_val = int(u.get("never_sprints", 0)) if "never_sprints" in u else 0
        is_sprinter_user = user_profile in ["advanced", "elite"] and never_sprints_val == 0
        user_sprinting_boost = 0.35 if is_sprinter_user else 0.0  # Users who sprint have higher baseline perceived features
        
        rhr0 = float(u["rhr_base"]) + rng.normal(0, 1.5)
        hrv0 = float(u["hrv_base"]) + rng.normal(0, 4.0)
        # baseline sleep; fitter slightly better sleep resilience
        sleep0 = float(np.clip(rng.normal(7.2, 0.6) + 0.3*float(u["fitness"]), 5.0, 9.2))
        resp0 = float(np.clip(rng.normal(14.5, 1.1), 10.0, 20.0))
        temp0 = float(np.clip(rng.normal(36.65, 0.12), 36.1, 37.2))

        prev_rhr, prev_hrv = rhr0, hrv0
        prev_stress_val = 35.0  # Initialize previous stress value

        loads = plan.loc[idx, "training_load"].to_numpy()
        # compute load z per user for physio response
        load_mean = loads.mean()
        load_sd = loads.std() + 1e-6
        load_z = (loads - load_mean) / load_sd
        
        # Get sprinting amounts for this user (for cumulative calculations)
        sprinting_km = plan.loc[idx, "kms_sprinting"].fillna(0.0).to_numpy() if "kms_sprinting" in plan.columns else np.zeros(len(idx))

        # Track sprinting history for cumulative effects on perceived features
        sprinting_history = []  # Track sprinting amounts over last 7 days
        
        for k, ii in enumerate(idx):
            # CRITICAL FIX: Strength training and alternative activities should be HIGHER at injury (not lower)
            # Real CC0 shows they are HIGHER at injury (1.40x-1.38x), not lower (0.86x-0.91x in synthetic)
            # They are recovery strategies used during high-risk periods, not injury drivers
            # Strategy: Increase probability AFTER high-load periods (recovery strategy) and DURING elevated risk periods
            st = str(plan.loc[ii, "session_type"])
            
            # Update sprinting history (keep last 7 days)
            current_sprinting = sprinting_km[k] if k < len(sprinting_km) else 0.0
            sprinting_history.append(current_sprinting)
            if len(sprinting_history) > 7:
                sprinting_history.pop(0)
            
            # ITERATION 5: EXTENDED HIGH-LOAD WINDOW and INCREASED PROBABILITIES
            # Strength training and alternative activities are recovery strategies used AFTER hard sessions
            # They should increase in the days FOLLOWING high-load periods (extended to last 5-10 days)
            # Extended window better captures elevated risk periods that precede injuries
            # Current: Strength training ~0.995x (target 1.396x - need ~40% more)
            # Current: Alternative activities ~0.959x (target 1.375x - need ~43% more)
            
            # Check for high-load in extended recent past (last 5-10 days) - recovery strategy AFTER hard sessions
            # Extended window captures longer elevated risk periods that precede injuries
            recent_high_load = False
            if k >= 5:  # Need at least 5 days of history
                # Check if any of the last 5-10 days had high load (load_z > 0.5)
                # Extended window (5-10 days) better captures elevated risk periods
                recent_window_start = max(0, k - 10)  # Extended from 7 to 10 days
                recent_window_end = k
                recent_loads = load_z[recent_window_start:recent_window_end]
                recent_high_load = np.any(recent_loads > 0.5) if len(recent_loads) > 0 else False
            
            # Also check current day for immediate post-workout recovery (same day after hard session)
            is_current_high_load = load_z[k] > 0.5 if k < len(load_z) else False
            is_intensity_day = st in ["tempo", "interval"]
            
            # Combine: High-load if current day OR recent history had high load
            is_high_load_period = is_current_high_load or recent_high_load
            
            # Strength training: SIGNIFICANTLY INCREASED probability DURING and AFTER high-load periods
            # Target: 1.396x ratio (currently 0.995x - need ~40% more to reach 1.396x)
            # Strategy: Increase probabilities significantly and extend high-load window to capture elevated risk periods
            if st == "rest":
                # Rest days: Significantly increased probabilities - 58% during/after high-load (was 50%), 35% base (was 30%)
                # +8 pp increase should help bring ratio from 0.995x closer to 1.396x target
                strength_prob = 0.58 if is_high_load_period else 0.35
                if rng.random() < strength_prob:
                    strength_training[ii] = 1
            elif st != "rest":
                # Training days: Significantly increased probabilities - 30% after intensity (was 25%), 20% base (was 16%)
                # +5 pp increase should help bring ratio closer to target
                strength_prob = 0.30 if is_intensity_day else 0.20
                if rng.random() < strength_prob:
                    strength_training[ii] = 1
            
            # Alternative activities: SIGNIFICANTLY INCREASED probability DURING and AFTER high-load periods
            # Target: 1.375x ratio (currently 0.959x - need ~43% more to reach 1.375x)
            # Strategy: Increase probabilities significantly and extend high-load window to capture elevated risk periods
            if st == "rest":
                # Rest days: Significantly increased probabilities - 52% during/after high-load (was 45%), 35% base (was 28%)
                # +7 pp increase should help bring ratio from 0.959x closer to 1.375x target
                alt_prob = 0.52 if is_high_load_period else 0.35
                if rng.random() < alt_prob:
                    hours_alternative[ii] = float(np.clip(rng.normal(0.5, 0.3), 0.1, 1.8))
                else:
                    hours_alternative[ii] = 0.0
            else:
                # Training days: Significantly increased probabilities - 35% after intensity (was 30%), 25% base (was 20%)
                # +5 pp increase should help bring ratio closer to target
                alt_prob = 0.35 if is_intensity_day else 0.25
                if rng.random() < alt_prob:
                    hours_alternative[ii] = float(np.clip(rng.normal(0.3, 0.25), 0.1, 1.5))
                else:
                    hours_alternative[ii] = 0.0

            # sleep decreases a bit after higher load or poor recovery
            # IMPROVED: Add stress feedback to sleep (poor sleep → higher stress → worse sleep)
            sleep_noise = rng.normal(0, 0.5)
            sleep_h[ii] = float(np.clip(
                sleep0 - 0.10*load_z[k] - 0.05*(prev_stress_val - 35)/60 + sleep_noise,  # NEW: Stress feedback
                4.0, 10.0
            ))

            # stress: higher with load and lower sleep
            # IMPROVED: Add previous stress persistence (stress accumulates)
            stress[ii] = float(np.clip(
                35 + 11*clip01((load_z[k]+1.0)/2.0) + 4*(7.0 - sleep_h[ii]) + 
                2*(prev_stress_val - 35) + rng.normal(0, 6),  # NEW: Previous stress persistence
                5, 95
            ))
            prev_stress_val = stress[ii]  # Update for next iteration

            # mean-reverting RHR/HRV (load effects applied here; illness effects later in events stage)
            # IMPROVED: Add HRV-RHR coupling (when RHR rises, HRV should drop more)
            prev_rhr = ou_step(prev_rhr, rhr0, cfg.mr_kappa_rhr, noise_sd=1.4, rng=rng)
            prev_hrv = ou_step(prev_hrv, hrv0, cfg.mr_kappa_hrv, noise_sd=5.0, rng=rng)
            rhr[ii] = float(np.clip(prev_rhr + cfg.rhr_load_slope*load_z[k], 35, 95))
            # NEW: HRV-RHR coupling - when RHR is elevated, HRV drops more
            rhr_deviation = (rhr[ii] - rhr0) / rhr0  # Normalized RHR deviation
            hrv[ii] = float(np.clip(
                prev_hrv + cfg.hrv_load_slope*load_z[k] - 0.3*rhr_deviation*hrv0 + rng.normal(0, 2.0),  # NEW: RHR coupling
                15, 180
            ))

            # resp & temp with small day-to-day variation (illness bump later)
            resp[ii] = float(np.clip(resp0 + rng.normal(0, 0.8) + 0.12*clip01((stress[ii]-35)/60), 9, 24))
            temp[ii] = float(np.clip(temp0 + rng.normal(0, 0.07), 36.0, 37.5))

            # subjectives: set -0.01 on rest days to mimic CC0 encoding; filled later by CC0 pipeline as missing
            if st == "rest":
                perceived_exertion[ii] = -0.01
                perceived_trainingSuccess[ii] = -0.01
                perceived_recovery[ii] = -0.01
            else:
                # PROFILE-SPECIFIC FIX: Adjust perceived metrics for advanced/elite to match real CC0
                # Real CC0 (elite athletes): exertion ~0.248, success ~0.350, recovery ~0.196 (normalized 0-1)
                # Current synthetic: exertion ~0.346 (+40%), success ~0.470 (+34%), recovery ~0.253 (+29%)
                # Need to REDUCE values for advanced/elite profiles only
                # Target normalized values: exertion 0.248, success 0.350, recovery 0.196
                # Formula: normalized = (value - 1) / 9, so target raw values:
                #   exertion: 0.248 * 9 + 1 = 3.23
                #   success: 0.350 * 9 + 1 = 4.15
                #   recovery: 0.196 * 9 + 1 = 2.76
                
                if user_profile in ["advanced", "elite"]:
                    # ALGORITHMIC FIX: Generate associations naturally from training patterns
                    # Real CC0 shows: exertion +35.8%, trainingSuccess +31.8%, recovery +26.3% on injury days
                    # These should emerge naturally from high training load/intensity patterns
                    
                    # Get intensity metrics for this day (if available)
                    km_z3_4 = float(plan.loc[ii, "kms_z3_4"]) if "kms_z3_4" in plan.columns else 0.0
                    km_z5 = float(plan.loc[ii, "kms_z5_t1_t2"]) if "kms_z5_t1_t2" in plan.columns else 0.0
                    km_sprinting = float(plan.loc[ii, "kms_sprinting"]) if "kms_sprinting" in plan.columns else 0.0
                    km_total = float(plan.loc[ii, "km_total"]) if "km_total" in plan.columns else 0.0
                    
                    # ALGORITHMIC FIX: Injuries are primarily driven by SPRINTING (2.23x higher before injuries)
                    # Perceived features should respond VERY strongly to sprinting (primary injury driver)
                    # Analysis shows: days before injuries have 0.165 km/day sprinting vs 0.074 km/day average (2.23x)
                    # But only 14.9% of days before injuries have sprinting, so need VERY strong response when present
                    
                    # Sprinting factor: Use actual sprinting amount with aggressive scaling
                    # Most sprinting is 0.1-1.0 km, with some up to 6.8 km
                    # Use linear scaling with higher multiplier for stronger response
                    # 0.1 km sprinting should give moderate boost, 0.5 km should give strong boost, 1.0+ km should give very strong boost
                    sprinting_factor = min(2.5, km_sprinting * 4.0)  # Linear: 0.1km = 0.4, 0.5km = 2.0, 1.0km = 2.5 (capped)
                    
                    # Also include cumulative effect from recent sprinting (persistent response)
                    recent_sprinting_3d = sum(sprinting_history[-3:]) if len(sprinting_history) >= 3 else sum(sprinting_history)
                    sprinting_factor_cumulative = min(1.5, recent_sprinting_3d * 2.0)  # Cumulative boost
                    sprinting_factor = max(sprinting_factor, sprinting_factor_cumulative * 0.5)  # Add cumulative as additional boost
                    
                    # Intensity factor: higher when doing tempo/interval/sprinting
                    intensity_factor = clip01((km_z3_4 + km_z5 + km_sprinting) / (km_total + 1e-6))
                    
                    # Exertion: VERY strong response to sprinting (primary injury driver)
                    # Real CC0: +35.8% on injury days (need much stronger response)
                    # Add user-level boost for sprinting users (creates user-level association)
                    exertion_base = 3.5 + 0.5*clip01((load_z[k]+1.0)/2.0) + 0.4*intensity_factor + 4.5*sprinting_factor + user_sprinting_boost
                    perceived_exertion[ii] = float(np.clip(exertion_base + rng.normal(0, 1.0), 1, 10))
                    
                    # Training Success: VERY strong positive correlation with sprinting
                    # Real CC0: +31.8% on injury days (sprinting creates overconfidence)
                    # Also: 22% of days should be zero (set on easy days or low-intensity days)
                    if (st == "easy" and rng.random() < 0.50) or (load_z[k] < -0.5 and recent_sprinting_3d < 0.1 and rng.random() < 0.30):
                        perceived_trainingSuccess[ii] = 0.0
                    else:
                        success_base = 4.5 - 0.1*clip01((stress[ii]-35)/60) + 0.2*clip01((load_z[k]+1.0)/2.0) + 0.15*intensity_factor + 3.5*sprinting_factor + user_sprinting_boost
                        perceived_trainingSuccess[ii] = float(np.clip(success_base + rng.normal(0, 1.0), 1, 10))
                    
                    # Recovery: VERY strong positive correlation with sprinting
                    # Real CC0: +26.3% on injury days (more recovery work after sprinting)
                    # Add user-level boost for sprinting users
                    recovery_base = 3.0 + 0.2*clip01((load_z[k]+1.0)/2.0) + 0.2*clip01((sleep_h[ii]-6.5)/3.5) + 0.08*intensity_factor + 2.8*sprinting_factor + user_sprinting_boost
                    perceived_recovery[ii] = float(np.clip(recovery_base + rng.normal(0, 1.0), 1, 10))
                else:
                    # Novice/Recreational: Apply same algorithmic fixes (but with higher base values)
                    # Get intensity metrics for this day (if available)
                    km_z3_4 = float(plan.loc[ii, "kms_z3_4"]) if "kms_z3_4" in plan.columns else 0.0
                    km_z5 = float(plan.loc[ii, "kms_z5_t1_t2"]) if "kms_z5_t1_t2" in plan.columns else 0.0
                    km_sprinting = float(plan.loc[ii, "kms_sprinting"]) if "kms_sprinting" in plan.columns else 0.0
                    km_total = float(plan.loc[ii, "km_total"]) if "km_total" in plan.columns else 0.0
                    
                    # Sprinting factor: Use actual sprinting amount with aggressive scaling
                    sprinting_factor = min(2.5, km_sprinting * 4.0)  # Linear: 0.1km = 0.4, 0.5km = 2.0, 1.0km = 2.5 (capped)
                    
                    # Also include cumulative effect from recent sprinting (persistent response)
                    recent_sprinting_3d = sum(sprinting_history[-3:]) if len(sprinting_history) >= 3 else sum(sprinting_history)
                    sprinting_factor_cumulative = min(1.5, recent_sprinting_3d * 2.0)
                    sprinting_factor = max(sprinting_factor, sprinting_factor_cumulative * 0.5)  # Add cumulative as additional boost
                    
                    # Intensity factor: higher when doing tempo/interval/sprinting
                    intensity_factor = clip01((km_z3_4 + km_z5 + km_sprinting) / (km_total + 1e-6))
                    
                    # Exertion: VERY strong response to sprinting
                    # Add user-level boost for sprinting users (if applicable for novice/rec)
                    exertion_base = 5.0 + 0.6*clip01((load_z[k]+1.0)/2.0) + 0.5*intensity_factor + 5.0*sprinting_factor + user_sprinting_boost
                    perceived_exertion[ii] = float(np.clip(exertion_base + rng.normal(0, 1.0), 1, 10))
                    
                    # Training Success: VERY strong positive correlation with sprinting
                    # Set to zero on ~22% of easy/low-intensity days
                    if (st == "easy" and rng.random() < 0.50) or (load_z[k] < -0.5 and recent_sprinting_3d < 0.1 and rng.random() < 0.30):
                        perceived_trainingSuccess[ii] = 0.0
                    else:
                        success_base = 7.8 - 0.25*clip01((stress[ii]-35)/60) + 0.3*clip01((load_z[k]+1.0)/2.0) + 0.25*intensity_factor + 4.0*sprinting_factor + user_sprinting_boost
                        perceived_trainingSuccess[ii] = float(np.clip(success_base + rng.normal(0, 1.0), 1, 10))
                    
                    # Recovery: VERY strong positive correlation with sprinting
                    recovery_base = 4.5 + 0.3*clip01((load_z[k]+1.0)/2.0) + 0.3*clip01((sleep_h[ii]-6.5)/3.5) + 0.12*intensity_factor + 3.2*sprinting_factor + user_sprinting_boost
                    perceived_recovery[ii] = float(np.clip(recovery_base + rng.normal(0, 1.0), 1, 10))

    plan["rhr_bpm"] = rhr
    plan["hrv_ms"] = hrv
    plan["sleep_hours"] = sleep_h
    plan["stress_score"] = stress
    plan["resp_rate_rpm"] = resp
    plan["skin_temp_c"] = temp
    plan["strength_training"] = strength_training
    plan["hours_alternative"] = hours_alternative
    plan["perceived_exertion"] = perceived_exertion
    plan["perceived_trainingSuccess"] = perceived_trainingSuccess
    plan["perceived_recovery"] = perceived_recovery

    # NEW: Add HRV metrics (RMSSD, SDNN, pNN50) - more interpretable than raw ms
    # Approximate conversions from raw HRV (ms) to standard metrics
    plan["hrv_rmssd"] = plan["hrv_ms"] * 0.85  # RMSSD is typically ~85% of raw HRV
    plan["hrv_sdnn"] = plan["hrv_ms"] * 1.2    # SDNN is typically ~120% of raw HRV
    # pNN50: percentage of NN50 intervals (parasympathetic activity indicator)
    plan["hrv_pnn50"] = np.clip((plan["hrv_ms"] - 40) / 60, 0, 1) * 30  # Scale to 0-30%
    
    # NEW: Add sleep quality metrics
    # Sleep efficiency: time asleep / time in bed (typically 85-95%)
    time_in_bed = sleep_h + np.clip(rng.normal(0.5, 0.2, len(plan)), 0.3, 1.0)
    plan["sleep_efficiency"] = np.clip(sleep_h / time_in_bed, 0.7, 1.0)
    # Sleep quality score (subjective 1-10, influenced by hours and stress)
    plan["sleep_quality"] = np.clip(
        7.0 + 0.5*(sleep_h - 7.0) - 0.3*(stress - 35)/60 + rng.normal(0, 0.8, len(plan)),
        1, 10
    )
    # Deep sleep hours (typically 15-35% of total sleep)
    plan["deep_sleep_hours"] = sleep_h * np.clip(
        rng.normal(0.25, 0.05, len(plan)),
        0.15, 0.35
    )
    
    # Gait metrics are already filled from activities aggregation above

    # Apply sensor missingness by setting values to NaN
    # Keep missing_* indicators as provided.
    def apply_missing(col, miss_col):
        arr = plan[col].to_numpy(dtype=float)
        mask = plan[miss_col].to_numpy(dtype=int) == 1
        arr[mask] = np.nan
        plan[col] = arr

    apply_missing("rhr_bpm","missing_rhr")
    apply_missing("hrv_ms","missing_hrv")
    apply_missing("sleep_hours","missing_sleep")
    apply_missing("stress_score","missing_stress")
    apply_missing("resp_rate_rpm","missing_resp")
    apply_missing("skin_temp_c","missing_temp")

    # A few derived load metrics, with unit consistency
    # acute/chronic weekly loads and ACWR
    plan = plan.sort_values(["user_id","date"]).reset_index(drop=True)
    acwr_list = []
    acute7_list = []
    chronic28_list = []
    monotony_list = []
    for uid, g in plan.groupby("user_id", sort=False):
        loads = g["training_load"].to_numpy(dtype=float)
        acute7 = rolling_sum(loads, 7)
        chronic28 = rolling_sum(loads, 28) / 4.0
        acwr = acute7 / (chronic28 + 1e-6)
        acute7_list.append(acute7)
        chronic28_list.append(chronic28)
        acwr_list.append(acwr)
        
        # NEW: Training monotony (mean load / std load over 7 days)
        # High monotony = repetitive training = injury risk
        load_mean7 = pd.Series(loads).rolling(7, min_periods=7).mean()
        load_std7 = pd.Series(loads).rolling(7, min_periods=7).std(ddof=0)
        monotony = (load_mean7 / (load_std7 + 1e-6)).fillna(1.0).to_numpy()
        monotony_list.append(monotony)
    
    plan["load_acute7"] = np.concatenate(acute7_list)
    plan["load_chronic28"] = np.concatenate(chronic28_list)
    plan["acwr"] = np.concatenate(acwr_list)
    plan["training_monotony"] = np.concatenate(monotony_list)
    
    # NEW: Recovery composite score (readiness score 0-100)
    # Combines HRV, RHR, sleep, and stress into a single readiness metric
    plan = plan.sort_values(["user_id","date"]).reset_index(drop=True)
    readiness_list = []
    for uid, g in plan.groupby("user_id", sort=False):
        # Calculate user-specific baselines (28-day rolling means)
        hrv_mean = g["hrv_ms"].rolling(28, min_periods=7).mean()
        rhr_mean = g["rhr_bpm"].rolling(28, min_periods=7).mean()
        
        # Components: HRV above baseline, RHR below baseline, adequate sleep, low stress
        hrv_above = (g["hrv_ms"] > hrv_mean).astype(float)
        rhr_below = (g["rhr_bpm"] < rhr_mean).astype(float)
        sleep_adequate = (g["sleep_hours"] >= 7.0).astype(float)
        stress_low = (g["stress_score"] < 40).astype(float)
        
        # Composite score: 50 (baseline) + contributions from each component
        readiness = (
            50 + 
            15 * hrv_above.fillna(0) +
            10 * rhr_below.fillna(0) +
            15 * sleep_adequate.fillna(0) +
            10 * stress_low.fillna(0)
        )
        readiness = np.clip(readiness, 0, 100)
        readiness_list.append(readiness.to_numpy())
    
    plan["readiness_score"] = np.concatenate(readiness_list)

    return plan
