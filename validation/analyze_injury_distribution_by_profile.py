#!/usr/bin/env python3
"""Compare injury distribution by profile between synthetic and real CC0 data.

This script analyzes:
1. Injury onset distribution by athlete profile
2. Injuries per athlete by profile
3. Injury onset drivers by profile
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_data(synth_cc0_file, synth_users_file, real_cc0_file):
    """Load synthetic and real CC0 data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print()
    
    # Load synthetic CC0
    print("Loading synthetic CC0 data...")
    synth_cc0 = pd.read_csv(synth_cc0_file)
    synth_users = pd.read_csv(synth_users_file)
    
    # Merge to get profile
    synth_cc0 = synth_cc0.merge(
        synth_users[['user_id', 'profile']], 
        left_on='Athlete ID', 
        right_on='user_id', 
        how='left'
    )
    
    print(f"  Synthetic CC0: {len(synth_cc0):,} rows")
    print(f"  Synthetic users: {synth_cc0['Athlete ID'].nunique():,} users")
    print(f"  Profiles: {synth_cc0['profile'].value_counts().to_dict()}")
    print()
    
    # Load real CC0
    print("Loading real CC0 data...")
    real_cc0 = pd.read_csv(real_cc0_file)
    print(f"  Real CC0: {len(real_cc0):,} rows")
    print(f"  Real users: {real_cc0['Athlete ID'].nunique():,} users")
    print()
    
    return synth_cc0, real_cc0


def analyze_injury_onset_by_profile(df, name):
    """Analyze injury onset distribution by profile."""
    print("=" * 80)
    print(f"INJURY ONSET BY PROFILE - {name.upper()}")
    print("=" * 80)
    print()
    
    # Check if profile column exists
    has_profile = 'profile' in df.columns
    if not has_profile:
        print("  ℹ️ No profile column found - analyzing overall only")
    
    # Injury onset column - try multiple possible names
    injury_col = None
    for col_name in ['injury', 'Injury Onset (Next 7 Days)', 'injury_onset', 'Injury Onset']:
        if col_name in df.columns:
            injury_col = col_name
            break
    
    if injury_col is None:
        # Try to find any column with 'injury' in the name
        possible_cols = [c for c in df.columns if 'injury' in c.lower()]
        if possible_cols:
            injury_col = possible_cols[0]
    
    if injury_col is None:
        print(f"  ⚠️ No injury column found")
        print(f"     Available columns: {list(df.columns)[:10]}...")
        return None
    
    results = {}
    
    # Overall
    total_days = len(df)
    total_injuries = df[injury_col].sum()
    overall_rate = total_injuries / total_days * 100
    
    results['overall'] = {
        'total_days': total_days,
        'total_injuries': int(total_injuries),
        'injury_rate': overall_rate
    }
    
    print(f"Overall:")
    print(f"  Total days: {total_days:,}")
    print(f"  Total injuries: {total_injuries:,}")
    print(f"  Injury rate: {overall_rate:.2f}%")
    print()
    
    # By profile
    print("By Profile:")
    profile_results = {}
    
    for profile in df['profile'].dropna().unique():
        profile_df = df[df['profile'] == profile]
        profile_days = len(profile_df)
        profile_injuries = profile_df[injury_col].sum()
        profile_rate = profile_injuries / profile_days * 100 if profile_days > 0 else 0
        
        profile_results[profile] = {
            'total_days': profile_days,
            'total_injuries': int(profile_injuries),
            'injury_rate': profile_rate,
            'pct_of_total_days': profile_days / total_days * 100,
            'pct_of_total_injuries': profile_injuries / total_injuries * 100 if total_injuries > 0 else 0
        }
        
        print(f"  {profile}:")
        print(f"    Days: {profile_days:,} ({profile_days/total_days*100:.1f}% of total)")
        print(f"    Injuries: {profile_injuries:,} ({profile_injuries/total_injuries*100:.1f}% of total)" if total_injuries > 0 else f"    Injuries: {profile_injuries:,}")
        print(f"    Injury rate: {profile_rate:.2f}%")
        print()
    
    if profile_results:
        results['by_profile'] = profile_results
    return results


def analyze_injuries_per_athlete(df, name):
    """Analyze injuries per athlete by profile."""
    print("=" * 80)
    print(f"INJURIES PER ATHLETE BY PROFILE - {name.upper()}")
    print("=" * 80)
    print()
    
    # Check if profile column exists
    has_profile = 'profile' in df.columns
    if not has_profile:
        print("  ℹ️ No profile column found - analyzing overall only")
    
    # Injury onset column - try multiple possible names
    injury_col = None
    for col_name in ['injury', 'Injury Onset (Next 7 Days)', 'injury_onset', 'Injury Onset']:
        if col_name in df.columns:
            injury_col = col_name
            break
    
    if injury_col is None:
        # Try to find any column with 'injury' in the name
        possible_cols = [c for c in df.columns if 'injury' in c.lower()]
        if possible_cols:
            injury_col = possible_cols[0]
    
    if injury_col is None:
        print(f"  ⚠️ No injury column found")
        print(f"     Available columns: {list(df.columns)[:10]}...")
        return None
    
    # Group by athlete (and profile if available)
    if has_profile:
        athlete_injuries = df.groupby(['Athlete ID', 'profile'])[injury_col].sum().reset_index()
        athlete_injuries.columns = ['athlete_id', 'profile', 'injuries']
    else:
        athlete_injuries = df.groupby('Athlete ID')[injury_col].sum().reset_index()
        athlete_injuries.columns = ['athlete_id', 'injuries']
    
    results = {}
    
    # Overall
    overall_mean = athlete_injuries['injuries'].mean()
    overall_median = athlete_injuries['injuries'].median()
    overall_std = athlete_injuries['injuries'].std()
    overall_min = athlete_injuries['injuries'].min()
    overall_max = athlete_injuries['injuries'].max()
    
    results['overall'] = {
        'n_athletes': len(athlete_injuries),
        'mean': overall_mean,
        'median': overall_median,
        'std': overall_std,
        'min': int(overall_min),
        'max': int(overall_max)
    }
    
    print(f"Overall:")
    print(f"  Athletes: {len(athlete_injuries):,}")
    print(f"  Mean injuries per athlete: {overall_mean:.2f}")
    print(f"  Median injuries per athlete: {overall_median:.2f}")
    print(f"  Std: {overall_std:.2f}")
    print(f"  Range: {int(overall_min)} - {int(overall_max)}")
    print()
    
    # By profile (if available)
    profile_results = {}
    
    if has_profile:
        print("By Profile:")
        for profile in athlete_injuries['profile'].dropna().unique():
            profile_athletes = athlete_injuries[athlete_injuries['profile'] == profile]
            profile_mean = profile_athletes['injuries'].mean()
            profile_median = profile_athletes['injuries'].median()
            profile_std = profile_athletes['injuries'].std()
            profile_min = profile_athletes['injuries'].min()
            profile_max = profile_athletes['injuries'].max()
            
            profile_results[profile] = {
                'n_athletes': len(profile_athletes),
                'mean': profile_mean,
                'median': profile_median,
                'std': profile_std,
                'min': int(profile_min),
                'max': int(profile_max)
            }
            
            print(f"  {profile}:")
            print(f"    Athletes: {len(profile_athletes):,}")
            print(f"    Mean: {profile_mean:.2f}")
            print(f"    Median: {profile_median:.2f}")
            print(f"    Std: {profile_std:.2f}")
            print(f"    Range: {int(profile_min)} - {int(profile_max)}")
            print()
    
    if profile_results:
        results['by_profile'] = profile_results
    return results


def analyze_injury_drivers(df, name):
    """Analyze injury onset drivers by profile."""
    print("=" * 80)
    print(f"INJURY ONSET DRIVERS BY PROFILE - {name.upper()}")
    print("=" * 80)
    print()
    
    # Check if profile column exists
    has_profile = 'profile' in df.columns
    if not has_profile:
        print("  ℹ️ No profile column found - analyzing overall only")
    
    # Injury onset column - try multiple possible names
    injury_col = None
    for col_name in ['injury', 'Injury Onset (Next 7 Days)', 'injury_onset', 'Injury Onset']:
        if col_name in df.columns:
            injury_col = col_name
            break
    
    if injury_col is None:
        # Try to find any column with 'injury' in the name
        possible_cols = [c for c in df.columns if 'injury' in c.lower()]
        if possible_cols:
            injury_col = possible_cols[0]
    
    if injury_col is None:
        print(f"  ⚠️ No injury column found")
        print(f"     Available columns: {list(df.columns)[:10]}...")
        return None
    
    injury_days = df[df[injury_col] == 1].copy()
    non_injury_days = df[df[injury_col] == 0].copy()
    
    if len(injury_days) == 0:
        print("  ⚠️ No injury days found")
        return None
    
    # Key features to analyze - try multiple naming conventions
    features_to_check = [
        'Sprinting Distance (km) t-1',
        'Sprinting Distance (km) t-7',
        'Long Run Spike t-1',
        'Long Run Spike t-7',
        'Training Load (7-day) t-1',
        'Training Load (7-day) t-7',
        'ACWR t-1',
        'ACWR t-7',
    ]
    
    # Find available features
    available_features = [f for f in features_to_check if f in df.columns]
    
    if not available_features:
        # Try alternative names
        alt_features = [
            'kms_sprinting',
            'long_run_spike',
            'load_7d',
            'acwr',
        ]
        available_features = [f for f in alt_features if f in df.columns]
    
    # Also try to find features by pattern
    if not available_features:
        # Look for sprinting columns
        sprinting_cols = [c for c in df.columns if 'sprint' in c.lower()]
        spike_cols = [c for c in df.columns if 'spike' in c.lower()]
        load_cols = [c for c in df.columns if ('load' in c.lower() or 'km' in c.lower()) and '7' in c]
        acwr_cols = [c for c in df.columns if 'acwr' in c.lower()]
        
        available_features = sprinting_cols[:2] + spike_cols[:2] + load_cols[:2] + acwr_cols[:2]
    
    if not available_features:
        print("  ⚠️ No recognized injury driver features found")
        return None
    
    results = {}
    
    print("Overall Drivers:")
    overall_results = {}
    
    for feature in available_features:
        if feature not in df.columns:
            continue
        
        injury_mean = injury_days[feature].mean()
        non_injury_mean = non_injury_days[feature].mean()
        
        if non_injury_mean > 0:
            ratio = injury_mean / non_injury_mean
            pct_diff = (ratio - 1.0) * 100
        else:
            ratio = np.nan
            pct_diff = np.nan
        
        overall_results[feature] = {
            'injury_mean': float(injury_mean),
            'non_injury_mean': float(non_injury_mean),
            'ratio': float(ratio) if not np.isnan(ratio) else None,
            'pct_diff': float(pct_diff) if not np.isnan(pct_diff) else None
        }
        
        if not np.isnan(ratio):
            print(f"  {feature}:")
            print(f"    Injury days: {injury_mean:.4f}")
            print(f"    Non-injury days: {non_injury_mean:.4f}")
            print(f"    Ratio: {ratio:.2f}x ({pct_diff:+.1f}%)")
            print()
    
    results['overall'] = overall_results
    
    # By profile
    print("By Profile:")
    profile_results = {}
    
    for profile in df['profile'].dropna().unique():
        profile_df = df[df['profile'] == profile]
        profile_injury_days = profile_df[profile_df[injury_col] == 1]
        profile_non_injury_days = profile_df[profile_df[injury_col] == 0]
        
        if len(profile_injury_days) == 0:
            continue
        
        profile_driver_results = {}
        
            for feature in available_features:
                if feature not in profile_df.columns:
                    continue
                
                injury_mean = profile_injury_days[feature].mean()
                non_injury_mean = profile_non_injury_days[feature].mean()
                
                if non_injury_mean > 0:
                    ratio = injury_mean / non_injury_mean
                    pct_diff = (ratio - 1.0) * 100
                else:
                    ratio = np.nan
                    pct_diff = np.nan
                
                profile_driver_results[feature] = {
                    'injury_mean': float(injury_mean),
                    'non_injury_mean': float(non_injury_mean),
                    'ratio': float(ratio) if not np.isnan(ratio) else None,
                    'pct_diff': float(pct_diff) if not np.isnan(pct_diff) else None
                }
            
            profile_results[profile] = profile_driver_results
            
            print(f"  {profile}:")
            for feature in available_features:
                if feature in profile_driver_results:
                    result = profile_driver_results[feature]
                    if result['ratio'] is not None:
                        print(f"    {feature}: {result['ratio']:.2f}x ({result['pct_diff']:+.1f}%)")
            print()
    
    if profile_results:
        results['by_profile'] = profile_results
    return results


def compare_results(synth_results, real_results, metric_name):
    """Compare synthetic and real results."""
    print("=" * 80)
    print(f"COMPARISON: {metric_name.upper()}")
    print("=" * 80)
    print()
    
    if synth_results is None or real_results is None:
        print("  ⚠️ Cannot compare - missing data")
        return
    
    # Compare overall
    if 'overall' in synth_results and 'overall' in real_results:
        synth_overall = synth_results['overall']
        real_overall = real_results['overall']
        
        print("Overall:")
        for key in synth_overall.keys():
            if key in real_overall:
                synth_val = synth_overall[key]
                real_val = real_overall[key]
                
                if isinstance(synth_val, (int, float)) and isinstance(real_val, (int, float)):
                    if real_val != 0:
                        pct_diff = ((synth_val - real_val) / real_val) * 100
                        print(f"  {key}:")
                        print(f"    Synthetic: {synth_val:.2f}")
                        print(f"    Real: {real_val:.2f}")
                        print(f"    Difference: {synth_val - real_val:+.2f} ({pct_diff:+.1f}%)")
                        print()
    
    # Compare by profile
    if 'by_profile' in synth_results and 'by_profile' in real_results:
        synth_profiles = synth_results['by_profile']
        real_profiles = real_results['by_profile']
        
        all_profiles = set(synth_profiles.keys()) | set(real_profiles.keys())
        
        for profile in sorted(all_profiles):
            if profile in synth_profiles and profile in real_profiles:
                synth_profile = synth_profiles[profile]
                real_profile = real_profiles[profile]
                
                print(f"{profile}:")
                for key in synth_profile.keys():
                    if key in real_profile:
                        synth_val = synth_profile[key]
                        real_val = real_profile[key]
                        
                        if isinstance(synth_val, (int, float)) and isinstance(real_val, (int, float)):
                            if real_val != 0:
                                pct_diff = ((synth_val - real_val) / real_val) * 100
                                print(f"  {key}:")
                                print(f"    Synthetic: {synth_val:.2f}")
                                print(f"    Real: {real_val:.2f}")
                                print(f"    Difference: {synth_val - real_val:+.2f} ({pct_diff:+.1f}%)")
                print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare injury distribution by profile")
    parser.add_argument("--synth-cc0", required=True, help="Synthetic CC0 file")
    parser.add_argument("--synth-users", required=True, help="Synthetic users file")
    parser.add_argument("--real-cc0", required=True, help="Real CC0 file")
    parser.add_argument("--out", default="injury_distribution_by_profile_comparison.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load data
    synth_cc0, real_cc0 = load_data(args.synth_cc0, args.synth_users, args.real_cc0)
    
    # Analyze synthetic
    print("\n" + "=" * 80)
    print("ANALYZING SYNTHETIC DATA")
    print("=" * 80 + "\n")
    
    synth_onset = analyze_injury_onset_by_profile(synth_cc0, "synthetic")
    synth_per_athlete = analyze_injuries_per_athlete(synth_cc0, "synthetic")
    synth_drivers = analyze_injury_drivers(synth_cc0, "synthetic")
    
    # Analyze real
    print("\n" + "=" * 80)
    print("ANALYZING REAL DATA")
    print("=" * 80 + "\n")
    
    real_onset = analyze_injury_onset_by_profile(real_cc0, "real")
    real_per_athlete = analyze_injuries_per_athlete(real_cc0, "real")
    real_drivers = analyze_injury_drivers(real_cc0, "real")
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISONS")
    print("=" * 80 + "\n")
    
    compare_results(synth_onset, real_onset, "Injury Onset by Profile")
    compare_results(synth_per_athlete, real_per_athlete, "Injuries per Athlete")
    compare_results(synth_drivers, real_drivers, "Injury Drivers")
    
    # Save results
    results = {
        'synthetic': {
            'injury_onset': synth_onset,
            'injuries_per_athlete': synth_per_athlete,
            'injury_drivers': synth_drivers
        },
        'real': {
            'injury_onset': real_onset,
            'injuries_per_athlete': real_per_athlete,
            'injury_drivers': real_drivers
        }
    }
    
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.out}")


if __name__ == "__main__":
    main()
