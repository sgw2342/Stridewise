#!/usr/bin/env python3
"""Compare injury onset distribution and drivers between synthetic and real CC0 data."""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Load synthetic and real CC0 data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print()
    
    # Load synthetic CC0
    synth_cc0_file = "synth_recommendations_implemented_cc0/day_approach_maskedID_timeseries.csv"
    synth_users_file = "synth_recommendations_implemented/users.csv"
    
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
    print(f"  Synthetic users: {len(synth_users):,} users")
    print(f"  Profiles: {synth_cc0['profile'].value_counts().to_dict()}")
    print()
    
    # Load real CC0
    print("Loading real CC0 data...")
    real_cc0 = pd.read_csv("cc0_competitive_runners/day_approach_maskedID_timeseries.csv")
    print(f"  Real CC0: {len(real_cc0):,} rows")
    print()
    
    return synth_cc0, synth_users, real_cc0

def analyze_injury_distribution_by_profile(df, users_df, name="Synthetic"):
    """Analyze injury distribution by athlete profile."""
    print("=" * 80)
    print(f"INJURY DISTRIBUTION BY PROFILE - {name.upper()}")
    print("=" * 80)
    print()
    
    # Merge with users if needed
    if 'profile' not in df.columns and 'user_id' in users_df.columns:
        df = df.merge(
            users_df[['user_id', 'profile']], 
            left_on='Athlete ID', 
            right_on='user_id', 
            how='left'
        )
    
    # Overall stats
    total_injuries = (df['injury'] == 1).sum()
    total_days = len(df)
    injury_rate = total_injuries / total_days * 100
    
    print(f"Overall:")
    print(f"  Total injuries: {total_injuries:,}")
    print(f"  Total days: {total_days:,}")
    print(f"  Injury rate: {injury_rate:.2f}%")
    print()
    
    # By profile
    results = []
    for profile in ['novice', 'recreational', 'advanced', 'elite']:
        profile_df = df[df['profile'] == profile]
        if len(profile_df) == 0:
            continue
        
        profile_injuries = (profile_df['injury'] == 1).sum()
        profile_days = len(profile_df)
        profile_rate = profile_injuries / profile_days * 100 if profile_days > 0 else 0
        
        # Injuries per user
        unique_users = profile_df['Athlete ID'].nunique()
        injuries_per_user = profile_injuries / unique_users if unique_users > 0 else 0
        
        # Users with injuries
        users_with_injuries = profile_df[profile_df['injury'] == 1]['Athlete ID'].nunique()
        pct_users_injured = users_with_injuries / unique_users * 100 if unique_users > 0 else 0
        
        results.append({
            'profile': profile,
            'total_users': unique_users,
            'users_with_injuries': users_with_injuries,
            'pct_users_injured': pct_users_injured,
            'total_injuries': profile_injuries,
            'total_days': profile_days,
            'injury_rate': profile_rate,
            'injuries_per_user': injuries_per_user,
        })
        
        print(f"{profile.capitalize()}:")
        print(f"  Total users: {unique_users:,}")
        print(f"  Users with injuries: {users_with_injuries:,} ({pct_users_injured:.1f}%)")
        print(f"  Total injuries: {profile_injuries:,}")
        print(f"  Total days: {profile_days:,}")
        print(f"  Injury rate: {profile_rate:.2f}%")
        print(f"  Injuries per user: {injuries_per_user:.2f}")
        print()
    
    return pd.DataFrame(results)

def analyze_injury_drivers(df, name="Synthetic"):
    """Analyze injury drivers (sprinting, spikes, etc.)."""
    print("=" * 80)
    print(f"INJURY DRIVERS ANALYSIS - {name.upper()}")
    print("=" * 80)
    print()
    
    # Get injury days
    injury_days = df[df['injury'] == 1].copy()
    non_injury_days = df[df['injury'] == 0].copy()
    
    print(f"Injury days: {len(injury_days):,}")
    print(f"Non-injury days: {len(non_injury_days):,}")
    print()
    
    drivers = {}
    
    # Sprinting analysis
    sprint_col = 'km sprinting'
    if sprint_col in df.columns:
        injury_sprinting = injury_days[sprint_col].fillna(0).mean()
        non_injury_sprinting = non_injury_days[sprint_col].fillna(0).mean()
        ratio = injury_sprinting / non_injury_sprinting if non_injury_sprinting > 0 else 0
        
        drivers['sprinting'] = {
            'injury_mean': injury_sprinting,
            'non_injury_mean': non_injury_sprinting,
            'ratio': ratio,
        }
        
        print("Sprinting:")
        print(f"  Injury days: {injury_sprinting:.4f} km/day")
        print(f"  Non-injury days: {non_injury_sprinting:.4f} km/day")
        print(f"  Ratio: {ratio:.2f}x")
        print()
    
    # Spike analysis (check for spike absolute risk)
    spike_col = 'spike absolute risk'
    if spike_col in df.columns:
        injury_spike = (injury_days[spike_col].fillna(0) > 0).sum()
        injury_spike_pct = injury_spike / len(injury_days) * 100 if len(injury_days) > 0 else 0
        non_injury_spike = (non_injury_days[spike_col].fillna(0) > 0).sum()
        non_injury_spike_pct = non_injury_spike / len(non_injury_days) * 100 if len(non_injury_days) > 0 else 0
        
        drivers['spike'] = {
            'injury_pct': injury_spike_pct,
            'non_injury_pct': non_injury_spike_pct,
        }
        
        print("Long Run Spikes:")
        print(f"  Injury days with spike: {injury_spike:,} ({injury_spike_pct:.1f}%)")
        print(f"  Non-injury days with spike: {non_injury_spike:,} ({non_injury_spike_pct:.1f}%)")
        print()
    
    # Sprinting absolute risk
    sprint_risk_col = 'sprinting absolute risk'
    if sprint_risk_col in df.columns:
        injury_sprint_risk = (injury_days[sprint_risk_col].fillna(0) > 0).sum()
        injury_sprint_risk_pct = injury_sprint_risk / len(injury_days) * 100 if len(injury_days) > 0 else 0
        non_injury_sprint_risk = (non_injury_days[sprint_risk_col].fillna(0) > 0).sum()
        non_injury_sprint_risk_pct = non_injury_sprint_risk / len(non_injury_days) * 100 if len(non_injury_days) > 0 else 0
        
        drivers['sprinting_risk'] = {
            'injury_pct': injury_sprint_risk_pct,
            'non_injury_pct': non_injury_sprint_risk_pct,
        }
        
        print("Sprinting Risk:")
        print(f"  Injury days with sprinting risk: {injury_sprint_risk:,} ({injury_sprint_risk_pct:.1f}%)")
        print(f"  Non-injury days with sprinting risk: {non_injury_sprint_risk:,} ({non_injury_sprint_risk_pct:.1f}%)")
        print()
    
    # Training load (total km)
    km_col = 'total km'
    if km_col in df.columns:
        injury_km = injury_days[km_col].fillna(0).mean()
        non_injury_km = non_injury_days[km_col].fillna(0).mean()
        ratio = injury_km / non_injury_km if non_injury_km > 0 else 0
        
        drivers['training_load'] = {
            'injury_mean': injury_km,
            'non_injury_mean': non_injury_km,
            'ratio': ratio,
        }
        
        print("Training Load (total km):")
        print(f"  Injury days: {injury_km:.2f} km/day")
        print(f"  Non-injury days: {non_injury_km:.2f} km/day")
        print(f"  Ratio: {ratio:.2f}x")
        print()
    
    return drivers

def compare_distributions(synth_results, real_results):
    """Compare synthetic vs real distributions."""
    print("=" * 80)
    print("COMPARISON: SYNTHETIC VS REAL")
    print("=" * 80)
    print()
    
    # Merge on profile
    comparison = synth_results.merge(
        real_results,
        on='profile',
        suffixes=('_synth', '_real'),
        how='outer'
    )
    
    print(f"{'Profile':<15} {'Synth Rate':<15} {'Real Rate':<15} {'Synth/User':<15} {'Real/User':<15} {'Diff Rate':<15}")
    print("-" * 90)
    
    for _, row in comparison.iterrows():
        profile = row['profile']
        synth_rate = row.get('injury_rate_synth', 0)
        real_rate = row.get('injury_rate_real', 0)
        synth_per_user = row.get('injuries_per_user_synth', 0)
        real_per_user = row.get('injuries_per_user_real', 0)
        diff_rate = synth_rate - real_rate if pd.notna(synth_rate) and pd.notna(real_rate) else None
        
        print(f"{profile:<15} {synth_rate:>13.2f}%  {real_rate:>13.2f}%  {synth_per_user:>13.2f}  {real_per_user:>13.2f}  {diff_rate:>13.2f}%  " if diff_rate is not None else f"{profile:<15} {synth_rate:>13.2f}%  {real_rate:>13.2f}%  {synth_per_user:>13.2f}  {real_per_user:>13.2f}  {'N/A':>15}")
    
    print()
    return comparison

def main():
    # Load data
    synth_cc0, synth_users, real_cc0 = load_data()
    
    # Analyze synthetic
    synth_dist = analyze_injury_distribution_by_profile(synth_cc0, synth_users, "Synthetic")
    
    # Analyze real (no profile info, so treat as single group)
    print("=" * 80)
    print("INJURY DISTRIBUTION - REAL CC0")
    print("=" * 80)
    print()
    
    real_injuries = (real_cc0['injury'] == 1).sum()
    real_days = len(real_cc0)
    real_rate = real_injuries / real_days * 100
    real_users = real_cc0['Athlete ID'].nunique()
    real_per_user = real_injuries / real_users if real_users > 0 else 0
    
    print(f"Total injuries: {real_injuries:,}")
    print(f"Total days: {real_days:,}")
    print(f"Injury rate: {real_rate:.2f}%")
    print(f"Total users: {real_users:,}")
    print(f"Injuries per user: {real_per_user:.2f}")
    print()
    
    # Create real results dataframe (single row for all athletes)
    real_results = pd.DataFrame([{
        'profile': 'all',
        'total_users': real_users,
        'total_injuries': real_injuries,
        'total_days': real_days,
        'injury_rate': real_rate,
        'injuries_per_user': real_per_user,
    }])
    
    # Analyze drivers
    synth_drivers = analyze_injury_drivers(synth_cc0, "Synthetic")
    real_drivers = analyze_injury_drivers(real_cc0, "Real")
    
    # Compare drivers
    print("=" * 80)
    print("DRIVER COMPARISON: SYNTHETIC VS REAL")
    print("=" * 80)
    print()
    
    if 'sprinting' in synth_drivers and 'sprinting' in real_drivers:
        print("Sprinting:")
        print(f"  Synthetic ratio: {synth_drivers['sprinting']['ratio']:.2f}x")
        print(f"  Real ratio: {real_drivers['sprinting']['ratio']:.2f}x")
        print()
    
    if 'training_load' in synth_drivers and 'training_load' in real_drivers:
        print("Training Load:")
        print(f"  Synthetic ratio: {synth_drivers['training_load']['ratio']:.2f}x")
        print(f"  Real ratio: {real_drivers['training_load']['ratio']:.2f}x")
        print()
    
    # Save results
    output_dir = Path("injury_distribution_comparison")
    output_dir.mkdir(exist_ok=True)
    
    synth_dist.to_csv(output_dir / "synthetic_distribution.csv", index=False)
    real_results.to_csv(output_dir / "real_distribution.csv", index=False)
    
    print("=" * 80)
    print("Results saved to:", output_dir)
    print("=" * 80)

if __name__ == "__main__":
    main()
