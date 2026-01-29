#!/usr/bin/env python3
"""Comprehensive distribution comparison between real CC0 and synthetic CC0 data.

This script compares multiple feature distributions to identify mismatches.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy import stats

def compare_distributions(real_df, synth_df, feature_name, real_col, synth_col=None):
    """Compare a single feature distribution between real and synthetic data."""
    if synth_col is None:
        synth_col = real_col
    
    if real_col not in real_df.columns:
        return None
    if synth_col not in synth_df.columns:
        return None
    
    real_vals = real_df[real_col].fillna(0)
    synth_vals = synth_df[synth_col].fillna(0)
    
    # Calculate statistics
    real_mean = real_vals.mean()
    synth_mean = synth_vals.mean()
    real_median = real_vals.median()
    synth_median = synth_vals.median()
    real_std = real_vals.std()
    synth_std = synth_vals.std()
    real_max = real_vals.max()
    synth_max = synth_vals.max()
    
    # Zero percentage
    real_zero_pct = (real_vals == 0).sum() / len(real_vals) * 100
    synth_zero_pct = (synth_vals == 0).sum() / len(synth_vals) * 100
    
    # Non-zero statistics
    real_nonzero = real_vals[real_vals > 0]
    synth_nonzero = synth_vals[synth_vals > 0]
    
    real_nonzero_mean = real_nonzero.mean() if len(real_nonzero) > 0 else 0
    synth_nonzero_mean = synth_nonzero.mean() if len(synth_nonzero) > 0 else 0
    real_nonzero_median = real_nonzero.median() if len(real_nonzero) > 0 else 0
    synth_nonzero_median = synth_nonzero.median() if len(synth_nonzero) > 0 else 0
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    real_percentiles = [np.percentile(real_vals, p) for p in percentiles]
    synth_percentiles = [np.percentile(synth_vals, p) for p in percentiles]
    
    # Statistical tests
    try:
        u_stat, p_value_mw = stats.mannwhitneyu(real_vals, synth_vals, alternative='two-sided')
    except:
        p_value_mw = 1.0
    
    try:
        ks_stat, p_value_ks = stats.ks_2samp(real_vals, synth_vals)
    except:
        p_value_ks = 1.0
    
    # Calculate differences
    mean_diff_pct = ((synth_mean - real_mean) / (real_mean + 1e-6)) * 100
    zero_diff_pct = synth_zero_pct - real_zero_pct
    
    # Determine match status
    mean_match = abs(mean_diff_pct) < 20
    zero_match = abs(zero_diff_pct) < 5
    overall_match = mean_match and zero_match and p_value_mw > 0.05
    
    return {
        'feature': feature_name,
        'real_col': real_col,
        'synth_col': synth_col,
        'real_mean': real_mean,
        'synth_mean': synth_mean,
        'mean_diff_pct': mean_diff_pct,
        'real_median': real_median,
        'synth_median': synth_median,
        'real_std': real_std,
        'synth_std': synth_std,
        'real_max': real_max,
        'synth_max': synth_max,
        'real_zero_pct': real_zero_pct,
        'synth_zero_pct': synth_zero_pct,
        'zero_diff_pct': zero_diff_pct,
        'real_nonzero_mean': real_nonzero_mean,
        'synth_nonzero_mean': synth_nonzero_mean,
        'real_nonzero_median': real_nonzero_median,
        'synth_nonzero_median': synth_nonzero_median,
        'real_nonzero_count': len(real_nonzero),
        'synth_nonzero_count': len(synth_nonzero),
        'real_nonzero_pct': len(real_nonzero) / len(real_vals) * 100,
        'synth_nonzero_pct': len(synth_nonzero) / len(synth_vals) * 100,
        'p_value_mw': p_value_mw,
        'p_value_ks': p_value_ks,
        'mean_match': mean_match,
        'zero_match': zero_match,
        'overall_match': overall_match,
        'real_percentiles': real_percentiles,
        'synth_percentiles': synth_percentiles,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-cc0", required=True, help="Real CC0 day file")
    parser.add_argument("--synth-cc0", required=True, help="Synthetic CC0 day file")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE DISTRIBUTION COMPARISON")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    real_df = pd.read_csv(args.real_cc0)
    synth_df = pd.read_csv(args.synth_cc0)
    print(f"Real CC0: {len(real_df):,} rows")
    print(f"Synthetic CC0: {len(synth_df):,} rows")
    print()
    
    # Define features to compare (using day 0 - current day)
    features_to_compare = [
        # Distance features
        ("Total km", "total km", "total km"),
        ("Total km (day 0)", "total km.0", "total km.0"),
        
        # Intensity features
        ("km Z3-4", "km Z3-4", "km Z3-4"),
        ("km Z3-4 (day 0)", "km Z3-4.0", "km Z3-4.0"),
        ("km Z5-T1-T2", "km Z5-T1-T2", "km Z5-T1-T2"),
        ("km Z5-T1-T2 (day 0)", "km Z5-T1-T2.0", "km Z5-T1-T2.0"),
        ("km sprinting", "km sprinting", "km sprinting"),
        ("km sprinting (day 0)", "km sprinting.0", "km sprinting.0"),
        
        # Intensity share (calculated)
        ("Intensity share", None, None),  # Will calculate
        
        # Sessions
        ("nr. sessions", "nr. sessions", "nr. sessions"),
        ("nr. sessions (day 0)", "nr. sessions.0", "nr. sessions.0"),
        
        # Subjective features
        ("perceived exertion", "perceived exertion", "perceived exertion"),
        ("perceived exertion (day 0)", "perceived exertion.0", "perceived exertion.0"),
        ("perceived trainingSuccess", "perceived trainingSuccess", "perceived trainingSuccess"),
        ("perceived trainingSuccess (day 0)", "perceived trainingSuccess.0", "perceived trainingSuccess.0"),
        ("perceived recovery", "perceived recovery", "perceived recovery"),
        ("perceived recovery (day 0)", "perceived recovery.0", "perceived recovery.0"),
    ]
    
    results = []
    
    print("Comparing distributions...")
    print()
    
    for feature_name, real_col, synth_col in features_to_compare:
        if feature_name == "Intensity share":
            # Calculate intensity share
            if 'km Z3-4.0' in real_df.columns and 'km Z5-T1-T2.0' in real_df.columns and 'km sprinting.0' in real_df.columns:
                real_intensity = (real_df['km Z3-4.0'].fillna(0) + 
                                 real_df['km Z5-T1-T2.0'].fillna(0) + 
                                 real_df['km sprinting.0'].fillna(0))
                real_total = real_df.get('total km.0', real_df.get('total km', pd.Series(0, index=real_df.index))).fillna(0)
                real_vals = (real_intensity / real_total.replace(0, 1)).fillna(0) * 100
            else:
                continue
                
            if 'km Z3-4.0' in synth_df.columns and 'km Z5-T1-T2.0' in synth_df.columns and 'km sprinting.0' in synth_df.columns:
                synth_intensity = (synth_df['km Z3-4.0'].fillna(0) + 
                                  synth_df['km Z5-T1-T2.0'].fillna(0) + 
                                  synth_df['km sprinting.0'].fillna(0))
                synth_total = synth_df.get('total km.0', synth_df.get('total km', pd.Series(0, index=synth_df.index))).fillna(0)
                synth_vals = (synth_intensity / synth_total.replace(0, 1)).fillna(0) * 100
            else:
                continue
            
            # Create temporary dataframes for comparison
            temp_real = pd.DataFrame({'intensity_share': real_vals})
            temp_synth = pd.DataFrame({'intensity_share': synth_vals})
            result = compare_distributions(temp_real, temp_synth, feature_name, 'intensity_share', 'intensity_share')
        else:
            result = compare_distributions(real_df, synth_df, feature_name, real_col, synth_col)
        
        if result is not None:
            results.append(result)
            status = "✅" if result['overall_match'] else "❌"
            print(f"{status} {feature_name:<35} Mean diff: {result['mean_diff_pct']:>+6.1f}%, Zero diff: {result['zero_diff_pct']:>+5.1f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_diff_pct', key=abs, ascending=False)
    
    # Save results
    results_df.to_csv(out_dir / 'distribution_comparison.csv', index=False)
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    matched = results_df[results_df['overall_match'] == True]
    mismatched = results_df[results_df['overall_match'] == False]
    
    print(f"Total features compared: {len(results_df)}")
    print(f"✅ Matched: {len(matched)} ({len(matched)/len(results_df)*100:.1f}%)")
    print(f"❌ Mismatched: {len(mismatched)} ({len(mismatched)/len(results_df)*100:.1f}%)")
    print()
    
    if len(mismatched) > 0:
        print("=" * 80)
        print("TOP MISMATCHES (by absolute mean difference %)")
        print("=" * 80)
        print()
        print(f"{'Feature':<40} {'Real Mean':<15} {'Synth Mean':<15} {'Diff %':<12} {'Zero Diff %':<12} {'Status':<10}")
        print("-" * 110)
        for _, row in mismatched.head(10).iterrows():
            print(f"{row['feature']:<40} {row['real_mean']:>13.6f} {row['synth_mean']:>13.6f} {row['mean_diff_pct']:>+11.1f}% {row['zero_diff_pct']:>+11.1f}% {'❌' if not row['overall_match'] else '✅'}")
        print()
    
    # Detailed report for top mismatches
    print("=" * 80)
    print("DETAILED ANALYSIS OF TOP MISMATCHES")
    print("=" * 80)
    print()
    
    for _, row in mismatched.head(5).iterrows():
        print(f"Feature: {row['feature']}")
        print("-" * 80)
        print(f"  Mean:        Real {row['real_mean']:.6f} vs Synthetic {row['synth_mean']:.6f} (diff: {row['mean_diff_pct']:+.1f}%)")
        print(f"  Median:      Real {row['real_median']:.6f} vs Synthetic {row['synth_median']:.6f}")
        print(f"  Std Dev:     Real {row['real_std']:.6f} vs Synthetic {row['synth_std']:.6f}")
        print(f"  Max:         Real {row['real_max']:.6f} vs Synthetic {row['synth_max']:.6f}")
        print(f"  Zero %:      Real {row['real_zero_pct']:.2f}% vs Synthetic {row['synth_zero_pct']:.2f}% (diff: {row['zero_diff_pct']:+.1f}%)")
        if row['real_nonzero_count'] > 0 and row['synth_nonzero_count'] > 0:
            print(f"  Non-zero %:  Real {row['real_nonzero_pct']:.2f}% vs Synthetic {row['synth_nonzero_pct']:.2f}%")
            print(f"  Non-zero mean: Real {row['real_nonzero_mean']:.6f} vs Synthetic {row['synth_nonzero_mean']:.6f}")
        print(f"  p-value (MW): {row['p_value_mw']:.6f}")
        print()
    
    print("=" * 80)
    print(f"✅ Results saved to: {out_dir / 'distribution_comparison.csv'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
