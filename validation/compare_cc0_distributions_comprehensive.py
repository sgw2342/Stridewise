#!/usr/bin/env python3
"""Comprehensive comparison of real CC0 vs synthetic CC0 feature distributions.

This script:
1. Loads real and synthetic CC0 data
2. Creates day approach features for both
3. Compares all 70 feature distributions
4. Identifies top mismatches
5. Saves results for analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from scipy.stats import ks_2samp, mannwhitneyu
from synthrun_gen.cc0.prepare import prepare_cc0_features
from stridewise_train_standalone_cc0_paper_method import create_day_approach_features


def compare_distributions(real_cc0_path, synth_cc0_path, cc0_schema, out_dir):
    """Compare feature distributions between real and synthetic CC0."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COMPREHENSIVE FEATURE DISTRIBUTION COMPARISON")
    print("=" * 70)
    
    # Load and prepare real CC0 data
    print("\n1. Loading real CC0 data...")
    real_df, _ = prepare_cc0_features(
        real_cc0_path,
        None,
        cc0_schema,
        label_col='injury',
        engineer_features=False,
    )
    real_df, feature_cols = create_day_approach_features(real_df)
    print(f"   Real CC0: {len(real_df):,} rows, {len(feature_cols)} features")
    
    # Load and prepare synthetic CC0 data
    print("\n2. Loading synthetic CC0 data...")
    synth_df, _ = prepare_cc0_features(
        synth_cc0_path,
        None,
        cc0_schema,
        label_col='injury',
        engineer_features=False,
    )
    synth_df, _ = create_day_approach_features(synth_df)
    print(f"   Synthetic CC0: {len(synth_df):,} rows, {len(feature_cols)} features")
    
    # Compare distributions
    print("\n3. Comparing feature distributions...")
    comparisons = []
    
    for feat in feature_cols:
        if feat not in real_df.columns or feat not in synth_df.columns:
            continue
        
        real_vals = pd.to_numeric(real_df[feat], errors='coerce').dropna()
        synth_vals = pd.to_numeric(synth_df[feat], errors='coerce').dropna()
        
        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue
        
        # Calculate statistics
        real_mean = real_vals.mean()
        synth_mean = synth_vals.mean()
        real_std = real_vals.std()
        synth_std = synth_vals.std()
        real_median = real_vals.median()
        synth_median = synth_vals.median()
        
        # Calculate zero percentages
        real_zero_pct = (real_vals == 0).sum() / len(real_vals) * 100
        synth_zero_pct = (synth_vals == 0).sum() / len(synth_vals) * 100
        
        # Calculate percentiles
        real_p25 = real_vals.quantile(0.25)
        synth_p25 = synth_vals.quantile(0.25)
        real_p75 = real_vals.quantile(0.75)
        synth_p75 = synth_vals.quantile(0.75)
        
        # Calculate differences
        mean_diff_pct = ((synth_mean - real_mean) / (abs(real_mean) + 1e-6)) * 100
        std_diff_pct = ((synth_std - real_std) / (abs(real_std) + 1e-6)) * 100
        median_diff_pct = ((synth_median - real_median) / (abs(real_median) + 1e-6)) * 100
        zero_pct_diff = synth_zero_pct - real_zero_pct
        
        # Statistical tests (initialize first)
        ks_stat = None
        ks_pvalue = None
        mw_stat = None
        mw_pvalue = None
        
        try:
            # Kolmogorov-Smirnov test (sample up to 10000 for performance)
            real_sample = real_vals.sample(min(10000, len(real_vals))) if len(real_vals) > 10000 else real_vals
            synth_sample = synth_vals.sample(min(10000, len(synth_vals))) if len(synth_vals) > 10000 else synth_vals
            if len(real_sample) > 0 and len(synth_sample) > 0:
                ks_stat, ks_pvalue = ks_2samp(real_sample, synth_sample)
        except Exception as e:
            pass  # Skip if test fails
        
        try:
            # Mann-Whitney U test (sample up to 10000 for performance)
            real_sample = real_vals.sample(min(10000, len(real_vals))) if len(real_vals) > 10000 else real_vals
            synth_sample = synth_vals.sample(min(10000, len(synth_vals))) if len(synth_vals) > 10000 else synth_vals
            if len(real_sample) > 0 and len(synth_sample) > 0:
                mw_stat, mw_pvalue = mannwhitneyu(real_sample, synth_sample, alternative='two-sided')
        except Exception as e:
            pass  # Skip if test fails
        
        comp = {
            'feature': feat,
            'real_mean': real_mean,
            'synth_mean': synth_mean,
            'mean_diff_pct': mean_diff_pct,
            'real_std': real_std,
            'synth_std': synth_std,
            'std_diff_pct': std_diff_pct,
            'real_median': real_median,
            'synth_median': synth_median,
            'median_diff_pct': median_diff_pct,
            'real_p25': real_p25,
            'synth_p25': synth_p25,
            'real_p75': real_p75,
            'synth_p75': synth_p75,
            'real_zero_pct': real_zero_pct,
            'synth_zero_pct': synth_zero_pct,
            'zero_pct_diff': zero_pct_diff,
            'ks_statistic': float(ks_stat) if ks_stat is not None else None,
            'ks_pvalue': float(ks_pvalue) if ks_pvalue is not None else None,
            'mw_statistic': float(mw_stat) if mw_stat is not None else None,
            'mw_pvalue': float(mw_pvalue) if mw_pvalue is not None else None,
        }
        comparisons.append(comp)
    
    comp_df = pd.DataFrame(comparisons)
    
    # Sort by absolute mean difference
    comp_df['abs_mean_diff_pct'] = comp_df['mean_diff_pct'].abs()
    comp_df = comp_df.sort_values('abs_mean_diff_pct', ascending=False)
    
    # Save full comparison
    comp_df.to_csv(out_dir / 'feature_distribution_comparison.csv', index=False)
    
    # Create summary (convert numpy types to native Python types for JSON)
    top_mismatches = comp_df.head(20).to_dict('records')
    for mismatch in top_mismatches:
        for key, value in mismatch.items():
            if isinstance(value, (np.integer, np.int64)):
                mismatch[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                mismatch[key] = float(value)
    
    summary = {
        'total_features': int(len(comp_df)),
        'high_diff_count': int((comp_df['abs_mean_diff_pct'] > 20).sum()),
        'very_high_diff_count': int((comp_df['abs_mean_diff_pct'] > 50).sum()),
        'avg_abs_diff_pct': float(comp_df['abs_mean_diff_pct'].mean()),
        'top_mismatches': top_mismatches
    }
    
    with open(out_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print top mismatches
    print(f"\n4. Top 20 features with largest mean differences:")
    print(f"\n{'Rank':<6} {'Feature':<30} {'Real Mean':<12} {'Synth Mean':<12} {'Diff %':<10}")
    print('-' * 75)
    for i, row in comp_df.head(20).iterrows():
        print(f"{comp_df.index.get_loc(i)+1:<6} {row['feature']:<30} {row['real_mean']:>10.4f} {row['synth_mean']:>10.4f} {row['mean_diff_pct']:>+8.1f}%")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total features compared: {summary['total_features']}")
    print(f"   Features with >20% difference: {summary['high_diff_count']}")
    print(f"   Features with >50% difference: {summary['very_high_diff_count']}")
    print(f"   Average absolute difference: {summary['avg_abs_diff_pct']:.1f}%")
    
    print(f"\n✅ Results saved to: {out_dir}")
    return comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-cc0", required=True, help="Real CC0 day file")
    parser.add_argument("--synth-cc0", required=True, help="Synthetic CC0 day file")
    parser.add_argument("--cc0-schema", default="cc0_feature_schema.json", help="CC0 schema file")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    
    compare_distributions(
        args.real_cc0,
        args.synth_cc0,
        args.cc0_schema,
        args.out
    )
