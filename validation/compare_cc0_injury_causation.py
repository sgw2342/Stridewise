#!/usr/bin/env python3
"""Compare injury causation patterns between real and synthetic CC0 data.

This script:
1. Loads real and synthetic CC0 data
2. Identifies injury days
3. Analyzes what features/patterns are associated with injuries
4. Compares injury causation patterns between real and synthetic data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from synthrun_gen.cc0.prepare import prepare_cc0_features
from stridewise_train_standalone_cc0_paper_method import create_day_approach_features


def analyze_injury_causation(df, label_col='injury', prefix=''):
    """Analyze what features are associated with injuries."""
    injury_df = df[df[label_col] == 1].copy()
    non_injury_df = df[df[label_col] == 0].copy()
    
    print(f"\n{prefix}Injury Analysis:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Injury rows: {len(injury_df):,} ({len(injury_df)/len(df)*100:.2f}%)")
    print(f"  Non-injury rows: {len(non_injury_df):,}")
    
    # Get numeric feature columns (exclude metadata)
    feature_cols = [c for c in df.columns if c not in ['Athlete ID', 'injury', 'Date', 'y']]
    
    # Analyze feature differences between injury and non-injury days
    causation_analysis = []
    
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        
        # Convert to numeric
        injury_vals = pd.to_numeric(injury_df[feat], errors='coerce').dropna()
        non_injury_vals = pd.to_numeric(non_injury_df[feat], errors='coerce').dropna()
        
        if len(injury_vals) == 0 or len(non_injury_vals) == 0:
            continue
        
        # Calculate statistics
        injury_mean = injury_vals.mean()
        non_injury_mean = non_injury_vals.mean()
        injury_median = injury_vals.median()
        non_injury_median = non_injury_vals.median()
        
        # Calculate difference
        mean_diff = injury_mean - non_injury_mean
        mean_diff_pct = (mean_diff / (abs(non_injury_mean) + 1e-6)) * 100 if non_injury_mean != 0 else 0
        
        # Calculate percentiles
        injury_p75 = injury_vals.quantile(0.75)
        non_injury_p75 = non_injury_vals.quantile(0.75)
        
        # Calculate zero percentages
        injury_zero_pct = (injury_vals == 0).sum() / len(injury_vals) * 100
        non_injury_zero_pct = (non_injury_vals == 0).sum() / len(non_injury_vals) * 100
        
        causation_analysis.append({
            'feature': feat,
            'injury_mean': injury_mean,
            'non_injury_mean': non_injury_mean,
            'mean_diff': mean_diff,
            'mean_diff_pct': mean_diff_pct,
            'injury_median': injury_median,
            'non_injury_median': non_injury_median,
            'injury_p75': injury_p75,
            'non_injury_p75': non_injury_p75,
            'injury_zero_pct': injury_zero_pct,
            'non_injury_zero_pct': non_injury_zero_pct,
        })
    
    causation_df = pd.DataFrame(causation_analysis)
    
    # Sort by absolute mean difference (features most associated with injuries)
    causation_df['abs_mean_diff_pct'] = causation_df['mean_diff_pct'].abs()
    causation_df = causation_df.sort_values('abs_mean_diff_pct', ascending=False)
    
    return causation_df, len(injury_df), len(non_injury_df)


def compare_injury_causation(real_cc0_path, synth_cc0_path, cc0_schema, out_dir):
    """Compare injury causation patterns between real and synthetic CC0."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("INJURY CAUSATION COMPARISON: Real vs Synthetic CC0")
    print("=" * 80)
    
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
    print(f"   Real CC0: {len(real_df):,} rows")
    
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
    print(f"   Synthetic CC0: {len(synth_df):,} rows")
    
    # Analyze injury causation for real data
    print("\n3. Analyzing real CC0 injury causation...")
    real_causation, real_n_injuries, real_n_non_injuries = analyze_injury_causation(
        real_df, label_col='y', prefix='Real: '
    )
    
    # Analyze injury causation for synthetic data
    print("\n4. Analyzing synthetic CC0 injury causation...")
    synth_causation, synth_n_injuries, synth_n_non_injuries = analyze_injury_causation(
        synth_df, label_col='y', prefix='Synthetic: '
    )
    
    # Merge for comparison
    print("\n5. Comparing injury causation patterns...")
    comparison = real_causation.merge(
        synth_causation,
        on='feature',
        suffixes=('_real', '_synth'),
        how='inner'
    )
    
    # Calculate differences in injury associations
    comparison['mean_diff_pct_diff'] = comparison['mean_diff_pct_synth'] - comparison['mean_diff_pct_real']
    comparison['abs_mean_diff_pct_diff'] = comparison['mean_diff_pct_diff'].abs()
    
    # Sort by difference in association strength
    comparison = comparison.sort_values('abs_mean_diff_pct_diff', ascending=False)
    
    # Save results
    real_causation.to_csv(out_dir / 'real_injury_causation.csv', index=False)
    synth_causation.to_csv(out_dir / 'synthetic_injury_causation.csv', index=False)
    comparison.to_csv(out_dir / 'injury_causation_comparison.csv', index=False)
    
    # Create summary
    print("\n6. Top 20 features with strongest injury association (Real CC0):")
    print(f"\n{'Rank':<6} {'Feature':<35} {'Injury Mean':<15} {'Non-Injury Mean':<18} {'Diff %':<10}")
    print('-' * 90)
    for i, row in real_causation.head(20).iterrows():
        print(f"{real_causation.index.get_loc(i)+1:<6} {row['feature']:<35} {row['injury_mean']:>12.4f} {row['non_injury_mean']:>15.4f} {row['mean_diff_pct']:>+8.1f}%")
    
    print("\n7. Top 20 features with strongest injury association (Synthetic CC0):")
    print(f"\n{'Rank':<6} {'Feature':<35} {'Injury Mean':<15} {'Non-Injury Mean':<18} {'Diff %':<10}")
    print('-' * 90)
    for i, row in synth_causation.head(20).iterrows():
        print(f"{synth_causation.index.get_loc(i)+1:<6} {row['feature']:<35} {row['injury_mean']:>12.4f} {row['non_injury_mean']:>15.4f} {row['mean_diff_pct']:>+8.1f}%")
    
    print("\n8. Top 20 features with largest differences in injury association:")
    print(f"\n{'Rank':<6} {'Feature':<35} {'Real Diff %':<15} {'Synth Diff %':<15} {'Difference':<15}")
    print('-' * 90)
    for i, row in comparison.head(20).iterrows():
        print(f"{comparison.index.get_loc(i)+1:<6} {row['feature']:<35} {row['mean_diff_pct_real']:>+12.1f}% {row['mean_diff_pct_synth']:>+12.1f}% {row['mean_diff_pct_diff']:>+12.1f}%")
    
    # Create summary statistics
    summary = {
        'real_injury_rate': real_n_injuries / (real_n_injuries + real_n_non_injuries),
        'synth_injury_rate': synth_n_injuries / (synth_n_injuries + synth_n_non_injuries),
        'real_n_injuries': int(real_n_injuries),
        'synth_n_injuries': int(synth_n_injuries),
        'real_top_features': real_causation.head(10)['feature'].tolist(),
        'synth_top_features': synth_causation.head(10)['feature'].tolist(),
        'avg_association_diff': float(comparison['abs_mean_diff_pct_diff'].mean()),
        'max_association_diff': float(comparison['abs_mean_diff_pct_diff'].max()),
    }
    
    with open(out_dir / 'injury_causation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Real injury rate: {summary['real_injury_rate']*100:.2f}%")
    print(f"   Synthetic injury rate: {summary['synth_injury_rate']*100:.2f}%")
    print(f"   Average association difference: {summary['avg_association_diff']:.1f}%")
    print(f"   Max association difference: {summary['max_association_diff']:.1f}%")
    
    print(f"\n✅ Results saved to: {out_dir}")
    
    return real_causation, synth_causation, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-cc0", required=True, help="Real CC0 day file")
    parser.add_argument("--synth-cc0", required=True, help="Synthetic CC0 day file")
    parser.add_argument("--cc0-schema", default="cc0_feature_schema.json", help="CC0 schema file")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    
    compare_injury_causation(
        args.real_cc0,
        args.synth_cc0,
        args.cc0_schema,
        args.out
    )
