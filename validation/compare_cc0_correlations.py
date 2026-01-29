#!/usr/bin/env python3
"""Compare correlation structures between synthetic and real CC0 data.

This script validates that feature relationships are preserved in synthetic data.
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def load_cc0_data(day_csv: str) -> pd.DataFrame:
    """Load CC0 day approach data."""
    df = pd.read_csv(day_csv)
    return df

def compute_correlation_matrix(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Compute correlation matrix for specified features."""
    # Select numeric features only
    numeric_df = df[features].select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    return corr_matrix

def compare_correlations(real_corr: pd.DataFrame, synth_corr: pd.DataFrame, 
                         key_relationships: list) -> dict:
    """Compare correlation matrices and key relationships."""
    results = {
        'overall_correlation_diff': np.abs(real_corr - synth_corr).mean().mean(),
        'key_relationships': {}
    }
    
    for rel in key_relationships:
        feat1, feat2 = rel
        if feat1 in real_corr.index and feat2 in real_corr.columns:
            real_corr_val = real_corr.loc[feat1, feat2]
            synth_corr_val = synth_corr.loc[feat1, feat2] if feat1 in synth_corr.index and feat2 in synth_corr.columns else np.nan
            diff = abs(real_corr_val - synth_corr_val) if not np.isnan(synth_corr_val) else np.nan
            
            results['key_relationships'][f"{feat1}_vs_{feat2}"] = {
                'real': float(real_corr_val),
                'synthetic': float(synth_corr_val) if not np.isnan(synth_corr_val) else None,
                'difference': float(diff) if not np.isnan(diff) else None,
                'preserved': diff < 0.1 if not np.isnan(diff) else False
            }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare correlation structures between synthetic and real CC0 data')
    parser.add_argument('--real', required=True, help='Path to real CC0 day CSV')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic CC0 day CSV')
    parser.add_argument('--out', required=True, help='Output directory for results')
    parser.add_argument('--schema', default='./cc0_feature_schema.json', help='CC0 feature schema')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    real_df = load_cc0_data(args.real)
    synth_df = load_cc0_data(args.synthetic)
    
    # Load schema to get feature names
    with open(args.schema) as f:
        schema = json.load(f)
    
    # Get numeric features (exclude labels and IDs)
    numeric_features = [f for f in real_df.columns 
                       if f not in ['y', 'injury', 'maskedID', 'athlete_id', 'anchor_date'] 
                       and pd.api.types.is_numeric_dtype(real_df[f])]
    
    # Common features
    common_features = [f for f in numeric_features if f in synth_df.columns]
    print(f"Comparing {len(common_features)} common features...")
    
    # Compute correlation matrices
    print("Computing correlation matrices...")
    real_corr = compute_correlation_matrix(real_df, common_features)
    synth_corr = compute_correlation_matrix(synth_df, common_features)
    
    # Key relationships to validate (based on feature importance analysis)
    key_relationships = [
        ('z5_day5', 'rest_day6'),
        ('z5_day5', 'success_day5'),
        ('z5_day4', 'rest_day4'),
        ('z5_day6', 'rest_day6'),
        ('sprint_day5', 'z5_day5'),
        ('intensity_share_day5', 'z5_day5'),
    ]
    
    # Compare correlations
    print("Comparing correlations...")
    results = compare_correlations(real_corr, synth_corr, key_relationships)
    
    # Save results
    results_file = out_dir / 'correlation_comparison.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    print("\n" + "=" * 70)
    print("CORRELATION STRUCTURE COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nOverall Correlation Difference: {results['overall_correlation_diff']:.4f}")
    print("\nKey Relationships:")
    for rel_name, rel_data in results['key_relationships'].items():
        print(f"\n  {rel_name}:")
        print(f"    Real: {rel_data['real']:.4f}")
        print(f"    Synthetic: {rel_data['synthetic']:.4f}" if rel_data['synthetic'] is not None else "    Synthetic: N/A")
        if rel_data['difference'] is not None:
            print(f"    Difference: {rel_data['difference']:.4f}")
            print(f"    Preserved: {'✅' if rel_data['preserved'] else '❌'}")
    
    # Visualize correlation differences
    corr_diff = np.abs(real_corr - synth_corr)
    plt.figure(figsize=(12, 10))
    if HAS_SEABORN:
        sns.heatmap(corr_diff, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Absolute Correlation Difference'})
    else:
        plt.imshow(corr_diff, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Absolute Correlation Difference')
    plt.title('Correlation Structure Differences (Real vs Synthetic)')
    plt.tight_layout()
    plt.savefig(out_dir / 'correlation_differences_heatmap.png', dpi=150)
    plt.close()
    print(f"\n✅ Correlation comparison complete!")
    print(f"   Results saved to: {results_file}")
    print(f"   Heatmap saved to: {out_dir / 'correlation_differences_heatmap.png'}")

if __name__ == '__main__':
    main()
