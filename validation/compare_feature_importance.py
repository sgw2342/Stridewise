#!/usr/bin/env python3
"""Compare feature importance between models trained on synthetic vs real data.

This validates that models learn similar patterns from synthetic and real data.
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_feature_importance(file_path: str) -> pd.DataFrame:
    """Load feature importance from CSV or JSON."""
    path = Path(file_path)
    if path.suffix == '.csv':
        df = pd.read_csv(path)
        # Assume first column is feature name, second is importance
        if len(df.columns) >= 2:
            df.columns = ['feature', 'importance']
        return df
    elif path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
        # Extract feature importance if in nested structure
        if 'feature_importance' in data:
            return pd.DataFrame(data['feature_importance'])
        else:
            # Assume it's a dict of feature: importance
            return pd.DataFrame(list(data.items()), columns=['feature', 'importance'])
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")

def compare_importance(real_importance: pd.DataFrame, synth_importance: pd.DataFrame, 
                       top_n: int = 20) -> dict:
    """Compare feature importance rankings."""
    # Normalize importance (if needed)
    real_importance['importance_norm'] = real_importance['importance'] / real_importance['importance'].max()
    synth_importance['importance_norm'] = synth_importance['importance'] / synth_importance['importance'].max()
    
    # Get top N features
    real_top = real_importance.nlargest(top_n, 'importance')
    synth_top = synth_importance.nlargest(top_n, 'importance')
    
    # Find common top features
    real_top_set = set(real_top['feature'])
    synth_top_set = set(synth_top['feature'])
    common_top = real_top_set & synth_top_set
    
    # Compute rank correlation
    real_ranks = {feat: idx for idx, feat in enumerate(real_top['feature'])}
    synth_ranks = {feat: idx for idx, feat in enumerate(synth_top['feature'])}
    
    # Rank differences for common features
    rank_diffs = {}
    for feat in common_top:
        rank_diffs[feat] = abs(real_ranks[feat] - synth_ranks[feat])
    
    results = {
        'top_n': top_n,
        'real_top_features': real_top['feature'].tolist(),
        'synthetic_top_features': synth_top['feature'].tolist(),
        'common_top_features': list(common_top),
        'common_ratio': len(common_top) / top_n,
        'average_rank_difference': np.mean(list(rank_diffs.values())) if rank_diffs else None,
        'rank_differences': rank_diffs
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare feature importance between synthetic and real data models')
    parser.add_argument('--real-importance', required=True, help='Feature importance from model trained on real CC0')
    parser.add_argument('--synthetic-importance', required=True, help='Feature importance from model trained on synthetic data')
    parser.add_argument('--out', required=True, help='Output directory for results')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top features to compare')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature importance
    print("Loading feature importance...")
    real_importance = load_feature_importance(args.real_importance)
    synth_importance = load_feature_importance(args.synthetic_importance)
    
    # Compare
    print("Comparing feature importance...")
    results = compare_importance(real_importance, synth_importance, top_n=args.top_n)
    
    # Save results
    results_file = out_dir / 'feature_importance_comparison.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE COMPARISON")
    print("=" * 70)
    print(f"\nTop {args.top_n} Features Overlap: {len(results['common_top_features'])}/{args.top_n} ({results['common_ratio']*100:.1f}%)")
    
    if results['average_rank_difference'] is not None:
        print(f"Average Rank Difference: {results['average_rank_difference']:.2f}")
    
    print("\nCommon Top Features:")
    for feat in results['common_top_features'][:10]:
        rank_diff = results['rank_differences'].get(feat, 'N/A')
        print(f"  {feat}: Rank diff = {rank_diff}")
    
    print(f"\n✅ Feature importance comparison complete!")
    print(f"   Results saved to: {results_file}")

if __name__ == '__main__':
    import numpy as np
    main()
