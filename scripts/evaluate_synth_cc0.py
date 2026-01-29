#!/usr/bin/env python3
"""Evaluate synthetic CC0 data with trained standalone model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt
from synthrun_gen.eval.metrics import (
    calibration_bins,
    expected_calibration_error,
)
import joblib

from synthrun_gen.cc0.prepare import prepare_cc0_features
from synthrun_gen.models.zscore import PerAthleteHealthyZScorer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth-cc0", required=True, help="Synthetic CC0 day file")
    p.add_argument("--model-dir", required=True, help="Directory with trained model")
    p.add_argument("--cc0-schema", default="cc0_feature_schema.json")
    p.add_argument("--cc0-label-col", default="injury")
    p.add_argument("--zscorer-path", default=None, help="Optional path to custom z-scorer (for synthetic data)")
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path(args.model_dir)
    
    print("=" * 70)
    print("EVALUATING SYNTHETIC CC0 WITH TRAINED STANDALONE MODEL")
    print("=" * 70)
    
    # Load model
    print("\n1. Loading trained model...")
    # Try different model file names
    if (model_dir / "bagged_models.joblib").exists():
        from stridewise_train_standalone_cc0_paper_method import predict_bagged
        models = joblib.load(model_dir / "bagged_models.joblib")
        model = models  # Will use predict_bagged function
        use_bagged = True
        print(f"   Loaded bagged models: {len(models)} models")
    elif (model_dir / "standalone_model.joblib").exists():
        model = joblib.load(model_dir / "standalone_model.joblib")
        use_bagged = False
        print(f"   Loaded single model")
    else:
        raise FileNotFoundError(f"Model file not found in {model_dir}")
    
    with open(model_dir / "model_meta.json", "r") as f:
        model_meta = json.load(f)
    
    print(f"   Model type: {model_meta.get('model_type', 'unknown')}")
    print(f"   Features: {model_meta.get('n_features', 'unknown')}")
    
    # Load z-scorer (allow override for synthetic data)
    print("\n2. Loading z-scorer...")
    zscorer_path = model_dir / "zscorer.json"
    if hasattr(args, 'zscorer_path') and args.zscorer_path:
        zscorer_path = Path(args.zscorer_path)
        print(f"   Using custom z-scorer: {zscorer_path}")
    with open(zscorer_path, "r") as f:
        z_dict = json.load(f)
    z = PerAthleteHealthyZScorer.from_dict(z_dict)
    
    # Load synthetic CC0 data
    print("\n3. Loading synthetic CC0 data...")
    cc0_df, cc0_meta = prepare_cc0_features(
        args.synth_cc0,
        None,  # No week file
        args.cc0_schema,
        label_col=args.cc0_label_col,
        engineer_features=False,  # Don't engineer features (use raw CC0 format)
    )
    
    print(f"   Rows: {len(cc0_df):,}")
    print(f"   Label prevalence: {cc0_df['y'].mean():.4f} ({100*cc0_df['y'].mean():.2f}%)")
    
    # Create day approach features (required before z-scoring)
    print("\n4. Creating day approach features...")
    from stridewise_train_standalone_cc0_paper_method import create_day_approach_features
    cc0_df, day_approach_features = create_day_approach_features(cc0_df)
    print(f"   Created {len(day_approach_features)} day approach features")
    
    # Get feature columns (should match model_meta['feature_cols'])
    feature_cols = model_meta.get('feature_cols', day_approach_features)
    if not feature_cols:
        # Fallback: use day approach features
        feature_cols = day_approach_features
    
    print(f"   Features to use: {len(feature_cols)}")
    
    # Check if all required features are present
    missing_features = [c for c in feature_cols if c not in cc0_df.columns]
    if missing_features:
        print(f"   ⚠️  WARNING: {len(missing_features)} features missing:")
        for feat in missing_features[:10]:
            print(f"      - {feat}")
        if len(missing_features) > 10:
            print(f"      ... and {len(missing_features) - 10} more")
        # Use only features that exist
        feature_cols = [c for c in feature_cols if c in cc0_df.columns]
        print(f"   Using {len(feature_cols)} available features")
    
    # Transform with z-scorer (use correct athlete column name)
    print("\n5. Transforming data with z-scorer...")
    # Check athlete column name in data
    athlete_col_name = "athlete_id"
    if "Athlete ID" in cc0_df.columns:
        athlete_col_name = "Athlete ID"
    elif "athlete_id" not in cc0_df.columns:
        # Try to find any ID column
        id_candidates = [c for c in cc0_df.columns if "id" in c.lower() or "athlete" in c.lower()]
        if id_candidates:
            athlete_col_name = id_candidates[0]
            print(f"   Using athlete column: {athlete_col_name}")
    
    cc0_z = z.transform(cc0_df, athlete_col=athlete_col_name)
    
    # Check which features exist after transformation
    available_features = [c for c in feature_cols if c in cc0_z.columns]
    missing_features = [c for c in feature_cols if c not in cc0_z.columns]
    
    if missing_features:
        print(f"   ⚠️  WARNING: {len(missing_features)} features missing after z-scoring:")
        for feat in missing_features[:10]:
            print(f"      - {feat}")
        if len(missing_features) > 10:
            print(f"      ... and {len(missing_features) - 10} more")
        print(f"   ⚠️  These features will be filled with zeros")
    
    # Create X with all required features, filling missing ones with zeros
    X = pd.DataFrame(index=cc0_z.index)
    for c in feature_cols:
        if c in cc0_z.columns:
            X[c] = pd.to_numeric(cc0_z[c], errors="coerce").fillna(0.0)
        else:
            X[c] = 0.0  # Fill missing features with zeros
    
    y = cc0_z["y"].to_numpy(dtype=int) if "y" in cc0_z.columns else None
    
    print(f"   Data shape: {X.shape}")
    print(f"   Available features: {len(available_features)}/{len(feature_cols)}")
    
    # Make predictions
    print("\n6. Making predictions...")
    if use_bagged:
        from stridewise_train_standalone_cc0_paper_method import predict_bagged
        p = predict_bagged(model, X)
    elif hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[:, 1]
    else:
        p = model.predict(X).astype(float)
    
    print(f"   Prediction range: [{p.min():.4f}, {p.max():.4f}]")
    print(f"   Mean prediction: {p.mean():.4f}")
    
    # Calculate metrics
    if y is not None:
        print("\n7. Calculating metrics...")
        auc = roc_auc_score(y, p)
        pr_auc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        logloss = log_loss(y, p)
        
        # Calibration metrics
        edges, counts, mean_p, frac_pos = calibration_bins(y, p, n_bins=10)
        ece = expected_calibration_error(counts, mean_p, frac_pos)
        
        # Calculate precision/recall at different thresholds
        precision, recall, thresholds = precision_recall_curve(y, p)
        fpr, tpr, roc_thresholds = roc_curve(y, p)
        
        # Find threshold that gives max F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_f1_idx]
        best_precision = precision[best_f1_idx]
        best_recall = recall[best_f1_idx]
        
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE ON SYNTHETIC CC0 DATA")
        print("=" * 70)
        print(f"\nLabel Prevalence: {np.mean(y):.4f} ({100*np.mean(y):.2f}%)")
        print(f"\nMetrics:")
        print(f"  ROC AUC:     {auc:.4f}")
        print(f"  PR AUC:      {pr_auc:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Log Loss:    {logloss:.4f}")
        print(f"  ECE:         {ece:.4f}" if ece else "  ECE:         N/A")
        print(f"\nBest F1 Score:")
        print(f"  F1:          {best_f1:.4f}")
        print(f"  Precision:   {best_precision:.4f}")
        print(f"  Recall:      {best_recall:.4f}")
        print(f"  Threshold:   {best_f1_threshold:.4f}")
        
        # Compare to real CC0 performance (if available in model_meta)
        if 'metrics' in model_meta:
            real_metrics = model_meta['metrics']
            print(f"\n" + "=" * 70)
            print("COMPARISON WITH REAL CC0 TEST PERFORMANCE")
            print("=" * 70)
            if 'test_auc' in real_metrics:
                print(f"  ROC AUC:     {auc:.4f} (synthetic) vs {real_metrics['test_auc']:.4f} (real test)")
                print(f"  Difference:  {auc - real_metrics['test_auc']:+.4f}")
            if 'test_pr_auc' in real_metrics:
                print(f"  PR AUC:      {pr_auc:.4f} (synthetic) vs {real_metrics['test_pr_auc']:.4f} (real test)")
                print(f"  Difference:  {pr_auc - real_metrics['test_pr_auc']:+.4f}")
        
        metrics = {
            "n_rows": int(len(X)),
            "prevalence": float(np.mean(y)),
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc),
            "brier": float(brier),
            "logloss": float(logloss),
            "ece": float(ece) if ece else None,
            "best_f1": float(best_f1),
            "best_precision": float(best_precision),
            "best_recall": float(best_recall),
            "best_threshold": float(best_f1_threshold),
            "prediction_range": [float(p.min()), float(p.max())],
            "mean_prediction": float(p.mean()),
        }
        
        # Add comparison to real CC0 if available
        if 'metrics' in model_meta:
            real_metrics = model_meta['metrics']
            if 'test_auc' in real_metrics:
                metrics['real_test_auc'] = real_metrics['test_auc']
                metrics['auc_diff'] = float(auc - real_metrics['test_auc'])
            if 'test_pr_auc' in real_metrics:
                metrics['real_test_pr_auc'] = real_metrics['test_pr_auc']
                metrics['pr_auc_diff'] = float(pr_auc - real_metrics['test_pr_auc'])
    else:
        print("\n6. No labels available - predictions only")
        metrics = {
            "n_rows": int(len(X)),
            "prevalence": None,
        }
    
    # Save predictions
    pred_df = pd.DataFrame({
        "row_idx": cc0_df["row_idx"].values if "row_idx" in cc0_df.columns else range(len(X)),
        "p": p,
    })
    if y is not None:
        pred_df["y"] = y
    
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"\n✅ Predictions saved to: {out_dir / 'predictions.csv'}")
    
    # Create calibration curve
    if y is not None:
        print("\n8. Creating calibration curve...")
        try:
            edges, counts, mean_p, frac_pos = calibration_bins(y, p, n_bins=10)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1)
            valid_bins = ~np.isnan(mean_p) & ~np.isnan(frac_pos)
            if valid_bins.sum() > 0:
                ax.plot(mean_p[valid_bins], frac_pos[valid_bins], 'o-', label='Model', linewidth=2, markersize=8)
            ax.set_xlabel('Mean Predicted Probability', fontsize=12)
            ax.set_ylabel('Fraction of Positives', fontsize=12)
            ece_val = metrics.get('ece', None)
            ax.set_title(f'Calibration Curve\nECE = {ece_val:.4f}' if ece_val else 'Calibration Curve', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "calibration_curve.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {out_dir / 'calibration_curve.png'}")
        except Exception as e:
            print(f"   ⚠️  Could not create calibration curve: {e}")
    
    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: {out_dir / 'metrics.json'}")
    
    return metrics

if __name__ == "__main__":
    main()
