#!/usr/bin/env python3
"""Evaluate a trained main_model model on a new dataset.

Usage:
  python evaluate_main_model.py \
    --model-dir ./main_model_xgboost_best \
    --daily ./improved_elite_250_v3/daily.csv \
    --users ./improved_elite_250_v3/users.csv \
    --activities ./improved_elite_250_v3/activities.csv \
    --out ./eval_250_elite_main_model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from synthrun_gen.eval.metrics import (
    calibration_bins,
    expected_calibration_error,
)

try:
    import xgboost as xgb
except ImportError:
    raise RuntimeError("xgboost not installed")

from stridewise_train_main_model import _read_any
from synthrun_gen.main_model.features import build_main_model_table


def _compute_metrics(y_true, y_pred):
    """Compute classification metrics including calibration."""
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = None
    try:
        pr_auc = average_precision_score(y_true, y_pred)
    except ValueError:
        pr_auc = None
    try:
        brier = brier_score_loss(y_true, y_pred)
    except ValueError:
        brier = None
    try:
        logloss = log_loss(y_true, y_pred)
    except ValueError:
        logloss = None
    
    # Calibration metrics
    ece = None
    try:
        edges, counts, mean_p, frac_pos = calibration_bins(y_true, y_pred, n_bins=10)
        ece = expected_calibration_error(counts, mean_p, frac_pos)
    except Exception:
        pass
    
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "logloss": logloss,
        "ece": ece,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained main_model model on new dataset")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained model")
    parser.add_argument("--daily", required=True, help="Path to daily.csv")
    parser.add_argument("--users", default=None, help="Path to users.csv (optional)")
    parser.add_argument("--activities", default=None, help="Path to activities.csv (optional)")
    parser.add_argument("--out", required=True, help="Output directory for evaluation results")
    parser.add_argument("--label", default="injury_next_7d", help="Label column")
    parser.add_argument("--min-history-days", type=int, default=28, help="Drop first N days per athlete")
    parser.add_argument("--include-users", action="store_true", default=True, help="Include user features")
    parser.add_argument("--include-activity-aggs", action="store_true", help="Include activity aggregates")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EVALUATING MAIN_MODEL MODEL ON NEW DATASET")
    print("=" * 70)
    
    # Load model and schema
    print("\n1. Loading trained model...")
    model_path = None
    for ext in [".json", ".cbm", ".txt"]:
        candidate = model_dir / f"main_model_xgb{ext}" if ext == ".json" else model_dir / f"main_model_cat{ext}" if ext == ".cbm" else model_dir / f"main_model_lgb{ext}"
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        # Try to find any model file
        model_files = list(model_dir.glob("main_model_*.json")) + list(model_dir.glob("main_model_*.cbm")) + list(model_dir.glob("main_model_*.txt"))
        if model_files:
            model_path = model_files[0]
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")
    
    print(f"   Model file: {model_path}")
    
    # Determine model type
    if model_path.suffix == ".json":
        model_type = "xgboost"
        booster = xgb.Booster()
        booster.load_model(str(model_path))
    elif model_path.suffix == ".cbm":
        model_type = "catboost"
        import catboost as cb
        booster = cb.CatBoostClassifier()
        booster.load_model(str(model_path))
    elif model_path.suffix == ".txt":
        model_type = "lightgbm"
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(model_path))
    else:
        raise ValueError(f"Unknown model type: {model_path.suffix}")
    
    print(f"   Model type: {model_type}")
    
    # Load schema
    schema_path = model_dir / "main_model_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path) as f:
        schema_dict = json.load(f)
    
    # Extract feature columns from schema
    if "feature_cols" in schema_dict:
        feature_cols = schema_dict["feature_cols"]
    else:
        # Fallback: try to infer from schema structure
        feature_cols = schema_dict.get("features", [])
    
    print(f"   Features: {len(feature_cols)}")
    
    # Load data
    print("\n2. Loading dataset...")
    daily_df = _read_any(args.daily)
    print(f"   Daily rows: {len(daily_df):,}")
    
    users_df = None
    if args.users:
        users_df = _read_any(args.users)
        print(f"   Users: {len(users_df):,}")
    
    activities_df = None
    if args.activities:
        activities_df = _read_any(args.activities)
        print(f"   Activities: {len(activities_df):,}")
    
    # Build feature table
    print("\n3. Building feature table...")
    df_model, schema_new = build_main_model_table(
        daily=daily_df,
        users=users_df,
        activities=activities_df,
        label_col=args.label,
        include_users=args.include_users,
        include_activity_aggs=args.include_activity_aggs,
    )
    
    # Drop first N days per athlete
    if args.min_history_days > 0:
        df_model = df_model.sort_values(["user_id", "date"]).reset_index(drop=True)
        keep = df_model.groupby("user_id", sort=False).cumcount() >= args.min_history_days
        df_model = df_model.loc[keep].reset_index(drop=True)
    
    print(f"   Model rows: {len(df_model):,}")
    
    # Check label
    if args.label not in df_model.columns:
        raise ValueError(f"Label column '{args.label}' not found in data")
    
    y = df_model[args.label].values
    print(f"   Label prevalence: {np.mean(y):.4f} ({np.sum(y)}/{len(y)})")
    
    # Prepare features
    print("\n4. Preparing features...")
    # Ensure we only use features that exist in both schema and data
    available_features = [f for f in feature_cols if f in df_model.columns]
    missing_features = [f for f in feature_cols if f not in df_model.columns]
    
    if missing_features:
        print(f"   ⚠️  Warning: {len(missing_features)} features missing from data")
        print(f"      Missing: {missing_features[:10]}...")
    
    X = df_model[available_features].copy()
    
    # Handle missing values
    X = X.fillna(0.0)
    
    print(f"   Feature matrix: {X.shape}")
    print(f"   Missing values: {X.isna().sum().sum()}")
    
    # Make predictions
    print("\n5. Making predictions...")
    if model_type == "xgboost":
        dtest = xgb.DMatrix(X, label=y)
        predictions = booster.predict(dtest)
    elif model_type == "catboost":
        predictions = booster.predict_proba(X)[:, 1]
    elif model_type == "lightgbm":
        predictions = booster.predict(X, num_iteration=booster.best_iteration)
    
    print(f"   Prediction range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
    print(f"   Mean prediction: {np.mean(predictions):.4f}")
    
    # Compute metrics
    print("\n6. Computing metrics...")
    metrics = _compute_metrics(y, predictions)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nDataset: {args.daily}")
    print(f"Rows: {len(df_model):,}")
    print(f"Label prevalence: {np.mean(y):.4f} ({np.sum(y)}/{len(y)})")
    print(f"\nMetrics:")
    print(f"  ROC AUC:      {metrics['auc']:.4f}" if metrics['auc'] else "  ROC AUC:      N/A")
    print(f"  PR AUC:       {metrics['pr_auc']:.4f}" if metrics['pr_auc'] else "  PR AUC:       N/A")
    print(f"  Brier Score:  {metrics['brier']:.4f}" if metrics['brier'] else "  Brier Score:  N/A")
    print(f"  Log Loss:     {metrics['logloss']:.4f}" if metrics['logloss'] else "  Log Loss:     N/A")
    print(f"  ECE:          {metrics['ece']:.4f}" if metrics['ece'] is not None else "  ECE:          N/A")
    
    # Save results
    results = {
        "dataset": str(args.daily),
        "model_dir": str(args.model_dir),
        "model_type": model_type,
        "n_rows": int(len(df_model)),
        "n_features": int(len(available_features)),
        "n_missing_features": int(len(missing_features)),
        "label_prevalence": float(np.mean(y)),
        "label_count": int(np.sum(y)),
        "metrics": metrics,
        "prediction_summary": {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "p01": float(np.percentile(predictions, 1)),
            "p05": float(np.percentile(predictions, 5)),
            "p50": float(np.percentile(predictions, 50)),
            "p95": float(np.percentile(predictions, 95)),
            "p99": float(np.percentile(predictions, 99)),
        }
    }
    
    results_file = out_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    id_cols = ["user_id"]
    if "date" in df_model.columns:
        id_cols.append("date")
    elif "day" in df_model.columns:
        id_cols.append("day")
    
    pred_df = df_model[id_cols].copy()
    pred_df["label"] = y
    pred_df["prediction"] = predictions
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    
    # Create calibration curve
    print("\n7. Creating calibration curve...")
    try:
        edges, counts, mean_p, frac_pos = calibration_bins(y, predictions, n_bins=10)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1)
        valid_bins = ~np.isnan(mean_p) & ~np.isnan(frac_pos)
        if valid_bins.sum() > 0:
            ax.plot(mean_p[valid_bins], frac_pos[valid_bins], 'o-', label='Model', linewidth=2, markersize=8)
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Calibration Curve\nECE = {metrics["ece"]:.4f}' if metrics['ece'] else 'Calibration Curve', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "calibration_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {out_dir / 'calibration_curve.png'}")
    except Exception as e:
        print(f"   ⚠️  Could not create calibration curve: {e}")
    
    # Threshold stability across cohorts (if profile/user info available)
    print("\n8. Analyzing threshold stability across cohorts...")
    threshold_stability = {}
    try:
        if users_df is not None and 'profile' in users_df.columns:
            # Merge profile info
            pred_df_with_profile = pred_df.merge(
                users_df[['user_id', 'profile']], 
                on='user_id', 
                how='left'
            )
            
            if 'profile' in pred_df_with_profile.columns:
                threshold_stability['by_profile'] = {}
                for profile in pred_df_with_profile['profile'].dropna().unique():
                    profile_mask = pred_df_with_profile['profile'] == profile
                    profile_y = y[profile_mask]
                    profile_p = predictions[profile_mask]
                    
                    if len(np.unique(profile_y)) >= 2:
                        try:
                            profile_auc = roc_auc_score(profile_y, profile_p)
                            profile_brier = brier_score_loss(profile_y, profile_p)
                            edges_p, counts_p, mean_p_p, frac_pos_p = calibration_bins(profile_y, profile_p, n_bins=10)
                            profile_ece = expected_calibration_error(counts_p, mean_p_p, frac_pos_p)
                            
                            threshold_stability['by_profile'][profile] = {
                                'n_samples': int(profile_mask.sum()),
                                'prevalence': float(np.mean(profile_y)),
                                'auc': float(profile_auc),
                                'brier': float(profile_brier),
                                'ece': float(profile_ece) if profile_ece else None,
                            }
                        except Exception:
                            pass
                
                print(f"   ✅ Analyzed {len(threshold_stability['by_profile'])} profiles")
        else:
            print(f"   ⚠️  No profile information available for cohort analysis")
    except Exception as e:
        print(f"   ⚠️  Could not analyze threshold stability: {e}")
    
    # Add threshold stability to results
    if threshold_stability:
        results['threshold_stability'] = threshold_stability
    
    # Save updated results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Results: {results_file}")
    print(f"   Predictions: {out_dir / 'predictions.csv'}")
    if threshold_stability:
        print(f"   Threshold stability: Analyzed across {len(threshold_stability.get('by_profile', {}))} cohorts")
    
    # Compare with training metrics if available
    training_metrics_file = model_dir / "main_model_metrics.json"
    if training_metrics_file.exists():
        print("\n" + "=" * 70)
        print("COMPARISON WITH TRAINING PERFORMANCE")
        print("=" * 70)
        with open(training_metrics_file) as f:
            train_metrics = json.load(f)
        
        print(f"\nTraining (forward-time validation):")
        print(f"  ROC AUC:      {train_metrics.get('auc_val', 'N/A'):.4f}" if train_metrics.get('auc_val') else "  ROC AUC:      N/A")
        print(f"  PR AUC:       {train_metrics.get('pr_auc_val', 'N/A'):.4f}" if train_metrics.get('pr_auc_val') else "  PR AUC:       N/A")
        
        print(f"\nEvaluation (new dataset):")
        print(f"  ROC AUC:      {metrics['auc']:.4f}" if metrics['auc'] else "  ROC AUC:      N/A")
        print(f"  PR AUC:       {metrics['pr_auc']:.4f}" if metrics['pr_auc'] else "  PR AUC:       N/A")
        
        if metrics['auc'] and train_metrics.get('auc_val'):
            diff = metrics['auc'] - train_metrics['auc_val']
            print(f"\nDifference:")
            print(f"  AUC change:   {diff:+.4f}")
            if diff > 0:
                print(f"  ✅ Performance improved on new dataset")
            elif diff < -0.05:
                print(f"  ⚠️  Performance decreased on new dataset")
            else:
                print(f"  ➡️  Performance similar to training")


if __name__ == "__main__":
    main()
