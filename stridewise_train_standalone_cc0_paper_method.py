#!/usr/bin/env python3
"""Standalone model training using the paper's methodology.

Based on: Lövdal et al. (2021) "Injury Prediction in Competitive Runners With Machine Learning"
- Day approach: Training load from previous 7 days as time series (10 features per day)
- Bagged XGBoost models
- AUC achieved: 0.724 (day approach), 0.678 (week approach)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from synthrun_gen.cc0.prepare import prepare_cc0_features
from synthrun_gen.models.zscore import PerAthleteHealthyZScorer

try:
    import xgboost as xgb
except ImportError:
    raise RuntimeError("xgboost not installed")


def forward_time_split(df: pd.DataFrame, athlete_col: str, test_frac: float, time_col: str = None):
    """Per-athlete forward-time split.
    
    If time_col is provided, sorts by time within each athlete.
    Otherwise, uses order in dataframe.
    """
    df = df.reset_index(drop=True)
    train_idx = []
    test_idx = []
    
    # Sort by time if available
    if time_col and time_col in df.columns:
        df = df.sort_values([athlete_col, time_col])
    
    for athlete_id, group in df.groupby(athlete_col, sort=False):
        n = len(group)
        split_idx = int(n * (1 - test_frac))
        train_idx.extend(group.index[:split_idx].tolist())
        test_idx.extend(group.index[split_idx:].tolist())
    
    return train_idx, test_idx


def create_day_approach_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Create day approach features as per paper.
    
    Paper: "Training load from the previous 7 days was expressed as a time series,
    with each day's training being described by 10 features."
    
    The 10 features per day are likely:
    1. Distance (kms)
    2. Sessions
    3. Z3-4 kms
    4. Z5 kms
    5. Sprint kms
    6. Perceived exertion
    7. Perceived training success
    8. Perceived recovery (if available)
    9. Intensity share
    10. Rest day indicator
    
    We'll use the 7-day window (days 0-6) and create time series features.
    """
    df = df.copy()
    
    # Helper to get day values - CC0 uses suffixes: '.6', '.5', '.4', '.3', '.2', '.1', ''
    # According to README: .6 = day before event, .5 = 2 days before, ..., '' = 7 days before
    def get_day_values(base_col: str) -> np.ndarray:
        """Get values for days 0-6 (where 0 = 7 days before, 6 = day before event)."""
        values = []
        # CC0 format: suffix '' = 7 days before, suffix '1' = 6 days before, ..., suffix '6' = day before
        # We want: day0 = 7 days before, day1 = 6 days before, ..., day6 = day before
        # So we use suffixes in order: ['', '1', '2', '3', '4', '5', '6']
        # Real CC0 uses: "total km" (no suffix), "total km.1", ..., "total km.6"
        # Synthetic CC0 uses: "kms." (with dot), "kms.1", ..., "kms.6"
        suffixes = ['', '1', '2', '3', '4', '5', '6']  # In order: 7 days before to day before
        
        for suffix in suffixes:
            col = f"{base_col}.{suffix}" if suffix else f"{base_col}"
            # Synthetic format uses '.' for day 0 instead of no suffix
            if suffix == '':
                col_synth = f"{base_col}."
                if col_synth in df.columns:
                    col = col_synth
            
            if col in df.columns:
                val = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values
            else:
                val = np.zeros(len(df))
            values.append(val)
        return np.array(values).T  # Shape: (n_rows, 7)
    
    # Get the 10 core features per day (7-day time series)
    feature_names = []
    
    # Try to find the base column names - real CC0 uses "total km", "nr. sessions", etc.
    kms_base = None
    if any('total km' in c for c in df.columns):
        kms_base = 'total km'
    elif any('kms.' in c for c in df.columns):
        kms_base = 'kms'
    elif any('total_km' in c for c in df.columns):
        kms_base = 'total_km'
    
    sessions_base = None
    if any('nr. sessions' in c for c in df.columns):
        sessions_base = 'nr. sessions'
    elif any('sessions.' in c for c in df.columns):
        sessions_base = 'sessions'
    elif any('nr_sessions' in c for c in df.columns):
        sessions_base = 'nr_sessions'
    
    # 1. Distance (kms)
    if kms_base:
        kms = get_day_values(kms_base)
        for i in range(7):
            df[f'kms_day{i}'] = kms[:, i]
            feature_names.append(f'kms_day{i}')
    else:
        # Create zeros if not found
        for i in range(7):
            df[f'kms_day{i}'] = 0.0
            feature_names.append(f'kms_day{i}')
    
    # 2. Sessions
    if sessions_base:
        sessions = get_day_values(sessions_base)
        for i in range(7):
            df[f'sessions_day{i}'] = sessions[:, i]
            feature_names.append(f'sessions_day{i}')
    else:
        for i in range(7):
            df[f'sessions_day{i}'] = 0.0
            feature_names.append(f'sessions_day{i}')
    
    # 3. Z3-4 kms - real CC0 uses "km Z3-4"
    z3_4_base = None
    if any('km Z3-4' in c for c in df.columns):
        z3_4_base = 'km Z3-4'
    elif any('kms_z3_4' in c for c in df.columns):
        z3_4_base = 'kms_z3_4'
    
    if z3_4_base:
        z3_4 = get_day_values(z3_4_base)
        for i in range(7):
            df[f'z3_4_day{i}'] = z3_4[:, i]
            feature_names.append(f'z3_4_day{i}')
    else:
        for i in range(7):
            df[f'z3_4_day{i}'] = 0.0
            feature_names.append(f'z3_4_day{i}')
    
    # 4. Z5 kms - real CC0 uses "km Z5-T1-T2"
    z5_base = None
    if any('km Z5-T1-T2' in c for c in df.columns):
        z5_base = 'km Z5-T1-T2'
    elif any('kms_z5_t1_t2' in c for c in df.columns):
        z5_base = 'kms_z5_t1_t2'
    
    if z5_base:
        z5 = get_day_values(z5_base)
        for i in range(7):
            df[f'z5_day{i}'] = z5[:, i]
            feature_names.append(f'z5_day{i}')
    else:
        for i in range(7):
            df[f'z5_day{i}'] = 0.0
            feature_names.append(f'z5_day{i}')
    
    # 5. Sprint kms - real CC0 uses "km sprinting"
    sprint_base = None
    if any('km sprinting' in c for c in df.columns):
        sprint_base = 'km sprinting'
    elif any('kms_sprinting' in c for c in df.columns):
        sprint_base = 'kms_sprinting'
    
    if sprint_base:
        sprint = get_day_values(sprint_base)
        for i in range(7):
            df[f'sprint_day{i}'] = sprint[:, i]
            feature_names.append(f'sprint_day{i}')
    else:
        for i in range(7):
            df[f'sprint_day{i}'] = 0.0
            feature_names.append(f'sprint_day{i}')
    
    # 6. Perceived exertion - real CC0 uses "perceived exertion"
    exertion_base = None
    if any('perceived exertion' in c for c in df.columns):
        exertion_base = 'perceived exertion'
    elif any('perceived_exertion' in c for c in df.columns):
        exertion_base = 'perceived_exertion'
    
    if exertion_base:
        exertion = get_day_values(exertion_base)
        for i in range(7):
            df[f'exertion_day{i}'] = exertion[:, i]
            feature_names.append(f'exertion_day{i}')
    else:
        for i in range(7):
            df[f'exertion_day{i}'] = 0.0
            feature_names.append(f'exertion_day{i}')
    
    # 7. Perceived training success - real CC0 uses "perceived trainingSuccess"
    success_base = None
    if any('perceived trainingSuccess' in c for c in df.columns):
        success_base = 'perceived trainingSuccess'
    elif any('perceived_trainingSuccess' in c for c in df.columns):
        success_base = 'perceived_trainingSuccess'
    
    if success_base:
        success = get_day_values(success_base)
        for i in range(7):
            df[f'success_day{i}'] = success[:, i]
            feature_names.append(f'success_day{i}')
    else:
        for i in range(7):
            df[f'success_day{i}'] = 0.0
            feature_names.append(f'success_day{i}')
    
    # 8. Perceived recovery - real CC0 uses "perceived recovery"
    recovery_base = None
    if any('perceived recovery' in c for c in df.columns):
        recovery_base = 'perceived recovery'
    elif any('perceived_recovery' in c for c in df.columns):
        recovery_base = 'perceived_recovery'
    
    if recovery_base:
        recovery = get_day_values(recovery_base)
        for i in range(7):
            df[f'recovery_day{i}'] = recovery[:, i]
            feature_names.append(f'recovery_day{i}')
    else:
        for i in range(7):
            df[f'recovery_day{i}'] = 0.0
            feature_names.append(f'recovery_day{i}')
    
    # 9. Intensity share (Z5 + sprint / total)
    for i in range(7):
        total = df[f'kms_day{i}'].values if f'kms_day{i}' in df.columns else np.zeros(len(df))
        z5_val = df[f'z5_day{i}'].values if f'z5_day{i}' in df.columns else np.zeros(len(df))
        sprint_val = df[f'sprint_day{i}'].values if f'sprint_day{i}' in df.columns else np.zeros(len(df))
        df[f'intensity_share_day{i}'] = np.where(
            total > 0,
            (z5_val + sprint_val) / (total + 1e-6),
            0.0
        )
        feature_names.append(f'intensity_share_day{i}')
    
    # 10. Rest day indicator
    if any('rest_day.' in c for c in df.columns):
        rest = get_day_values('rest_day')
        for i in range(7):
            df[f'rest_day{i}'] = rest[:, i]
            feature_names.append(f'rest_day{i}')
    else:
        # Create from sessions
        for i in range(7):
            sess_col = f'sessions_day{i}'
            if sess_col in df.columns:
                df[f'rest_day{i}'] = (df[sess_col] == 0).astype(float)
            else:
                df[f'rest_day{i}'] = 0.0
            feature_names.append(f'rest_day{i}')
    
    # Ensure all features are numeric
    for col in feature_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    return df, feature_names


def create_bagged_xgboost(X_train, y_train, n_models=10, seed=42):
    """
    Create bagged XGBoost models as per paper.
    
    Paper: "A predictive system based on bagged XGBoost machine-learning models"
    """
    models = []
    
    for i in range(n_models):
        model = xgb.XGBClassifier(
            n_estimators=200,  # Paper doesn't specify, using reasonable default
            learning_rate=0.05,
            max_depth=5,  # Paper doesn't specify, using reasonable default
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,  # Paper doesn't specify, using default
            reg_alpha=0.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            random_state=seed + i,  # Different seed for each model
            tree_method="hist",
            n_jobs=1,  # Single thread per model
        )
        model.fit(X_train, y_train)
        models.append(model)
    
    return models


def predict_bagged(models, X):
    """Average predictions from bagged models."""
    predictions = np.zeros(len(X))
    for model in models:
        predictions += model.predict_proba(X)[:, 1]
    return predictions / len(models)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cc0-day", required=True, help="CC0 day file")
    p.add_argument("--cc0-schema", default="cc0_feature_schema.json")
    p.add_argument("--cc0-label-col", default=None)
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--min-healthy", type=int, default=10)
    p.add_argument("--n-models", type=int, default=10, help="Number of bagged models")
    p.add_argument("--approach", default="day", choices=["day", "week"], help="Day or week approach")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy schema
    schema_src = Path(args.cc0_schema)
    if schema_src.exists():
        (out_dir / "cc0_feature_schema.json").write_text(schema_src.read_text())
    
    print("=" * 70)
    print("PAPER METHODOLOGY: BAGGED XGBOOST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"   Approach: {args.approach}")
    print(f"   Number of bagged models: {args.n_models}")
    print(f"   Paper achieved: AUC 0.724 (day), 0.678 (week)")
    
    # Load CC0 data
    print("\n1. Loading CC0 data...")
    # Try to detect label column - real CC0 data uses 'injury'
    label_col = args.cc0_label_col
    if label_col is None:
        # Try common label column names
        import pandas as pd
        sample = pd.read_csv(args.cc0_day, nrows=1)
        for candidate in ['injury', 'injury_next_7d', 'y', 'label']:
            if candidate in sample.columns:
                label_col = candidate
                break
    
    cc0_df, cc0_meta = prepare_cc0_features(
        args.cc0_day,
        None,  # No week file
        args.cc0_schema,
        label_col=label_col,
        engineer_features=False,  # Don't use our engineered features, use paper's approach
    )
    
    print(f"   Rows: {len(cc0_df):,}")
    
    # Create features using paper's methodology
    if args.approach == "day":
        print("\n2. Creating day approach features (7-day time series)...")
        cc0_df, feature_cols = create_day_approach_features(cc0_df)
        print(f"   Features created: {len(feature_cols)}")
    else:
        # Week approach - would need week aggregation features
        # For now, we'll focus on day approach
        raise NotImplementedError("Week approach not yet implemented")
    
    # Forward-time split
    print("\n3. Creating train/test split...")
    # Try to find time column for proper sorting
    time_col = None
    for col in ['anchor_date', 'date', 'day']:
        if col in cc0_df.columns:
            time_col = col
            break
    
    train_idx, test_idx = forward_time_split(cc0_df[["athlete_id"]], "athlete_id", args.test_frac, time_col=time_col)
    
    cc0_train = cc0_df.iloc[train_idx].reset_index(drop=True)
    cc0_test = cc0_df.iloc[test_idx].reset_index(drop=True)
    
    print(f"   Train: {len(cc0_train):,} rows")
    print(f"   Test: {len(cc0_test):,} rows")
    
    # Fit z-scorer on train (healthy baseline)
    print("\n4. Fitting z-scorer...")
    z = PerAthleteHealthyZScorer(min_count=args.min_healthy)
    z.fit(cc0_train, athlete_col="athlete_id", y_col="y", feature_cols=feature_cols)
    
    with open(out_dir / "zscorer.json", "w", encoding="utf-8") as f:
        json.dump(z.to_dict(), f, indent=2)
    
    # Transform data
    cc0_train_z = z.transform(cc0_train, athlete_col="athlete_id")
    cc0_test_z = z.transform(cc0_test, athlete_col="athlete_id")
    
    X_train = cc0_train_z[feature_cols].copy()
    y_train = cc0_train_z["y"].to_numpy(dtype=int)
    X_test = cc0_test_z[feature_cols].copy()
    y_test = cc0_test_z["y"].to_numpy(dtype=int)
    
    # Handle missing values
    for c in feature_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0.0)
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(0.0)
    
    print(f"\n5. Training bagged XGBoost models...")
    print(f"   Label prevalence (train): {np.mean(y_train):.4f}")
    print(f"   Label prevalence (test): {np.mean(y_test):.4f}")
    
    # Check if train set has any positive labels
    if np.sum(y_train) == 0:
        print(f"\n⚠️  WARNING: Train set has no positive labels!")
        print(f"   This will cause XGBoost to fail.")
        print(f"   Trying stratified split instead...")
        from sklearn.model_selection import train_test_split
        # Use stratified split to ensure both sets have positive labels
        train_idx_new, test_idx_new = train_test_split(
            np.arange(len(cc0_df)),
            test_size=args.test_frac,
            stratify=cc0_df["y"],
            random_state=args.seed
        )
        cc0_train = cc0_df.iloc[train_idx_new].reset_index(drop=True)
        cc0_test = cc0_df.iloc[test_idx_new].reset_index(drop=True)
        
        # Re-fit z-scorer
        z = PerAthleteHealthyZScorer(min_count=args.min_healthy)
        z.fit(cc0_train, athlete_col="athlete_id", y_col="y", feature_cols=feature_cols)
        
        # Re-transform
        cc0_train_z = z.transform(cc0_train, athlete_col="athlete_id")
        cc0_test_z = z.transform(cc0_test, athlete_col="athlete_id")
        
        X_train = cc0_train_z[feature_cols].copy()
        y_train = cc0_train_z["y"].to_numpy(dtype=int)
        X_test = cc0_test_z[feature_cols].copy()
        y_test = cc0_test_z["y"].to_numpy(dtype=int)
        
        # Handle missing values
        for c in feature_cols:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0.0)
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(0.0)
        
        print(f"   After stratified split:")
        print(f"   Label prevalence (train): {np.mean(y_train):.4f}")
        print(f"   Label prevalence (test): {np.mean(y_test):.4f}")
    
    # Train bagged models
    print(f"   Training {args.n_models} models...")
    models = create_bagged_xgboost(X_train, y_train, n_models=args.n_models, seed=args.seed)
    
    # Predictions
    print(f"\n6. Making predictions...")
    p_train = predict_bagged(models, X_train)
    p_test = predict_bagged(models, X_test)
    
    # Calibrate (optional, but paper may have done this)
    # We'll calibrate the average predictions
    from sklearn.calibration import CalibratedClassifierCV
    # Create a wrapper that uses bagged predictions
    class BaggedPredictor:
        def __init__(self, models):
            self.models = models
        def predict_proba(self, X):
            return predict_bagged(self.models, X).reshape(-1, 1)
    
    # Simple calibration using isotonic regression on test set
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(p_train, y_train)
    p_train_cal = calibrator.predict(p_train)
    p_test_cal = calibrator.predict(p_test)
    
    # Metrics
    print(f"\n7. Model Performance:")
    train_auc = roc_auc_score(y_train, p_train_cal)
    test_auc = roc_auc_score(y_test, p_test_cal)
    train_pr = average_precision_score(y_train, p_train_cal)
    test_pr = average_precision_score(y_test, p_test_cal)
    test_brier = brier_score_loss(y_test, p_test_cal)
    test_logloss = log_loss(y_test, p_test_cal)
    overfitting_gap = train_auc - test_auc
    
    print(f"   Train AUC: {train_auc:.4f}")
    print(f"   Test AUC: {test_auc:.4f}")
    print(f"   Overfitting gap: {overfitting_gap:.4f}")
    print(f"   Train PR-AUC: {train_pr:.4f}")
    print(f"   Test PR-AUC: {test_pr:.4f}")
    print(f"   Test Brier: {test_brier:.4f}")
    print(f"   Test Log Loss: {test_logloss:.4f}")
    
    print(f"\n📊 COMPARISON TO PAPER:")
    print(f"   Paper (day approach): AUC = 0.724")
    print(f"   Our result: AUC = {test_auc:.4f}")
    print(f"   Difference: {test_auc - 0.724:+.4f}")
    
    if test_auc > 0.70:
        print(f"   ✅ Excellent! Close to paper's performance")
    elif test_auc > 0.60:
        print(f"   ⚠️  Good, but below paper's performance")
    else:
        print(f"   ⚠️  Below paper's performance - may need adjustments")
    
    # Save predictions
    pred_df = pd.DataFrame({
        "row_idx": cc0_test["row_idx"].values,
        "y": y_test,
        "p": p_test_cal,
    })
    pred_df.to_csv(out_dir / "pred_test_paper_method.csv", index=False)
    
    # Save metadata
    meta = {
        "approach": args.approach,
        "n_models": args.n_models,
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_prevalence": float(np.mean(y_train)),
        "test_prevalence": float(np.mean(y_test)),
        "metrics": {
            "train_auc": float(train_auc),
            "test_auc": float(test_auc),
            "overfitting_gap": float(overfitting_gap),
            "train_pr_auc": float(train_pr),
            "test_pr_auc": float(test_pr),
            "test_brier": float(test_brier),
            "test_logloss": float(test_logloss),
        },
        "paper_comparison": {
            "paper_auc": 0.724 if args.approach == "day" else 0.678,
            "our_auc": float(test_auc),
            "difference": float(test_auc - (0.724 if args.approach == "day" else 0.678)),
        },
        "feature_cols": feature_cols,
    }
    
    with open(out_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    # Save models
    import joblib
    joblib.dump(models, out_dir / "bagged_models.joblib")
    joblib.dump(calibrator, out_dir / "calibrator.joblib")
    
    print(f"\n✅ Training complete!")
    print(f"   Models saved to: {out_dir / 'bagged_models.joblib'}")
    print(f"   Predictions saved to: {out_dir / 'pred_test_paper_method.csv'}")
    print(f"   Test AUC: {test_auc:.4f} (Paper: 0.724)")
    
    return test_auc


if __name__ == "__main__":
    main()
