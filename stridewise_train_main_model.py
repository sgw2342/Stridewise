#!/usr/bin/env python3
"""Task B: Train the Main Model (rich-path) model on synthetic data.

This script is intentionally conservative: it outputs a handful of *practical checks*
that help you avoid training a main_model that looks good on a random split but collapses
on forward-time evaluation.

Artifacts written to --out:
  - main_model_xgb.json
  - main_model_schema.json
  - main_model_metrics.json                    (forward-time per-athlete)
  - main_model_random_split_metrics.json       (random-per-athlete check)
  - main_model_diagnostics.json                (gap checks, baselines, probability spread)
  - main_model_feature_importance.csv          (gain-based)
  - label_rate_by_decile.csv                (sanity: key drivers vs label)
  - main_model_sanity.json

Usage:
  python stridewise_train_main_model.py \
    --daily ./out/daily.csv --users ./out/users.csv --activities ./out/activities.csv \
    --out ./main_model_out
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--daily", required=True, help="Path to daily.csv or daily.parquet")
    p.add_argument("--users", default=None, help="Optional users.csv/parquet")
    p.add_argument("--activities", default=None, help="Optional activities.csv/parquet")
    p.add_argument("--out", required=True, help="Output directory for main_model artifacts")
    p.add_argument("--label", default="injury_next_7d", help="Label column")

    # Split + checks
    p.add_argument("--val-frac", type=float, default=0.20, help="Per-athlete forward-time validation fraction")
    p.add_argument("--skip-random-split-check", action="store_true", help="Skip random split diagnostic model")

    # Data hygiene / leakage prevention
    p.add_argument(
        "--min-history-days",
        type=int,
        default=28,
        help="Drop the first N days per athlete (stabilises rolling features / avoids early-season ACWR artefacts).",
    )
    p.add_argument(
        "--keep-ongoing-events",
        action="store_true",
        help="By default, we drop rows where injury_ongoing==1 or illness_ongoing==1 to avoid teaching the model to detect current events.",
    )

    # Feature table options
    p.add_argument("--include-users", action="store_true", default=True, help="Merge in users profile fields (default: True, includes fitness/profile)")
    p.add_argument("--include-activity-aggs", action="store_true", help="Merge in daily activity aggregates")

    # Model selection (default: catboost - best for injury prediction)
    p.add_argument("--model", type=str, default="catboost", choices=["xgboost", "lightgbm", "catboost"], help="Model type: xgboost, lightgbm, or catboost (default: catboost)")
    
    # Training (OPTIMIZED DEFAULTS for injury prediction - Best PR-AUC from grid search)
    # ADJUSTED: Increased regularization to prevent temporal features from dominating
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rounds", type=int, default=2000, help="Max boosting rounds (optimized: 2000)")
    p.add_argument("--early-stopping", type=int, default=100, help="Early stopping rounds (optimized: 100)")
    p.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    p.add_argument("--max-depth", type=int, default=4, help="Max tree depth (optimized: 4 for best PR-AUC)")
    p.add_argument("--subsample", type=float, default=0.70, help="Row subsampling (RESTORED: 0.70 from previous best)")
    p.add_argument("--colsample", type=float, default=0.70, help="Column subsampling (RESTORED: 0.70 from previous best)")
    p.add_argument("--reg-lambda", type=float, default=5.0, help="L2 regularization (RESTORED: 5.0 from previous best)")
    p.add_argument("--min-child-weight", type=float, default=6.5, help="Minimum child weight (RESTORED: 6.5 from previous best)")
    p.add_argument("--gamma", type=float, default=0.0)

    p.add_argument(
        "--use-scale-pos-weight",
        action="store_true",
        help="Use scale_pos_weight = neg/pos (can improve ranking, but hurts probability calibration; off by default for distillation).",
    )
    
    p.add_argument(
        "--min-feature-importance",
        type=float,
        default=1.0,
        help="Minimum feature importance (gain) to keep. Features below this threshold will be removed. Set to 0 to keep all features.",
    )

    return p.parse_args()


def _read_any(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_logloss(y_true: np.ndarray, p_pred: np.ndarray, clip: float = 1e-7) -> float:
    """scikit-learn removed eps= in log_loss; clip manually."""
    from sklearn.metrics import log_loss

    p = np.asarray(p_pred, dtype=float)
    p = np.clip(p, clip, 1.0 - clip)
    return float(log_loss(np.asarray(y_true, dtype=int), p))


def _prob_summary(p: np.ndarray) -> Dict[str, float]:
    p = np.asarray(p, dtype=float)
    q = np.quantile(p, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "n": int(p.size),
        "mean": float(np.mean(p)),
        "std": float(np.std(p)),
        "min": float(q[0]),
        "p01": float(q[1]),
        "p05": float(q[2]),
        "p50": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
        "max": float(q[6]),
    }


def forward_time_split(df_id: pd.DataFrame, val_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-athlete forward-time split indices.

    For each athlete: last val_frac of rows -> val, rest -> train.
    """
    df_id = df_id.sort_values(["user_id", "date"]).reset_index(drop=True)
    is_val = np.zeros(len(df_id), dtype=bool)
    for _, g in df_id.groupby("user_id", sort=False):
        n = len(g)
        n_val = max(1, int(round(n * val_frac)))
        is_val[g.index.to_numpy()[-n_val:]] = True
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    return train_idx, val_idx


def random_split_per_user(df_id: pd.DataFrame, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random split within each athlete (diagnostic only)."""
    rng = np.random.default_rng(seed)
    df_id = df_id.reset_index(drop=True)
    is_val = np.zeros(len(df_id), dtype=bool)
    for _, g in df_id.groupby("user_id", sort=False):
        idx = g.index.to_numpy()
        n_val = max(1, int(round(len(idx) * val_frac)))
        choose = rng.choice(idx, size=n_val, replace=False)
        is_val[choose] = True
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    return train_idx, val_idx


def _xgb_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: list[str],
    args,
):
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names, missing=np.nan)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names, missing=np.nan)

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    spw = float(neg / max(pos, 1.0))

    params = {
        "objective": "binary:logistic",
        "eta": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample),
        "lambda": float(args.reg_lambda),
        "min_child_weight": float(args.min_child_weight),
        "gamma": float(args.gamma),
        # Use logloss for early stopping stability on rare-event labels.
        # (We still compute AUC/PR-AUC after training.)
        "eval_metric": ["logloss"],
        "tree_method": "hist",
        "seed": int(args.seed),
        "nthread": max(1, os.cpu_count() or 2),
        # Set base_score to the empirical prevalence so early iterations start reasonable.
        "base_score": float(np.clip(np.mean(y_train), 1e-6, 1.0 - 1e-6)),
    }

    # Optional ranking boost for rare events.
    # NOTE: This often improves AUC/PR-AUC but makes raw probabilities poorly calibrated.
    if getattr(args, "use_scale_pos_weight", False):
        params["scale_pos_weight"] = spw

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(args.max_rounds),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=int(args.early_stopping),
        verbose_eval=False,
    )
    return booster, dtrain, dval


def _lgb_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: list[str],
    args,
):
    import lightgbm as lgb

    # LightGBM Dataset
    train_data = lgb.Dataset(
        X_train, 
        label=y_train, 
        feature_name=feature_names,
        free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_val, 
        label=y_val, 
        feature_name=feature_names,
        reference=train_data,
        free_raw_data=False
    )

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    spw = float(neg / max(pos, 1.0))

    # LightGBM parameters (mapped from XGBoost equivalents)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": min(2 ** int(args.max_depth), 31),  # LightGBM uses num_leaves (cap at 31 for depth 5)
        "max_depth": int(args.max_depth),  # Also set for consistency
        "learning_rate": float(args.learning_rate),
        "feature_fraction": float(args.colsample),  # colsample_bytree in XGBoost
        "bagging_fraction": float(args.subsample),  # subsample in XGBoost
        "bagging_freq": 1,
        "lambda_l2": float(args.reg_lambda),  # reg_lambda in XGBoost
        "min_child_samples": max(1, int(args.min_child_weight)),  # min_child_weight in XGBoost
        "min_gain_to_split": float(args.gamma),  # gamma in XGBoost
        "seed": int(args.seed),
        "verbosity": -1,
        "force_row_wise": True,
        "num_threads": max(1, os.cpu_count() or 2),
    }

    if getattr(args, "use_scale_pos_weight", False):
        params["scale_pos_weight"] = spw

    callbacks = [
        lgb.early_stopping(stopping_rounds=int(args.early_stopping), verbose=False),
        lgb.log_evaluation(period=0),  # Disable verbose output
    ]

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=int(args.max_rounds),
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    
    # Return booster and datasets for compatibility
    return booster, train_data, val_data


def _cat_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: list[str],
    args,
):
    from catboost import CatBoostClassifier

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    spw = float(neg / max(pos, 1.0))

    # CatBoost parameters (mapped from XGBoost equivalents)
    params = {
        "iterations": int(args.max_rounds),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.max_depth),
        "l2_leaf_reg": float(args.reg_lambda),  # reg_lambda in XGBoost
        "min_child_samples": max(1, int(args.min_child_weight)),  # min_child_weight in XGBoost
        "subsample": float(args.subsample),
        "colsample_bylevel": float(args.colsample),  # colsample_bytree in XGBoost
        "random_seed": int(args.seed),
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "early_stopping_rounds": int(args.early_stopping),
        "verbose": False,
        "thread_count": max(1, os.cpu_count() or 2),
        "bootstrap_type": "Bernoulli",  # For subsample
    }

    if getattr(args, "use_scale_pos_weight", False):
        params["class_weights"] = [1.0, spw]

    # Identify categorical features (if any)
    cat_features = []
    for i, col in enumerate(X_train.columns):
        if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
            cat_features.append(i)

    # Train CatBoost
    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features if cat_features else None,
        use_best_model=True,
    )
    
    # Create wrapper for compatibility
    class CatWrapper:
        def __init__(self, model, X_train, y_train, X_val, y_val, max_rounds):
            self.model = model
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            # CatBoost uses best_iteration_ attribute (if use_best_model=True)
            self._best_iteration = getattr(model, 'best_iteration_', max_rounds - 1)
            self._num_boosted_rounds = getattr(model, 'tree_count_', max_rounds)
        
        def predict(self, data):
            if isinstance(data, pd.DataFrame):
                return self.model.predict_proba(data)[:, 1]
            else:
                # Handle other data types
                return self.model.predict_proba(data)[:, 1]
        
        @property
        def best_iteration(self):
            return self._best_iteration
        
        def num_boosted_rounds(self):
            return self._num_boosted_rounds
        
        def save_model(self, path):
            self.model.save_model(path)
        
        def feature_importance(self, importance_type="gain"):
            # CatBoost uses "PredictionValuesChange" as gain equivalent
            if importance_type == "gain":
                return self.model.get_feature_importance(type="PredictionValuesChange")
            else:
                return self.model.get_feature_importance(type=importance_type)
        
        def feature_name(self):
            return self.model.feature_names_
    
    wrapper = CatWrapper(model, X_train, y_train, X_val, y_val, int(args.max_rounds))
    return wrapper, X_train, X_val


def _compute_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Optional[float]]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out: Dict[str, Optional[float]] = {
        "auc": None,
        "pr_auc": None,
        "logloss": None,
        "brier": None,
    }
    if len(np.unique(y)) >= 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    out["logloss"] = _safe_logloss(y, p)
    out["brier"] = float(brier_score_loss(y, p))
    return out


def _split_integrity(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> Dict[str, object]:
    """Verify forward-time split is truly forward within each athlete."""
    d = df[["user_id", "date"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.reset_index(drop=True)
    d["is_train"] = False
    d["is_val"] = False
    d.loc[train_idx, "is_train"] = True
    d.loc[val_idx, "is_val"] = True
    violations = 0
    checked = 0
    for _, g in d.groupby("user_id", sort=False):
        tr = g[g.is_train]
        va = g[g.is_val]
        if len(tr) == 0 or len(va) == 0:
            continue
        checked += 1
        if tr["date"].max() > va["date"].min():
            violations += 1
    return {"athletes_checked": int(checked), "violations": int(violations), "ok": violations == 0}


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = _read_any(args.daily)
    users = _read_any(args.users) if args.users else None
    acts = _read_any(args.activities) if args.activities else None

    # --- Data hygiene ---
    # 1) Avoid training on days that are *inside* an ongoing event. Even though the label
    #    is onset-based, event days have strong post-event signatures (rest, low load, etc.).
    #    We drop them by default so the main_model learns pre-onset patterns.
    if not args.keep_ongoing_events:
        keep_mask = np.ones(len(daily), dtype=bool)
        if "injury_ongoing" in daily.columns:
            keep_mask &= (pd.to_numeric(daily["injury_ongoing"], errors="coerce").fillna(0).astype(int) == 0).to_numpy()
        if "illness_ongoing" in daily.columns:
            keep_mask &= (pd.to_numeric(daily["illness_ongoing"], errors="coerce").fillna(0).astype(int) == 0).to_numpy()
        daily = daily.loc[keep_mask].reset_index(drop=True)

    # Build main_model feature table
    from synthrun_gen.main_model.features import build_main_model_table, LEAKAGE_DROP_COLS
    from synthrun_gen.audit import label_rate_by_decile, per_user_forward_time_rate_baseline

    df_model, schema = build_main_model_table(
        daily=daily,
        users=users,
        activities=acts,
        label_col=args.label,
        include_users=args.include_users,
        include_activity_aggs=args.include_activity_aggs,
    )

    # 2) Drop the first N days per athlete (stabilises rolling features / avoids early-series artefacts).
    if int(args.min_history_days) > 0:
        df_model = df_model.sort_values(["user_id", "date"]).reset_index(drop=True)
        keep = df_model.groupby("user_id", sort=False).cumcount() >= int(args.min_history_days)
        df_model = df_model.loc[keep].reset_index(drop=True)

    leaked = [c for c in LEAKAGE_DROP_COLS if c in schema.feature_cols]
    if leaked:
        raise RuntimeError(f"Leakage columns present in features: {leaked}")

    # Forward-time split
    df_id = df_model[["user_id", "date", args.label]].copy()
    train_idx, val_idx = forward_time_split(df_id, args.val_frac)

    X = df_model[schema.feature_cols]
    y = pd.to_numeric(df_model[args.label], errors="coerce").fillna(0).astype(int).to_numpy()

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # Train booster
    if args.model == "lightgbm":
        try:
            booster, dtrain, dval = _lgb_train(X_train, y_train, X_val, y_val, schema.feature_cols, args)
        except ImportError:
            raise RuntimeError(
                "lightgbm is required for LightGBM training. Install with: pip install lightgbm"
            )
        except Exception as e:
            raise RuntimeError(f"LightGBM training failed: {e}") from e
    elif args.model == "catboost":
        try:
            booster, dtrain, dval = _cat_train(X_train, y_train, X_val, y_val, schema.feature_cols, args)
        except ImportError:
            raise RuntimeError(
                "catboost is required for CatBoost training. Install with: pip install catboost"
            )
        except Exception as e:
            raise RuntimeError(f"CatBoost training failed: {e}") from e
    else:  # xgboost
        try:
            booster, dtrain, dval = _xgb_train(X_train, y_train, X_val, y_val, schema.feature_cols, args)
        except ImportError:
            raise RuntimeError(
                "xgboost is required for XGBoost training. Install with: pip install xgboost"
            )
        except Exception as e:
            raise RuntimeError(f"XGBoost training failed: {e}") from e

    # Predict (handle XGBoost, LightGBM, and CatBoost)
    if args.model == "lightgbm":
        p_train = booster.predict(X_train, num_iteration=booster.best_iteration).astype(float)
        p_val = booster.predict(X_val, num_iteration=booster.best_iteration).astype(float)
    elif args.model == "catboost":
        p_train = booster.predict(X_train).astype(float)
        p_val = booster.predict(X_val).astype(float)
    else:  # xgboost
        p_train = booster.predict(dtrain).astype(float)
        p_val = booster.predict(dval).astype(float)

    m_tr = _compute_metrics(y_train, p_train)
    m_va = _compute_metrics(y_val, p_val)

    main_model_metrics = {
        "label": args.label,
        "split": "forward_time_per_user",
        "n_rows": int(len(df_model)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "prevalence_train": float(np.mean(y_train)),
        "prevalence_val": float(np.mean(y_val)),
        "auc_train": m_tr["auc"],
        "auc_val": m_va["auc"],
        "pr_auc_train": m_tr["pr_auc"],
        "pr_auc_val": m_va["pr_auc"],
        "logloss_train": m_tr["logloss"],
        "logloss_val": m_va["logloss"],
        "brier_train": m_tr["brier"],
        "brier_val": m_va["brier"],
        "best_iteration": int(
            booster.best_iteration if args.model in ["lightgbm", "catboost"] 
            else getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1)
        ),
        "num_boosted_rounds": int(
            booster.num_trees() if args.model == "lightgbm" 
            else (booster.num_boosted_rounds() if args.model == "catboost" else booster.num_boosted_rounds())
        ),
        "prob_summary": {"train": _prob_summary(p_train), "val": _prob_summary(p_val)},
    }

    # --- Practical checks ---
    integrity = _split_integrity(df_model, train_idx, val_idx)

    baseline_p = np.full_like(y_val, float(np.mean(y_train)), dtype=float)
    baseline = {
        "val_logloss": _safe_logloss(y_val, baseline_p),
        "val_brier": float(np.mean((baseline_p - y_val) ** 2)),
        "val_auc": None,
    }

    user_rate_baseline = per_user_forward_time_rate_baseline(df_model, label_col=args.label, val_frac=args.val_frac)

    # Label-driver directionality (deciles)
    driver_cols = [
        c
        for c in [
            "training_load_sum7",
            "training_load_sum28",
            "km_total_sum7",
            "km_total_sum28",
            "acwr_clipped",
            "rhr_bpm_z7_28",
            "hrv_ms_z7_28",
            "sleep_hours_z7_28",
            "stress_score_z7_28",
        ]
        if c in df_model.columns
    ]
    dec = label_rate_by_decile(df_model, label_col=args.label, cols=driver_cols, n_bins=10)
    dec.to_csv(out_dir / "label_rate_by_decile.csv", index=False)

    # Random split diagnostic model (trained separately)
    random_metrics = None
    if not args.skip_random_split_check:
        tr2, va2 = random_split_per_user(df_id, args.val_frac, seed=args.seed)
        X_tr2, y_tr2 = X.iloc[tr2], y[tr2]
        X_va2, y_va2 = X.iloc[va2], y[va2]
        if args.model == "lightgbm":
            booster2, dtr2, dva2 = _lgb_train(X_tr2, y_tr2, X_va2, y_va2, schema.feature_cols, args)
            p_tr2 = booster2.predict(X_tr2).astype(float)
            p_va2 = booster2.predict(X_va2).astype(float)
        else:
            if args.model == "lightgbm":
                booster2, dtr2, dva2 = _lgb_train(X_tr2, y_tr2, X_va2, y_va2, schema.feature_cols, args)
                p_tr2 = booster2.predict(X_tr2, num_iteration=booster2.best_iteration).astype(float)
                p_va2 = booster2.predict(X_va2, num_iteration=booster2.best_iteration).astype(float)
            elif args.model == "catboost":
                booster2, dtr2, dva2 = _cat_train(X_tr2, y_tr2, X_va2, y_va2, schema.feature_cols, args)
                p_tr2 = booster2.predict(X_tr2).astype(float)
                p_va2 = booster2.predict(X_va2).astype(float)
            else:
                booster2, dtr2, dva2 = _xgb_train(X_tr2, y_tr2, X_va2, y_va2, schema.feature_cols, args)
                p_tr2 = booster2.predict(dtr2).astype(float)
                p_va2 = booster2.predict(dva2).astype(float)
        mt2 = _compute_metrics(y_tr2, p_tr2)
        mv2 = _compute_metrics(y_va2, p_va2)
        random_metrics = {
            "label": args.label,
            "split": "random_per_user",
            "n_train": int(len(tr2)),
            "n_val": int(len(va2)),
            "best_iteration": int(
                booster2.best_iteration if args.model in ["lightgbm", "catboost"]
                else getattr(booster2, "best_iteration", booster2.num_boosted_rounds() - 1)
            ),
            "num_boosted_rounds": int(
                booster2.num_trees() if args.model == "lightgbm"
                else (booster2.num_boosted_rounds() if args.model == "catboost" else booster2.num_boosted_rounds())
            ),
            "prevalence_train": float(np.mean(y_tr2)),
            "prevalence_val": float(np.mean(y_va2)),
            "auc_train": mt2["auc"],
            "auc_val": mv2["auc"],
            "pr_auc_train": mt2["pr_auc"],
            "pr_auc_val": mv2["pr_auc"],
            "logloss_train": mt2["logloss"],
            "logloss_val": mv2["logloss"],
            "brier_train": mt2["brier"],
            "brier_val": mv2["brier"],
            "prob_summary": {"train": _prob_summary(p_tr2), "val": _prob_summary(p_va2)},
        }
        with open(out_dir / "main_model_random_split_metrics.json", "w", encoding="utf-8") as f:
            json.dump(random_metrics, f, indent=2)

    # Diagnostic summary: the three checks in one place
    gap = None
    if random_metrics is not None and main_model_metrics.get("auc_val") is not None and random_metrics.get("auc_val") is not None:
        gap = float(random_metrics["auc_val"] - main_model_metrics["auc_val"])

    diagnostics = {
        "split_integrity": integrity,
        "baseline_constant_prevalence": baseline,
        "user_rate_baseline_forward_time": user_rate_baseline,
        "probability_spread": {
            "forward_time_val": main_model_metrics["prob_summary"]["val"],
            "random_val": None if random_metrics is None else random_metrics["prob_summary"]["val"],
        },
        "random_vs_forward_auc_gap": gap,
        "notes": [
            "If random split AUC is much higher than forward-time AUC, labels may be dominated by per-user propensity (or strong time drift).",
            "If probability spread on forward-time is tiny (p95 ~ p50), the model is effectively predicting a constant rate.",
            "Use delta/z features (included by default) so the main_model can learn within-athlete deviations (more realistic for forward-time).",
        ],
    }
    with open(out_dir / "main_model_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    # Save model + schema + metrics
    if args.model == "lightgbm":
        model_path = out_dir / "main_model_lgb.txt"
        booster.save_model(str(model_path))
    elif args.model == "catboost":
        model_path = out_dir / "main_model_cat.cbm"
        booster.save_model(str(model_path))
    else:
        model_path = out_dir / "main_model_xgb.json"
        booster.save_model(str(model_path))

    with open(out_dir / "main_model_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema.to_dict(), f, indent=2)

    with open(out_dir / "main_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(main_model_metrics, f, indent=2)

    # Feature importance (gain)
    if args.model == "lightgbm":
        # LightGBM feature importance
        importance_dict = booster.feature_importance(importance_type="gain")
        feature_names = booster.feature_name()
        score = dict(zip(feature_names, importance_dict))
    elif args.model == "catboost":
        # CatBoost feature importance
        importance_dict = booster.feature_importance(importance_type="gain")
        feature_names = booster.feature_name()
        score = dict(zip(feature_names, importance_dict))
    else:
        # XGBoost feature importance
        score = booster.get_score(importance_type="gain")
    
    imp = pd.DataFrame({"feature": schema.feature_cols})
    imp["gain"] = imp["feature"].map(score).fillna(0.0).astype(float)
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(out_dir / "main_model_feature_importance.csv", index=False)
    
    # Feature selection: remove low-importance features if requested
    if args.min_feature_importance > 0:
        low_importance = imp[imp["gain"] < args.min_feature_importance]["feature"].tolist()
        if low_importance:
            print(f"\n⚠️  Removing {len(low_importance)} features with importance < {args.min_feature_importance}")
            print(f"   Remaining features: {len(schema.feature_cols) - len(low_importance)}")
            # Note: This is informational only - features are already trained
            # For actual feature selection, would need to retrain with filtered features
            with open(out_dir / "removed_features.json", "w", encoding="utf-8") as f:
                json.dump({"removed_count": len(low_importance), "removed_features": low_importance[:20]}, f, indent=2)

    # Main Model sanity file
    sanity = {
        "ok": True,
        "n_features": int(len(schema.feature_cols)),
        "n_rows": int(len(df_model)),
        "label": args.label,
        "nan_rate_max": float(X.isna().mean().max()),
        "nan_rate_mean": float(X.isna().mean().mean()),
        "forward_time_auc_val": main_model_metrics.get("auc_val"),
        "random_auc_val": None if random_metrics is None else random_metrics.get("auc_val"),
    }
    with open(out_dir / "main_model_sanity.json", "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)

    print("✅ Main Model training complete")
    print(f"Model:   {model_path}")
    print(f"Schema:  {out_dir / 'main_model_schema.json'}")
    print(f"Metrics: {out_dir / 'main_model_metrics.json'}")
    if random_metrics is not None:
        print(f"Random split metrics: {out_dir / 'main_model_random_split_metrics.json'}")
    print(f"Diagnostics: {out_dir / 'main_model_diagnostics.json'}")
    print(f"Deciles: {out_dir / 'label_rate_by_decile.csv'}")


if __name__ == "__main__":
    main()
