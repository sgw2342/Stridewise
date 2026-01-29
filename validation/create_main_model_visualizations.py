#!/usr/bin/env python3
"""Create comprehensive performance visualizations for the main model.

This script generates visualizations demonstrating main model performance:
- ROC curves
- PR curves
- Calibration curves
- Feature importance
- Prediction distributions
- Performance by threshold
- Performance by cohort/profile

Usage:
  python validation/create_main_model_visualizations.py \
    --model-dir ./models/main_model_large_dataset \
    --predictions ./eval_results/predictions.csv \
    --out ./main_model_visualizations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)
try:
    from synthrun_gen.eval.metrics import (
        calibration_bins,
        expected_calibration_error,
    )
except ImportError:
    # Fallback implementation if synthrun_gen not available
    def calibration_bins(y, p, n_bins=10):
        """Return (bin_edges, bin_count, bin_mean_p, bin_frac_pos)."""
        import numpy as np
        y = np.asarray(y, dtype=int)
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 0.0, 1.0)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.digitize(p, edges[1:-1], right=False)
        counts = np.zeros(n_bins, dtype=int)
        mean_p = np.full(n_bins, np.nan, dtype=float)
        frac_pos = np.full(n_bins, np.nan, dtype=float)
        for b in range(n_bins):
            m = (bin_idx == b)
            counts[b] = int(m.sum())
            if counts[b] > 0:
                mean_p[b] = float(np.mean(p[m]))
                frac_pos[b] = float(np.mean(y[m]))
        return edges, counts, mean_p, frac_pos
    
    def expected_calibration_error(counts, mean_p, frac_pos):
        """Calculate Expected Calibration Error."""
        import numpy as np
        counts = np.asarray(counts, dtype=float)
        if counts.sum() <= 0:
            return None
        w = counts / counts.sum()
        d = np.abs(np.nan_to_num(frac_pos, nan=0.0) - np.nan_to_num(mean_p, nan=0.0))
        return float(np.sum(w * d))

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


def load_metrics(metrics_file: Path) -> dict:
    """Load metrics from JSON file."""
    with open(metrics_file) as f:
        return json.load(f)


def load_predictions(predictions_file: Path) -> pd.DataFrame:
    """Load predictions from CSV file."""
    return pd.read_csv(predictions_file)


def load_feature_importance(importance_file: Path) -> pd.DataFrame:
    """Load feature importance from CSV file."""
    if importance_file.exists():
        return pd.read_csv(importance_file)
    return None


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, out_dir: Path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    
    # Add validation AUC from metrics if available
    if 'auc_val' in metrics:
        ax.text(0.6, 0.2, f'Validation AUC: {metrics["auc_val"]:.4f}', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Main Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_dir / 'roc_curve.png'}")


def plot_pr_curve(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, out_dir: Path):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    # Baseline (prevalence)
    prevalence = np.mean(y_true)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, color='#A23B72', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    ax.axhline(prevalence, color='gray', lw=1, linestyle='--', 
               label=f'Baseline (Prevalence = {prevalence:.2%})')
    
    # Add validation PR-AUC from metrics if available
    if 'pr_auc_val' in metrics:
        ax.text(0.6, 0.2, f'Validation PR-AUC: {metrics["pr_auc_val"]:.4f}', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - Main Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_dir / 'pr_curve.png'}")


def plot_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, out_dir: Path):
    """Plot calibration curve."""
    edges, counts, mean_p, frac_pos = calibration_bins(y_true, y_pred, n_bins=10)
    ece = expected_calibration_error(counts, mean_p, frac_pos)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    
    # Model calibration
    valid_bins = ~np.isnan(mean_p) & ~np.isnan(frac_pos)
    if valid_bins.sum() > 0:
        ax.plot(mean_p[valid_bins], frac_pos[valid_bins], 'o-', 
               color='#2E86AB', linewidth=2, markersize=10, label='Model')
        
        # Add bin counts as text
        for i in range(len(mean_p)):
            if valid_bins[i] and counts[i] > 0:
                ax.text(mean_p[i], frac_pos[i], f'n={counts[i]}', 
                       fontsize=9, ha='center', va='bottom')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Calibration Curve - Main Model\nECE = {ece:.4f}' if ece else 'Calibration Curve - Main Model', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'calibration_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_dir / 'calibration_curve.png'}")


def plot_feature_importance(importance_df: pd.DataFrame, out_dir: Path, top_n: int = 20):
    """Plot feature importance."""
    if importance_df is None:
        print("⚠️  Feature importance file not found, skipping...")
        return
    
    # Find importance column (could be 'importance', 'gain', 'feature_importance', etc.)
    importance_col = None
    for col in ['importance', 'gain', 'feature_importance', 'score']:
        if col in importance_df.columns:
            importance_col = col
            break
    
    if importance_col is None:
        print("⚠️  Could not find importance column, skipping...")
        return
    
    # Find feature name column
    feature_col = None
    for col in ['feature', 'feature_name', 'name', 'Feature']:
        if col in importance_df.columns:
            feature_col = col
            break
    
    if feature_col is None:
        # Use first column as feature name
        feature_col = importance_df.columns[0]
    
    # Get top N features
    top_features = importance_df.nlargest(top_n, importance_col)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(top_features))
    
    bars = ax.barh(y_pos, top_features[importance_col].values, color='#2E86AB')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[feature_col].values, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(f'Importance ({importance_col})', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - Main Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row[importance_col], i, f' {row[importance_col]:.2f}', 
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_dir / 'feature_importance.png'}")


def plot_prediction_distributions(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, out_dir: Path):
    """Plot prediction distributions for positive and negative classes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Separate predictions by class
    pred_pos = y_pred[y_true == 1]
    pred_neg = y_pred[y_true == 0]
    
    # Histogram
    axes[0].hist(pred_neg, bins=50, alpha=0.6, label='Negative Class', color='#2E86AB', density=True)
    axes[0].hist(pred_pos, bins=50, alpha=0.6, label='Positive Class', color='#A23B72', density=True)
    axes[0].axvline(np.mean(pred_neg), color='#2E86AB', linestyle='--', linewidth=2, label=f'Neg Mean: {np.mean(pred_neg):.3f}')
    axes[0].axvline(np.mean(pred_pos), color='#A23B72', linestyle='--', linewidth=2, label=f'Pos Mean: {np.mean(pred_pos):.3f}')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Prediction Distribution by Class', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    box_data = [pred_neg, pred_pos]
    bp = axes[1].boxplot(box_data, labels=['Negative', 'Positive'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')
    axes[1].set_ylabel('Predicted Probability', fontsize=12)
    axes[1].set_title('Prediction Distribution (Box Plot)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'prediction_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_dir / 'prediction_distributions.png'}")


def plot_threshold_analysis(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    """Plot performance metrics across different thresholds."""
    thresholds = np.linspace(0.01, 0.99, 99)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    specificity_scores = []
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        specificity_scores.append(specificity)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Precision, Recall, F1
    axes[0].plot(thresholds, precision_scores, label='Precision', linewidth=2, color='#2E86AB')
    axes[0].plot(thresholds, recall_scores, label='Recall', linewidth=2, color='#A23B72')
    axes[0].plot(thresholds, f1_scores, label='F1 Score', linewidth=2, color='#F18F01')
    axes[0].set_xlabel('Threshold', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Precision, Recall, and F1 Score vs Threshold', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Find optimal F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    axes[0].axvline(best_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Best F1: {best_threshold:.3f}')
    axes[0].legend(fontsize=11)
    
    # Specificity
    axes[1].plot(thresholds, specificity_scores, label='Specificity', linewidth=2, color='#6A994E')
    axes[1].set_xlabel('Threshold', fontsize=12)
    axes[1].set_ylabel('Specificity', fontsize=12)
    axes[1].set_title('Specificity vs Threshold', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(best_threshold, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_dir / 'threshold_analysis.png'}")
    
    return best_threshold


def plot_performance_summary(metrics: dict, out_dir: Path):
    """Create a comprehensive performance summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Key Metrics Bar Chart
    ax = axes[0, 0]
    metric_names = ['ROC AUC', 'PR AUC', 'Brier Score']
    train_values = [
        metrics.get('auc_train', 0),
        metrics.get('pr_auc_train', 0),
        metrics.get('brier_train', 0)
    ]
    val_values = [
        metrics.get('auc_val', 0),
        metrics.get('pr_auc_val', 0),
        metrics.get('brier_val', 0)
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    ax.bar(x - width/2, train_values, width, label='Train', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, val_values, width, label='Validation', color='#A23B72', alpha=0.8)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Metrics: Train vs Validation', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (train, val) in enumerate(zip(train_values, val_values)):
        ax.text(i - width/2, train, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Overfitting Check
    ax = axes[0, 1]
    auc_gap = metrics.get('auc_train', 0) - metrics.get('auc_val', 0)
    pr_gap = metrics.get('pr_auc_train', 0) - metrics.get('pr_auc_val', 0)
    
    gaps = [auc_gap, pr_gap]
    gap_names = ['AUC Gap', 'PR-AUC Gap']
    colors = ['#6A994E' if abs(g) < 0.05 else '#F18F01' for g in gaps]
    
    ax.bar(gap_names, gaps, color=colors, alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1, label='Warning (0.05)')
    ax.axhline(-0.05, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('Gap (Train - Val)', fontsize=12)
    ax.set_title('Overfitting Check', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, gap in enumerate(gaps):
        ax.text(i, gap, f'{gap:+.3f}', ha='center', 
               va='bottom' if gap > 0 else 'top', fontsize=10, fontweight='bold')
    
    # 3. Dataset Statistics
    ax = axes[1, 0]
    stats = {
        'Total Rows': metrics.get('n_rows', 0),
        'Train Rows': metrics.get('n_train', 0),
        'Val Rows': metrics.get('n_val', 0),
    }
    ax.bar(stats.keys(), stats.values(), color='#2E86AB', alpha=0.8)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Dataset Statistics', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for k, v in stats.items():
        ax.text(k, v, f'{v:,}', ha='center', va='bottom', fontsize=10)
    
    # 4. Prevalence
    ax = axes[1, 1]
    train_prev = metrics.get('prevalence_train', 0)
    val_prev = metrics.get('prevalence_val', 0)
    
    ax.bar(['Train', 'Validation'], [train_prev, val_prev], 
           color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax.set_ylabel('Prevalence', fontsize=12)
    ax.set_title('Label Prevalence', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    ax.text(0, train_prev, f'{train_prev:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(1, val_prev, f'{val_prev:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Main Model Performance Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_dir / 'performance_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_dir / 'performance_summary.png'}")


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, thresholds: list = None):
    """Plot confusion matrices at different thresholds."""
    if thresholds is None:
        # Default thresholds: optimal F1, 0.1, 0.3, 0.5
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
        thresholds = [0.1, best_threshold, 0.3, 0.5]
    
    n_thresholds = len(thresholds)
    n_cols = 2
    n_rows = (n_thresholds + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]
        y_pred_binary = (y_pred >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'Confusion Matrix (Threshold = {threshold:.3f})', fontsize=12, fontweight='bold')
        
        # Add labels
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
    
    # Hide unused subplots
    for idx in range(n_thresholds, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_dir / 'confusion_matrices.png'}")


def main():
    parser = argparse.ArgumentParser(description='Create comprehensive main model visualizations')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory containing trained model')
    parser.add_argument('--predictions', type=str, help='Path to predictions CSV (optional, will use from model dir if not provided)')
    parser.add_argument('--out', type=str, required=True, help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MAIN MODEL PERFORMANCE VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Load metrics
    print("1. Loading metrics...")
    metrics_file = model_dir / "main_model_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    metrics = load_metrics(metrics_file)
    print(f"   ✅ Loaded metrics from {metrics_file}")
    
    # Load predictions
    print("\n2. Loading predictions...")
    if args.predictions:
        predictions_file = Path(args.predictions)
    else:
        # Try to find predictions in common locations
        predictions_file = model_dir / "predictions.csv"
        if not predictions_file.exists():
            print("   ⚠️  Predictions file not found. Some visualizations will be skipped.")
            predictions_file = None
    
    if predictions_file and predictions_file.exists():
        pred_df = load_predictions(predictions_file)
        y_true = pred_df['label'].values if 'label' in pred_df.columns else None
        y_pred = pred_df['prediction'].values if 'prediction' in pred_df.columns else None
        
        if y_true is None or y_pred is None:
            print("   ⚠️  Could not find label/prediction columns. Some visualizations will be skipped.")
            y_true = None
            y_pred = None
        else:
            print(f"   ✅ Loaded {len(pred_df):,} predictions")
    else:
        y_true = None
        y_pred = None
        print("   ⚠️  Predictions not available. Creating visualizations from metrics only.")
    
    # Load feature importance
    print("\n3. Loading feature importance...")
    importance_file = model_dir / "main_model_feature_importance.csv"
    importance_df = load_feature_importance(importance_file) if importance_file.exists() else None
    if importance_df is not None:
        print(f"   ✅ Loaded {len(importance_df)} features")
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    
    # Performance summary (always available)
    plot_performance_summary(metrics, out_dir)
    
    if y_true is not None and y_pred is not None:
        # ROC curve
        plot_roc_curve(y_true, y_pred, metrics, out_dir)
        
        # PR curve
        plot_pr_curve(y_true, y_pred, metrics, out_dir)
        
        # Calibration curve
        plot_calibration_curve(y_true, y_pred, metrics, out_dir)
        
        # Prediction distributions
        plot_prediction_distributions(y_true, y_pred, metrics, out_dir)
        
        # Threshold analysis
        best_threshold = plot_threshold_analysis(y_true, y_pred, out_dir)
        print(f"   📊 Optimal F1 threshold: {best_threshold:.4f}")
        
        # Confusion matrices
        plot_confusion_matrices(y_true, y_pred, out_dir)
    else:
        print("   ⚠️  Skipping prediction-based visualizations (no predictions available)")
    
    # Feature importance
    if importance_df is not None:
        plot_feature_importance(importance_df, out_dir)
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print()


if __name__ == '__main__':
    main()
