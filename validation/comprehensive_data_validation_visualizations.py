#!/usr/bin/env python3
"""Comprehensive visualization suite for comparing synthetic vs real CC0 data.

This script creates visualizations to validate:
1. Marginal distributions (per-feature histograms/quantiles)
2. Joint structure (correlations, copulas, conditional distributions)
3. Time-series properties (autocorrelation, seasonality, persistence, lag relationships)
4. Known physiological couplings (e.g., load ↑ → RHR ↑ and HRV ↓ with plausible lag)

Usage:
  python comprehensive_data_validation_visualizations.py \
    --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
    --synth-cc0 ./synth_cc0/day_approach_maskedID_timeseries.csv \
    --real-daily ./real_daily.csv \
    --synth-daily ./synth_daily.csv \
    --cc0-schema ./cc0_feature_schema.json \
    --out ./validation_visualizations
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
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_cc0_data(file_path: Path, schema_path: Optional[Path] = None):
    """Load CC0 data and return DataFrame."""
    df = pd.read_csv(file_path)
    
    # Try to identify ID column
    id_cols = ['maskedID', 'athlete_id', 'Athlete ID', 'user_id', 'id']
    id_col = None
    for col in id_cols:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError(f"Could not find ID column in {file_path}")
    
    # Try to identify date/day column
    date_cols = ['day', 'date', 'date_index', 'row_idx']
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    return df, id_col, date_col


def load_daily_data(file_path: Path):
    """Load daily data and return DataFrame."""
    df = pd.read_csv(file_path)
    
    # Ensure date_index or day column exists
    if 'date_index' not in df.columns and 'day' in df.columns:
        df['date_index'] = df['day']
    
    return df


def plot_marginal_distributions(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    max_features: int = 20
):
    """Plot marginal distributions (histograms and quantiles) for key features."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Select top features by variance or importance
    if len(feature_cols) > max_features:
        # Select features with highest variance
        variances = []
        for feat in feature_cols:
            if feat in real_df.columns and feat in synth_df.columns:
                try:
                    real_vals = pd.to_numeric(real_df[feat], errors='coerce').dropna()
                    if len(real_vals) > 0:
                        variances.append((feat, real_vals.var()))
                except:
                    pass
        variances.sort(key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in variances[:max_features]]
    else:
        selected_features = [f for f in feature_cols if f in real_df.columns and f in synth_df.columns]
    
    # Create subplots
    n_features = len(selected_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, feat in enumerate(selected_features):
        ax = axes[idx]
        
        try:
            real_vals = pd.to_numeric(real_df[feat], errors='coerce').dropna()
            synth_vals = pd.to_numeric(synth_df[feat], errors='coerce').dropna()
            
            if len(real_vals) == 0 or len(synth_vals) == 0:
                ax.text(0.5, 0.5, f'{feat}\nNo data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Plot histograms
            ax.hist(real_vals, bins=50, alpha=0.6, label='Real', density=True, color='#2E86AB')
            ax.hist(synth_vals, bins=50, alpha=0.6, label='Synthetic', density=True, color='#A23B72')
            
            # Add quantile lines
            for q in [0.25, 0.5, 0.75]:
                real_q = real_vals.quantile(q)
                synth_q = synth_vals.quantile(q)
                ax.axvline(real_q, color='#2E86AB', linestyle='--', alpha=0.7, linewidth=1)
                ax.axvline(synth_q, color='#A23B72', linestyle='--', alpha=0.7, linewidth=1)
            
            ax.set_title(f'{feat}\nReal: μ={real_vals.mean():.2f}, σ={real_vals.std():.2f}\n'
                        f'Synth: μ={synth_vals.mean():.2f}, σ={synth_vals.std():.2f}', 
                        fontsize=9)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{feat}\nError: {str(e)[:30]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'marginal_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved marginal distributions: {out_dir / 'marginal_distributions.png'}")


def plot_quantile_comparison(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    max_features: int = 15
):
    """Plot quantile-quantile (Q-Q) plots for feature comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Select features
    if len(feature_cols) > max_features:
        selected_features = feature_cols[:max_features]
    else:
        selected_features = [f for f in feature_cols if f in real_df.columns and f in synth_df.columns]
    
    n_features = len(selected_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, feat in enumerate(selected_features):
        ax = axes[idx]
        
        try:
            real_vals = pd.to_numeric(real_df[feat], errors='coerce').dropna()
            synth_vals = pd.to_numeric(synth_df[feat], errors='coerce').dropna()
            
            if len(real_vals) == 0 or len(synth_vals) == 0:
                ax.text(0.5, 0.5, f'{feat}\nNo data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Sample for performance if too large
            if len(real_vals) > 10000:
                real_vals = real_vals.sample(10000, random_state=42)
            if len(synth_vals) > 10000:
                synth_vals = synth_vals.sample(10000, random_state=42)
            
            # Q-Q plot
            quantiles = np.linspace(0.01, 0.99, 100)
            real_quantiles = np.quantile(real_vals, quantiles)
            synth_quantiles = np.quantile(synth_vals, quantiles)
            
            ax.scatter(real_quantiles, synth_quantiles, alpha=0.6, s=20)
            
            # Add diagonal line
            min_val = min(real_quantiles.min(), synth_quantiles.min())
            max_val = max(real_quantiles.max(), synth_quantiles.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2, label='Perfect match')
            
            ax.set_xlabel('Real Quantiles')
            ax.set_ylabel('Synthetic Quantiles')
            ax.set_title(f'{feat}', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{feat}\nError', ha='center', va='center', transform=ax.transAxes)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'quantile_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved quantile comparison: {out_dir / 'quantile_comparison.png'}")


def plot_correlation_matrices(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    max_features: int = 30
):
    """Plot correlation matrices for real vs synthetic data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Select numeric features
    selected_features = []
    for feat in feature_cols[:max_features]:
        if feat in real_df.columns and feat in synth_df.columns:
            try:
                real_vals = pd.to_numeric(real_df[feat], errors='coerce')
                synth_vals = pd.to_numeric(synth_df[feat], errors='coerce')
                if real_vals.notna().sum() > 100 and synth_vals.notna().sum() > 100:
                    selected_features.append(feat)
            except:
                pass
    
    if len(selected_features) == 0:
        print("⚠️  No valid features for correlation analysis")
        return
    
    # Calculate correlations
    real_numeric = real_df[selected_features].apply(pd.to_numeric, errors='coerce')
    synth_numeric = synth_df[selected_features].apply(pd.to_numeric, errors='coerce')
    
    real_corr = real_numeric.corr()
    synth_corr = synth_numeric.corr()
    
    # Plot side by side
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # Real correlation
    sns.heatmap(real_corr, ax=axes[0], cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={'label': 'Correlation'}, fmt='.2f')
    axes[0].set_title('Real Data Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Synthetic correlation
    sns.heatmap(synth_corr, ax=axes[1], cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={'label': 'Correlation'}, fmt='.2f')
    axes[1].set_title('Synthetic Data Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Difference
    corr_diff = synth_corr - real_corr
    sns.heatmap(corr_diff, ax=axes[2], cmap='RdBu_r', center=0,
                square=True, cbar_kws={'label': 'Difference'}, fmt='.2f')
    axes[2].set_title('Correlation Difference (Synthetic - Real)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'correlation_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved correlation matrices: {out_dir / 'correlation_matrices.png'}")
    
    # Save correlation difference statistics
    corr_diff_stats = {
        'mean_absolute_difference': float(np.abs(corr_diff.values).mean()),
        'max_absolute_difference': float(np.abs(corr_diff.values).max()),
        'features_with_large_diff': []
    }
    
    # Find features with large differences
    threshold = 0.2
    for i, feat1 in enumerate(selected_features):
        for j, feat2 in enumerate(selected_features):
            if i < j and abs(corr_diff.loc[feat1, feat2]) > threshold:
                corr_diff_stats['features_with_large_diff'].append({
                    'feature1': feat1,
                    'feature2': feat2,
                    'real_corr': float(real_corr.loc[feat1, feat2]),
                    'synth_corr': float(synth_corr.loc[feat1, feat2]),
                    'difference': float(corr_diff.loc[feat1, feat2])
                })
    
    with open(out_dir / 'correlation_difference_stats.json', 'w') as f:
        json.dump(corr_diff_stats, f, indent=2)


def plot_conditional_distributions(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_pairs: list[tuple[str, str]],
    out_dir: Path
):
    """Plot conditional distributions (scatter plots with conditional means)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_pairs = len(feature_pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (feat_x, feat_y) in enumerate(feature_pairs):
        ax = axes[idx]
        
        try:
            if feat_x not in real_df.columns or feat_y not in real_df.columns:
                ax.text(0.5, 0.5, f'{feat_x} vs {feat_y}\nMissing columns', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            real_x = pd.to_numeric(real_df[feat_x], errors='coerce').dropna()
            real_y = pd.to_numeric(real_df[feat_y], errors='coerce').dropna()
            synth_x = pd.to_numeric(synth_df[feat_x], errors='coerce').dropna()
            synth_y = pd.to_numeric(synth_df[feat_y], errors='coerce').dropna()
            
            # Align indices
            real_common = real_x.index.intersection(real_y.index)
            synth_common = synth_x.index.intersection(synth_y.index)
            
            if len(real_common) == 0 or len(synth_common) == 0:
                ax.text(0.5, 0.5, f'{feat_x} vs {feat_y}\nNo overlapping data', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            real_x_aligned = real_x.loc[real_common]
            real_y_aligned = real_y.loc[real_common]
            synth_x_aligned = synth_x.loc[synth_common]
            synth_y_aligned = synth_y.loc[synth_common]
            
            # Sample if too large
            if len(real_x_aligned) > 5000:
                sample_idx = np.random.choice(len(real_x_aligned), 5000, replace=False)
                real_x_aligned = real_x_aligned.iloc[sample_idx]
                real_y_aligned = real_y_aligned.iloc[sample_idx]
            if len(synth_x_aligned) > 5000:
                sample_idx = np.random.choice(len(synth_x_aligned), 5000, replace=False)
                synth_x_aligned = synth_x_aligned.iloc[sample_idx]
                synth_y_aligned = synth_y_aligned.iloc[sample_idx]
            
            # Scatter plots
            ax.scatter(real_x_aligned, real_y_aligned, alpha=0.3, s=10, 
                      label='Real', color='#2E86AB')
            ax.scatter(synth_x_aligned, synth_y_aligned, alpha=0.3, s=10, 
                      label='Synthetic', color='#A23B72', marker='x')
            
            # Add conditional means (binned)
            bins = np.linspace(min(real_x_aligned.min(), synth_x_aligned.min()),
                             max(real_x_aligned.max(), synth_x_aligned.max()), 20)
            real_bin_means = []
            real_bin_centers = []
            synth_bin_means = []
            synth_bin_centers = []
            
            for i in range(len(bins) - 1):
                real_mask = (real_x_aligned >= bins[i]) & (real_x_aligned < bins[i+1])
                synth_mask = (synth_x_aligned >= bins[i]) & (synth_x_aligned < bins[i+1])
                
                if real_mask.sum() > 0:
                    real_bin_means.append(real_y_aligned[real_mask].mean())
                    real_bin_centers.append((bins[i] + bins[i+1]) / 2)
                
                if synth_mask.sum() > 0:
                    synth_bin_means.append(synth_y_aligned[synth_mask].mean())
                    synth_bin_centers.append((bins[i] + bins[i+1]) / 2)
            
            if len(real_bin_centers) > 0:
                ax.plot(real_bin_centers, real_bin_means, 'o-', color='#2E86AB', 
                       linewidth=2, markersize=6, label='Real mean')
            if len(synth_bin_centers) > 0:
                ax.plot(synth_bin_centers, synth_bin_means, 's--', color='#A23B72', 
                       linewidth=2, markersize=6, label='Synthetic mean')
            
            # Calculate correlation
            real_corr, _ = pearsonr(real_x_aligned, real_y_aligned)
            synth_corr, _ = pearsonr(synth_x_aligned, synth_y_aligned)
            
            ax.set_xlabel(feat_x)
            ax.set_ylabel(feat_y)
            ax.set_title(f'{feat_x} vs {feat_y}\n'
                        f'Real r={real_corr:.3f}, Synth r={synth_corr:.3f}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'{feat_x} vs {feat_y}\nError: {str(e)[:40]}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'conditional_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved conditional distributions: {out_dir / 'conditional_distributions.png'}")


def plot_time_series_properties(
    real_daily: pd.DataFrame,
    synth_daily: pd.DataFrame,
    id_col: str,
    date_col: str,
    features: list[str],
    out_dir: Path,
    max_users: int = 5
):
    """Plot time-series properties: autocorrelation, seasonality, persistence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique users
    real_users = real_daily[id_col].unique()[:max_users]
    synth_users = synth_daily[id_col].unique()[:max_users]
    
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 5 * n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for feat_idx, feat in enumerate(features):
        if feat not in real_daily.columns or feat not in synth_daily.columns:
            continue
        
        # Real data autocorrelation
        ax_real = axes[feat_idx, 0]
        real_acfs = []
        
        for user_id in real_users:
            user_data = real_daily[real_daily[id_col] == user_id].sort_values(date_col)
            if len(user_data) < 30:  # Need minimum data for ACF
                continue
            
            vals = pd.to_numeric(user_data[feat], errors='coerce').dropna()
            if len(vals) < 30:
                continue
            
            try:
                acf_vals = acf(vals, nlags=min(30, len(vals)-1), fft=True)
                real_acfs.append(acf_vals)
            except:
                pass
        
        if len(real_acfs) > 0:
            real_acf_mean = np.mean(real_acfs, axis=0)
            real_acf_std = np.std(real_acfs, axis=0)
            lags = np.arange(len(real_acf_mean))
            
            ax_real.plot(lags, real_acf_mean, 'o-', color='#2E86AB', label='Real mean', linewidth=2)
            ax_real.fill_between(lags, real_acf_mean - real_acf_std, 
                                real_acf_mean + real_acf_std, alpha=0.3, color='#2E86AB')
            ax_real.axhline(0, color='black', linestyle='--', linewidth=1)
            ax_real.set_xlabel('Lag')
            ax_real.set_ylabel('Autocorrelation')
            ax_real.set_title(f'Real: {feat} Autocorrelation', fontweight='bold')
            ax_real.legend()
            ax_real.grid(True, alpha=0.3)
        
        # Synthetic data autocorrelation
        ax_synth = axes[feat_idx, 1]
        synth_acfs = []
        
        for user_id in synth_users:
            user_data = synth_daily[synth_daily[id_col] == user_id].sort_values(date_col)
            if len(user_data) < 30:
                continue
            
            vals = pd.to_numeric(user_data[feat], errors='coerce').dropna()
            if len(vals) < 30:
                continue
            
            try:
                acf_vals = acf(vals, nlags=min(30, len(vals)-1), fft=True)
                synth_acfs.append(acf_vals)
            except:
                pass
        
        if len(synth_acfs) > 0:
            synth_acf_mean = np.mean(synth_acfs, axis=0)
            synth_acf_std = np.std(synth_acfs, axis=0)
            lags = np.arange(len(synth_acf_mean))
            
            ax_synth.plot(lags, synth_acf_mean, 's-', color='#A23B72', label='Synthetic mean', linewidth=2)
            ax_synth.fill_between(lags, synth_acf_mean - synth_acf_std, 
                                 synth_acf_mean + synth_acf_std, alpha=0.3, color='#A23B72')
            ax_synth.axhline(0, color='black', linestyle='--', linewidth=1)
            ax_synth.set_xlabel('Lag')
            ax_synth.set_ylabel('Autocorrelation')
            ax_synth.set_title(f'Synthetic: {feat} Autocorrelation', fontweight='bold')
            ax_synth.legend()
            ax_synth.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'time_series_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved time-series autocorrelation: {out_dir / 'time_series_autocorrelation.png'}")


def plot_physiological_couplings(
    real_daily: pd.DataFrame,
    synth_daily: pd.DataFrame,
    id_col: str,
    date_col: str,
    couplings: list[dict],
    out_dir: Path,
    max_users: int = 10
):
    """Plot known physiological couplings with lag analysis.
    
    Args:
        couplings: List of dicts with keys:
            - 'x_feature': predictor feature (e.g., 'load_trimp')
            - 'y_feature': response feature (e.g., 'resting_hr')
            - 'expected_lag': expected lag in days (e.g., 1)
            - 'expected_direction': 'positive' or 'negative'
            - 'name': human-readable name
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_couplings = len(couplings)
    fig, axes = plt.subplots(n_couplings, 2, figsize=(16, 6 * n_couplings))
    if n_couplings == 1:
        axes = axes.reshape(1, -1)
    
    for coupling_idx, coupling in enumerate(couplings):
        x_feat = coupling['x_feature']
        y_feat = coupling['y_feature']
        expected_lag = coupling.get('expected_lag', 1)
        expected_dir = coupling.get('expected_direction', 'positive')
        name = coupling.get('name', f'{x_feat} → {y_feat}')
        
        # Real data
        ax_real = axes[coupling_idx, 0]
        real_corrs = []
        real_lags = list(range(-7, 8))  # -7 to +7 days
        
        for lag in real_lags:
            user_corrs = []
            
            for user_id in real_daily[id_col].unique()[:max_users]:
                user_data = real_daily[real_daily[id_col] == user_id].sort_values(date_col)
                if len(user_data) < abs(lag) + 10:
                    continue
                
                x_vals = pd.to_numeric(user_data[x_feat], errors='coerce')
                y_vals = pd.to_numeric(user_data[y_feat], errors='coerce')
                
                if lag == 0:
                    x_aligned = x_vals
                    y_aligned = y_vals
                elif lag > 0:
                    x_aligned = x_vals.iloc[:-lag]
                    y_aligned = y_vals.iloc[lag:]
                else:  # lag < 0
                    x_aligned = x_vals.iloc[-lag:]
                    y_aligned = y_vals.iloc[:lag]
                
                # Align indices
                common_idx = x_aligned.index.intersection(y_aligned.index)
                if len(common_idx) < 10:
                    continue
                
                x_common = x_aligned.loc[common_idx].dropna()
                y_common = y_aligned.loc[common_idx].dropna()
                common_final = x_common.index.intersection(y_common.index)
                
                if len(common_final) < 10:
                    continue
                
                try:
                    corr, _ = pearsonr(x_common.loc[common_final], y_common.loc[common_final])
                    if not np.isnan(corr):
                        user_corrs.append(corr)
                except:
                    pass
            
            if len(user_corrs) > 0:
                real_corrs.append(np.mean(user_corrs))
            else:
                real_corrs.append(0)
        
        ax_real.plot(real_lags, real_corrs, 'o-', color='#2E86AB', linewidth=2, markersize=8)
        ax_real.axvline(expected_lag, color='red', linestyle='--', linewidth=2, label=f'Expected lag: {expected_lag}')
        ax_real.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax_real.set_xlabel('Lag (days)')
        ax_real.set_ylabel('Correlation')
        ax_real.set_title(f'Real: {name}', fontweight='bold')
        ax_real.legend()
        ax_real.grid(True, alpha=0.3)
        
        # Synthetic data
        ax_synth = axes[coupling_idx, 1]
        synth_corrs = []
        
        for lag in real_lags:
            user_corrs = []
            
            for user_id in synth_daily[id_col].unique()[:max_users]:
                user_data = synth_daily[synth_daily[id_col] == user_id].sort_values(date_col)
                if len(user_data) < abs(lag) + 10:
                    continue
                
                x_vals = pd.to_numeric(user_data[x_feat], errors='coerce')
                y_vals = pd.to_numeric(user_data[y_feat], errors='coerce')
                
                if lag == 0:
                    x_aligned = x_vals
                    y_aligned = y_vals
                elif lag > 0:
                    x_aligned = x_vals.iloc[:-lag]
                    y_aligned = y_vals.iloc[lag:]
                else:
                    x_aligned = x_vals.iloc[-lag:]
                    y_aligned = y_vals.iloc[:lag]
                
                common_idx = x_aligned.index.intersection(y_aligned.index)
                if len(common_idx) < 10:
                    continue
                
                x_common = x_aligned.loc[common_idx].dropna()
                y_common = y_aligned.loc[common_idx].dropna()
                common_final = x_common.index.intersection(y_common.index)
                
                if len(common_final) < 10:
                    continue
                
                try:
                    corr, _ = pearsonr(x_common.loc[common_final], y_common.loc[common_final])
                    if not np.isnan(corr):
                        user_corrs.append(corr)
                except:
                    pass
            
            if len(user_corrs) > 0:
                synth_corrs.append(np.mean(user_corrs))
            else:
                synth_corrs.append(0)
        
        ax_synth.plot(real_lags, synth_corrs, 's-', color='#A23B72', linewidth=2, markersize=8)
        ax_synth.axvline(expected_lag, color='red', linestyle='--', linewidth=2, label=f'Expected lag: {expected_lag}')
        ax_synth.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax_synth.set_xlabel('Lag (days)')
        ax_synth.set_ylabel('Correlation')
        ax_synth.set_title(f'Synthetic: {name}', fontweight='bold')
        ax_synth.legend()
        ax_synth.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'physiological_couplings.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved physiological couplings: {out_dir / 'physiological_couplings.png'}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive data validation visualizations')
    parser.add_argument('--real-cc0', type=str, required=True, help='Real CC0 data file')
    parser.add_argument('--synth-cc0', type=str, required=True, help='Synthetic CC0 data file')
    parser.add_argument('--real-daily', type=str, help='Real daily data file (optional)')
    parser.add_argument('--synth-daily', type=str, help='Synthetic daily data file (optional)')
    parser.add_argument('--cc0-schema', type=str, help='CC0 schema file (optional)')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE DATA VALIDATION VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Load CC0 data
    print("1. Loading CC0 data...")
    real_cc0, real_id_col, real_date_col = load_cc0_data(Path(args.real_cc0))
    synth_cc0, synth_id_col, synth_date_col = load_cc0_data(Path(args.synth_cc0))
    
    # Get numeric feature columns (exclude ID, date, label columns)
    exclude_cols = [real_id_col, synth_id_col, real_date_col, synth_date_col, 
                   'injury', 'injury_next_7d', 'y', 'label']
    feature_cols = [c for c in real_cc0.columns 
                   if c not in exclude_cols and pd.api.types.is_numeric_dtype(real_cc0[c])]
    
    print(f"   Real CC0: {len(real_cc0):,} rows, {len(feature_cols)} numeric features")
    print(f"   Synthetic CC0: {len(synth_cc0):,} rows, {len(feature_cols)} numeric features")
    print()
    
    # 1. Marginal distributions
    print("2. Creating marginal distribution plots...")
    plot_marginal_distributions(real_cc0, synth_cc0, feature_cols, out_dir)
    print()
    
    # 2. Quantile comparison
    print("3. Creating quantile comparison plots...")
    plot_quantile_comparison(real_cc0, synth_cc0, feature_cols, out_dir)
    print()
    
    # 3. Correlation matrices
    print("4. Creating correlation matrices...")
    plot_correlation_matrices(real_cc0, synth_cc0, feature_cols, out_dir)
    print()
    
    # 4. Conditional distributions (key feature pairs)
    print("5. Creating conditional distribution plots...")
    # Select important feature pairs based on common physiological relationships
    key_pairs = []
    for feat1 in ['total_km', 'total_kms_week', 'km_total_mean7', 'km_total_mean28']:
        for feat2 in ['resting_hr', 'lnrmssd', 'hrv_z', 'rhr_z']:
            if feat1 in feature_cols and feat2 in feature_cols:
                key_pairs.append((feat1, feat2))
                if len(key_pairs) >= 8:
                    break
        if len(key_pairs) >= 8:
            break
    
    if len(key_pairs) > 0:
        plot_conditional_distributions(real_cc0, synth_cc0, key_pairs, out_dir)
    print()
    
    # 5. Time-series properties (if daily data available)
    if args.real_daily and args.synth_daily:
        print("6. Creating time-series property plots...")
        real_daily = load_daily_data(Path(args.real_daily))
        synth_daily = load_daily_data(Path(args.synth_daily))
        
        # Find ID and date columns
        daily_id_col = 'user_id' if 'user_id' in real_daily.columns else 'maskedID'
        daily_date_col = 'date_index' if 'date_index' in real_daily.columns else 'day'
        
        ts_features = ['load_trimp', 'resting_hr', 'lnrmssd', 'sleep_duration_h', 'stress_score_0_100']
        ts_features = [f for f in ts_features if f in real_daily.columns and f in synth_daily.columns]
        
        if len(ts_features) > 0:
            plot_time_series_properties(real_daily, synth_daily, daily_id_col, daily_date_col, 
                                      ts_features, out_dir)
        print()
        
        # 6. Physiological couplings
        print("7. Creating physiological coupling plots...")
        couplings = [
            {
                'x_feature': 'load_trimp',
                'y_feature': 'resting_hr',
                'expected_lag': 1,
                'expected_direction': 'positive',
                'name': 'Load → RHR (next day)'
            },
            {
                'x_feature': 'load_trimp',
                'y_feature': 'lnrmssd',
                'expected_lag': 1,
                'expected_direction': 'negative',
                'name': 'Load → HRV (next day)'
            },
            {
                'x_feature': 'acute_load_7d',
                'y_feature': 'resting_hr',
                'expected_lag': 0,
                'expected_direction': 'positive',
                'name': 'Acute Load → RHR (same day)'
            },
            {
                'x_feature': 'sleep_duration_h',
                'y_feature': 'lnrmssd',
                'expected_lag': 0,
                'expected_direction': 'positive',
                'name': 'Sleep → HRV (same day)'
            }
        ]
        
        # Filter to available features
        available_couplings = []
        for coupling in couplings:
            if (coupling['x_feature'] in real_daily.columns and 
                coupling['y_feature'] in real_daily.columns and
                coupling['x_feature'] in synth_daily.columns and 
                coupling['y_feature'] in synth_daily.columns):
                available_couplings.append(coupling)
        
        if len(available_couplings) > 0:
            plot_physiological_couplings(real_daily, synth_daily, daily_id_col, daily_date_col,
                                       available_couplings, out_dir)
        print()
    
    print("=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print()


if __name__ == '__main__':
    main()
