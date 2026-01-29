#!/usr/bin/env python3
"""Analyze real CC0 data for training blocks and athlete grouping.

This script:
1. Checks if training blocks with easy weeks exist in real CC0
2. Groups athletes into short/medium vs long distance based on training patterns
3. Compares to synthetic data patterns
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.cluster import KMeans

def analyze_training_blocks(df, user_id_col='Athlete ID', date_col='Date'):
    """Analyze training blocks (easy weeks) in the data."""
    print("=" * 80)
    print("TRAINING BLOCK ANALYSIS")
    print("=" * 80)
    print()
    
    # Convert date if needed
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values([user_id_col, date_col])
    
    # Calculate weekly aggregates
    df['week'] = df[date_col].dt.isocalendar().week
    df['year'] = df[date_col].dt.year
    df['week_id'] = df['year'].astype(str) + '_' + df['week'].astype(str)
    
    # Calculate intensity metrics
    if 'km Z3-4' in df.columns and 'km Z5-T1-T2' in df.columns and 'km sprinting' in df.columns:
        df['total_intensity_km'] = (
            df['km Z3-4'].fillna(0) + 
            df['km Z5-T1-T2'].fillna(0) + 
            df['km sprinting'].fillna(0)
        )
    else:
        # Try with .0 suffix
        df['total_intensity_km'] = (
            df.get('km Z3-4.0', pd.Series(0, index=df.index)).fillna(0) + 
            df.get('km Z5-T1-T2.0', pd.Series(0, index=df.index)).fillna(0) + 
            df.get('km sprinting.0', pd.Series(0, index=df.index)).fillna(0)
        )
    
    if 'total km' in df.columns:
        df['intensity_share'] = (df['total_intensity_km'] / df['total km'].replace(0, 1)).fillna(0)
    else:
        df['intensity_share'] = 0
    
    # Group by user and week
    weekly = df.groupby([user_id_col, 'week_id']).agg({
        'total km': 'sum',
        'total_intensity_km': 'sum',
        'intensity_share': 'mean',
    }).reset_index()
    
    # Identify easy weeks (low intensity, lower volume)
    for uid in weekly[user_id_col].unique():
        user_weekly = weekly[weekly[user_id_col] == uid].copy()
        if len(user_weekly) < 4:
            continue
        
        user_weekly = user_weekly.sort_values('week_id')
        median_km = user_weekly['total km'].median()
        median_intensity = user_weekly['intensity_share'].median()
        
        # Easy week: intensity < 10% AND volume < median
        user_weekly['is_easy_week'] = (
            (user_weekly['intensity_share'] < 0.10) & 
            (user_weekly['total km'] < median_km)
        )
        
        easy_weeks = user_weekly['is_easy_week'].sum()
        total_weeks = len(user_weekly)
        
        if easy_weeks > 0:
            # Check spacing between easy weeks
            easy_indices = user_weekly[user_weekly['is_easy_week']].index.tolist()
            if len(easy_indices) > 1:
                gaps = [easy_indices[i+1] - easy_indices[i] for i in range(len(easy_indices)-1)]
                avg_gap = np.mean(gaps) if gaps else 0
                print(f"User {uid}: {easy_weeks}/{total_weeks} easy weeks ({easy_weeks/total_weeks*100:.1f}%), avg gap: {avg_gap:.1f} weeks")
    
    print()
    print("✅ Training block analysis complete")
    print()

def analyze_athlete_grouping(df, user_id_col='Athlete ID'):
    """Group athletes into short/medium vs long distance based on training patterns."""
    print("=" * 80)
    print("ATHLETE GROUPING ANALYSIS")
    print("=" * 80)
    print()
    
    # Calculate user-level statistics
    user_stats = df.groupby(user_id_col).agg({
        'total km': ['mean', 'sum'],
        'km sprinting': ['mean', 'sum'],
        'km Z3-4': ['mean', 'sum'],
        'km Z5-T1-T2': ['mean', 'sum'],
    }).round(3)
    
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
    user_stats = user_stats.reset_index()
    
    # Calculate intensity metrics
    user_stats['total_km_mean'] = user_stats['total km_mean']
    user_stats['sprinting_km_mean'] = user_stats['km sprinting_mean'].fillna(0)
    user_stats['z3_4_km_mean'] = user_stats['km Z3-4_mean'].fillna(0)
    user_stats['z5_km_mean'] = user_stats['km Z5-T1-T2_mean'].fillna(0)
    user_stats['total_intensity_km'] = (
        user_stats['z3_4_km_mean'] + 
        user_stats['z5_km_mean'] + 
        user_stats['sprinting_km_mean']
    )
    user_stats['intensity_share'] = (
        user_stats['total_intensity_km'] / user_stats['total_km_mean'].replace(0, 1)
    ).fillna(0)
    user_stats['sprinting_share'] = (
        user_stats['sprinting_km_mean'] / user_stats['total_km_mean'].replace(0, 1)
    ).fillna(0)
    user_stats['tempo_share'] = (
        user_stats['z3_4_km_mean'] / user_stats['total_km_mean'].replace(0, 1)
    ).fillna(0)
    
    print("📊 Overall statistics:")
    print(f"  Total users: {len(user_stats)}")
    print(f"  Average weekly km: {user_stats['total_km_mean'].mean() * 7:.2f} km/week")
    print(f"  Average sprinting: {user_stats['sprinting_km_mean'].mean():.4f} km/day ({user_stats['sprinting_share'].mean() * 100:.2f}% of total)")
    print(f"  Average intensity share: {user_stats['intensity_share'].mean() * 100:.2f}%")
    print(f"  Average tempo share: {user_stats['tempo_share'].mean() * 100:.2f}%")
    print()
    
    # Cluster athletes based on sprinting, intensity, and total km
    X = user_stats[['sprinting_share', 'intensity_share', 'total_km_mean']].fillna(0)
    X_scaled = (X - X.mean()) / (X.std() + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Try 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    user_stats['cluster'] = kmeans.fit_predict(X_scaled)
    
    print("=" * 80)
    print("CLUSTER ANALYSIS (2 groups)")
    print("=" * 80)
    print()
    
    for cluster_id in [0, 1]:
        cluster_data = user_stats[user_stats['cluster'] == cluster_id]
        print(f"Cluster {cluster_id} (n={len(cluster_data)}, {len(cluster_data)/len(user_stats)*100:.1f}%):")
        print(f"  Avg weekly km: {cluster_data['total_km_mean'].mean() * 7:.2f} km/week")
        print(f"  Avg sprinting: {cluster_data['sprinting_km_mean'].mean():.4f} km/day ({cluster_data['sprinting_share'].mean() * 100:.2f}% of total)")
        print(f"  Avg intensity share: {cluster_data['intensity_share'].mean() * 100:.2f}%")
        print(f"  Avg tempo share: {cluster_data['tempo_share'].mean() * 100:.2f}%")
        print(f"  Avg Z3-4: {cluster_data['z3_4_km_mean'].mean():.4f} km/day")
        print(f"  Avg Z5: {cluster_data['z5_km_mean'].mean():.4f} km/day")
        print()
    
    # Identify which is short/medium vs long distance
    cluster_0_sprinting = user_stats[user_stats['cluster'] == 0]['sprinting_share'].mean()
    cluster_1_sprinting = user_stats[user_stats['cluster'] == 1]['sprinting_share'].mean()
    
    if cluster_0_sprinting > cluster_1_sprinting:
        short_medium_cluster = 0
        long_distance_cluster = 1
    else:
        short_medium_cluster = 1
        long_distance_cluster = 0
    
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print(f"Short/Medium Distance Athletes (Cluster {short_medium_cluster}):")
    short_data = user_stats[user_stats['cluster'] == short_medium_cluster]
    long_data = user_stats[user_stats['cluster'] == long_distance_cluster]
    print(f"  - Higher sprinting: {short_data['sprinting_share'].mean() * 100:.2f}% vs {long_data['sprinting_share'].mean() * 100:.2f}%")
    print(f"  - Higher intensity: {short_data['intensity_share'].mean() * 100:.2f}% vs {long_data['intensity_share'].mean() * 100:.2f}%")
    print(f"  - Total km: {short_data['total_km_mean'].mean() * 7:.2f} km/week vs {long_data['total_km_mean'].mean() * 7:.2f} km/week")
    print()
    print(f"Long Distance Athletes (Cluster {long_distance_cluster}):")
    print(f"  - Lower sprinting: {long_data['sprinting_share'].mean() * 100:.2f}%")
    print(f"  - Lower intensity: {long_data['intensity_share'].mean() * 100:.2f}%")
    print(f"  - More tempo: {long_data['tempo_share'].mean() * 100:.2f}% vs {short_data['tempo_share'].mean() * 100:.2f}%")
    print(f"  - Total km: {long_data['total_km_mean'].mean() * 7:.2f} km/week")
    print()
    
    return user_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc0-path", required=True, help="Path to real CC0 day file")
    parser.add_argument("--user-id-col", default="Athlete ID", help="User ID column name")
    parser.add_argument("--date-col", default="Date", help="Date column name")
    args = parser.parse_args()
    
    print("=" * 80)
    print("REAL CC0 DATA ANALYSIS: Training Blocks & Athlete Grouping")
    print("=" * 80)
    print()
    
    # Load data
    df = pd.read_csv(args.cc0_path)
    print(f"Loaded {len(df):,} rows")
    print(f"Total users: {df[args.user_id_col].nunique():,}")
    print()
    
    # Analyze training blocks
    if args.date_col in df.columns:
        analyze_training_blocks(df, args.user_id_col, args.date_col)
    else:
        print("⚠️ No date column found - skipping training block analysis")
        print()
    
    # Analyze athlete grouping
    user_stats = analyze_athlete_grouping(df, args.user_id_col)
    
    print("=" * 80)
    print("✅ Analysis complete")
    print("=" * 80)
