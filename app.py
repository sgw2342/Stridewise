#!/usr/bin/env python3
"""Injury Prediction Web App

A simple web application that provides daily injury risk predictions for runners.
Users input their daily status (illness, pains) and receive a risk assessment:
- Green: Good to train
- Orange: Avoid intense/long runs
- Red: Take rest days
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# Import for feature engineering
from synthrun_gen.main_model.features import build_main_model_table

app = Flask(__name__)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models/main_model_large_dataset"))
SCHEMA_PATH = MODEL_DIR / "main_model_schema.json"

# Global variables for model
model = None
model_type = None  # 'catboost', 'xgboost', or 'lightgbm'
model_schema = None
feature_cols = None


def load_model():
    """Load the trained model and schema."""
    global model, model_type, model_schema, feature_cols
    
    if model is not None:
        return  # Already loaded
    
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
    
    # Try to find model file (CatBoost, XGBoost, or LightGBM)
    model_path = None
    model_type = None
    
    # Try CatBoost first (default)
    catboost_path = MODEL_DIR / "main_model_cat.cbm"
    if catboost_path.exists():
        model_path = catboost_path
        model_type = "catboost"
    else:
        # Try XGBoost
        xgb_path = MODEL_DIR / "main_model_xgb.json"
        if xgb_path.exists():
            model_path = xgb_path
            model_type = "xgboost"
        else:
            # Try LightGBM
            lgb_path = MODEL_DIR / "main_model_lgb.txt"
            if lgb_path.exists():
                model_path = lgb_path
                model_type = "lightgbm"
    
    if model_path is None:
        # Try to find any model file
        model_files = list(MODEL_DIR.glob("main_model_*.cbm")) + \
                     list(MODEL_DIR.glob("main_model_*.json")) + \
                     list(MODEL_DIR.glob("main_model_*.txt"))
        if model_files:
            model_path = model_files[0]
            if model_path.suffix == ".cbm":
                model_type = "catboost"
            elif model_path.suffix == ".json":
                model_type = "xgboost"
            elif model_path.suffix == ".txt":
                model_type = "lightgbm"
    
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {MODEL_DIR}. Expected main_model_cat.cbm, main_model_xgb.json, or main_model_lgb.txt")
    
    # Load model based on type
    if model_type == "catboost":
        if cb is None:
            raise RuntimeError("catboost not installed. Install with: pip install catboost")
        model = cb.CatBoostClassifier()
        model.load_model(str(model_path))
        print(f"✅ CatBoost model loaded from {model_path}")
    elif model_type == "xgboost":
        if xgb is None:
            raise RuntimeError("xgboost not installed. Install with: pip install xgboost")
        model = xgb.Booster()
        model.load_model(str(model_path))
        print(f"✅ XGBoost model loaded from {model_path}")
    elif model_type == "lightgbm":
        if lgb is None:
            raise RuntimeError("lightgbm not installed. Install with: pip install lightgbm")
        model = lgb.Booster(model_file=str(model_path))
        print(f"✅ LightGBM model loaded from {model_path}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load schema
    with open(SCHEMA_PATH) as f:
        model_schema = json.load(f)
    
    feature_cols = model_schema.get("feature_cols", [])
    
    print(f"✅ Model loaded: {len(feature_cols)} features, type: {model_type}")


def generate_user_data(config: Dict, output_dir: Path) -> Dict[str, Path]:
    """
    Generate synthetic data for a single user based on configuration.
    
    Args:
        config: User configuration dictionary (currently not used, but saved for future use)
        output_dir: Directory to save generated data
    
    Returns:
        Dictionary with paths to generated files
    
    Note: Currently generates default synthetic data. The user's training configuration
    (sprint, long runs, tempo) is saved but not yet applied to data generation.
    Future enhancement: Modify generated activities based on user config.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a different seed each time to generate different data
    # Use timestamp to ensure uniqueness, but keep it deterministic within same second
    import time
    seed = int(time.time()) % 1000000  # Use timestamp-based seed for variation
    
    # Run the synthetic data generator
    cmd = [
        sys.executable,
        'stridewise_synth_generate.py',
        '--out', str(output_dir),
        '--n-users', '1',
        '--n-days', '90',
        '--seed', str(seed)  # Use variable seed instead of fixed 42
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Data generation failed: {result.stderr}")
        
        # Apply user's training configuration to the last 7 days
        apply_user_training_config(config, output_dir)
        
        # Return paths to generated files
        return {
            'daily': output_dir / 'daily.csv',
            'users': output_dir / 'users.csv',
            'activities': output_dir / 'activities.csv'
        }
    except subprocess.TimeoutExpired:
        raise RuntimeError("Data generation timed out")
    except Exception as e:
        raise RuntimeError(f"Failed to generate data: {str(e)}")


def apply_user_training_config(config: Dict, output_dir: Path):
    """
    Apply user's training configuration to the last 7 days of generated data.
    Modifies activities.csv to add sprint, long runs, and tempo runs as specified.
    
    Args:
        config: User configuration dictionary with training preferences
        output_dir: Directory containing generated data files
    """
    activities_file = output_dir / 'activities.csv'
    daily_file = output_dir / 'daily.csv'
    
    if not activities_file.exists() or not daily_file.exists():
        return  # Skip if files don't exist
    
    # Load activities and daily data
    activities_df = pd.read_csv(activities_file)
    daily_df = pd.read_csv(daily_file)
    
    # Convert dates
    activities_df['date'] = pd.to_datetime(activities_df['date'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Get the last 7 days
    max_date = daily_df['date'].max()
    last_7_days = pd.date_range(end=max_date, periods=7, freq='D')
    
    # Get user_id
    user_id = daily_df['user_id'].iloc[0]
    
    # Find next activity_id
    next_activity_id = activities_df['activity_id'].max() + 1 if len(activities_df) > 0 else 1
    
    new_activities = []
    
    # Helper function to create activity
    def create_activity(date, session_type, distance_km, duration_min, avg_hr_bpm, 
                       kms_z3_4=0, kms_z5=0, kms_sprinting=0, pace_min_per_km=None):
        """Create an activity row."""
        if pace_min_per_km is None:
            # Estimate pace from distance and duration
            pace_min_per_km = duration_min / distance_km if distance_km > 0 else 5.0
        
        return {
            'activity_id': next_activity_id + len(new_activities),
            'user_id': user_id,
            'date': date,
            'session_type': session_type,  # Use session_type to match activities.csv structure
            'distance_km': distance_km,
            'duration_min': duration_min,
            'avg_hr_bpm': avg_hr_bpm,
            'elev_gain_m': 0,  # Default
            'kms_z3_4': kms_z3_4,
            'kms_z5_t1_t2': kms_z5,
            'kms_sprinting': kms_sprinting,
            'pace_min_per_km': pace_min_per_km,
            'cadence_spm': 180,  # Default
            'gct_ms': 250,  # Default
            'stride_length_cm': 120,  # Default
            'vertical_oscillation_cm': 8,  # Default
            'gct_balance': 50.0,  # Default
        }
    
    # Distribute sessions across last 7 days
    days_with_activities = []
    
    # Add sprint training
    if config.get('include_sprint', False):
        sprint_sessions = config.get('sprint_sessions', 0)
        sprint_kms = config.get('sprint_kms', 0.0)
        
        # Only add activities if sessions > 0 and kms > 0
        if sprint_sessions > 0 and sprint_kms > 0:
            # Distribute sprint sessions across available days
            sprint_days = list(last_7_days[-sprint_sessions:]) if sprint_sessions <= 7 else list(last_7_days)
            
            for day in sprint_days:
                # Sprint session: short distance, high intensity
                duration = sprint_kms * 3.5  # ~3.5 min/km for sprint
                avg_hr = 180  # High heart rate for sprint
                new_activities.append(create_activity(
                    date=day,
                    session_type='interval',  # Use 'interval' for sprint sessions
                    distance_km=sprint_kms,
                    duration_min=duration,
                    avg_hr_bpm=avg_hr,
                    kms_sprinting=sprint_kms,
                    pace_min_per_km=3.5
                ))
                days_with_activities.append(day)
    
    # Add long runs
    if config.get('include_long', False):
        long_sessions = config.get('long_sessions', 0)
        long_distance = config.get('long_distance', 0.0)
        
        # Only add activities if sessions > 0 and distance > 0
        if long_sessions > 0 and long_distance > 0:
            # Distribute long runs across available days (avoiding sprint days)
            available_days = [d for d in last_7_days if d not in days_with_activities]
            long_days = available_days[-long_sessions:] if long_sessions <= len(available_days) else available_days
            
            for day in long_days:
                # Long run: longer distance, moderate pace
                pace = 5.0  # ~5 min/km for long run
                duration = long_distance * pace
                avg_hr = 140  # Moderate heart rate for long run
                new_activities.append(create_activity(
                    date=day,
                    session_type='long',
                    distance_km=long_distance,
                    duration_min=duration,
                    avg_hr_bpm=avg_hr,
                    kms_z3_4=long_distance * 0.8,  # Most in zone 3-4
                    pace_min_per_km=pace
                ))
                days_with_activities.append(day)
    
    # Add tempo runs
    if config.get('include_tempo', False):
        tempo_sessions = config.get('tempo_sessions', 0)
        tempo_zone = config.get('tempo_zone', 3)
        tempo_kms = config.get('tempo_kms', 0.0)
        
        # Only add activities if sessions > 0 and kms > 0
        if tempo_sessions > 0 and tempo_kms > 0:
            # Distribute tempo runs across available days
            available_days = [d for d in last_7_days if d not in days_with_activities]
            tempo_days = available_days[-tempo_sessions:] if tempo_sessions <= len(available_days) else available_days
            
            for day in tempo_days:
                # Tempo run: moderate distance, tempo pace
                pace = 4.0 if tempo_zone <= 3 else 3.5  # Faster for higher zones
                duration = tempo_kms * pace
                avg_hr = 150 if tempo_zone == 3 else (160 if tempo_zone == 4 else 170)
                
                # Zone distribution based on tempo_zone
                if tempo_zone == 3:
                    kms_z3_4 = tempo_kms
                    kms_z5 = 0
                elif tempo_zone == 4:
                    kms_z3_4 = tempo_kms * 0.3
                    kms_z5 = tempo_kms * 0.7
                else:  # zone 5
                    kms_z3_4 = 0
                    kms_z5 = tempo_kms
                
                new_activities.append(create_activity(
                    date=day,
                    session_type='tempo',
                    distance_km=tempo_kms,
                    duration_min=duration,
                    avg_hr_bpm=avg_hr,
                    kms_z3_4=kms_z3_4,
                    kms_z5=kms_z5,
                    pace_min_per_km=pace
                ))
                days_with_activities.append(day)
    
    # Add new activities to activities dataframe
    if new_activities:
        new_activities_df = pd.DataFrame(new_activities)
        
        # Ensure all columns match
        for col in activities_df.columns:
            if col not in new_activities_df.columns:
                new_activities_df[col] = 0
        
        # Reorder columns to match
        new_activities_df = new_activities_df[activities_df.columns]
        
        # Combine with existing activities
        activities_df = pd.concat([activities_df, new_activities_df], ignore_index=True)
        
        # Save updated activities
        activities_df.to_csv(activities_file, index=False)
        
        # Recalculate daily aggregates for the last 7 days from activities
        recalculate_daily_aggregates(activities_df, daily_df, last_7_days, daily_file)
        
        print(f"✅ Added {len(new_activities)} activities and updated daily aggregates based on user configuration")


def recalculate_daily_aggregates(activities_df: pd.DataFrame, daily_df: pd.DataFrame, 
                                 target_days: pd.DatetimeIndex, daily_file: Path):
    """
    Recalculate daily aggregates from activities for specified days.
    
    Args:
        activities_df: Activities dataframe (with new activities added)
        daily_df: Daily dataframe to update
        target_days: Days to recalculate
        daily_file: Path to save updated daily.csv
    """
    # Filter activities for target days
    target_activities = activities_df[activities_df['date'].isin(target_days)].copy()
    
    if len(target_activities) == 0:
        return
    
    # Group by user_id and date to calculate aggregates
    agg_dict = {
        'activity_id': 'count',  # Number of sessions
        'distance_km': 'sum',
        'duration_min': 'sum',
        'avg_hr_bpm': 'mean',
        'elev_gain_m': 'sum',
        'kms_z3_4': 'sum',
        'kms_z5_t1_t2': 'sum',
        'kms_sprinting': 'sum',
    }
    
    # Calculate aggregates
    daily_aggs = target_activities.groupby(['user_id', 'date']).agg(agg_dict).reset_index()
    daily_aggs.rename(columns={'activity_id': 'sessions'}, inplace=True)
    
    # Update daily_df for target days
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    for _, agg_row in daily_aggs.iterrows():
        # Find matching row in daily_df
        mask = (daily_df['user_id'] == agg_row['user_id']) & (daily_df['date'] == agg_row['date'])
        
        if mask.any():
            idx = daily_df[mask].index[0]
            # Update aggregate columns
            if 'sessions' in daily_df.columns:
                daily_df.at[idx, 'sessions'] = agg_row['sessions']
            if 'km_total' in daily_df.columns:
                daily_df.at[idx, 'km_total'] = agg_row['distance_km']
            if 'duration_min' in daily_df.columns:
                daily_df.at[idx, 'duration_min'] = agg_row['duration_min']
            if 'avg_hr_bpm' in daily_df.columns:
                daily_df.at[idx, 'avg_hr_bpm'] = agg_row['avg_hr_bpm']
            if 'elev_gain_m' in daily_df.columns:
                daily_df.at[idx, 'elev_gain_m'] = agg_row['elev_gain_m']
            if 'kms_z3_4' in daily_df.columns:
                daily_df.at[idx, 'kms_z3_4'] = agg_row['kms_z3_4']
            if 'kms_z5_t1_t2' in daily_df.columns:
                daily_df.at[idx, 'kms_z5_t1_t2'] = agg_row['kms_z5_t1_t2']
            if 'kms_sprinting' in daily_df.columns:
                daily_df.at[idx, 'kms_sprinting'] = agg_row['kms_sprinting']
    
    # Save updated daily.csv
    daily_df.to_csv(daily_file, index=False)


def predict_risk_with_model(user_data: Dict) -> Dict:
    """
    Predict injury risk using the trained model and generated user data.
    
    Args:
        user_data: Dictionary containing:
            - illness: bool (feeling ill)
            - pains: bool (pains/issues related to training)
    
    Returns:
        Dictionary with:
            - risk_level: str ('green', 'orange', 'red')
            - risk_score: float (0-1, as percentage)
            - message: str (user-friendly message)
    """
    load_model()
    
    # Check if user data file exists
    user_data_dir = Path('./user_data')
    daily_file = user_data_dir / 'daily.csv'
    
    if not daily_file.exists():
        raise FileNotFoundError("User data not found. Please create sample data first.")
    
    # Load user data
    daily_df = pd.read_csv(daily_file)
    users_df = pd.read_csv(user_data_dir / 'users.csv') if (user_data_dir / 'users.csv').exists() else None
    activities_df = pd.read_csv(user_data_dir / 'activities.csv') if (user_data_dir / 'activities.csv').exists() else None
    
    # Build feature table
    df_model, _ = build_main_model_table(
        daily=daily_df,
        users=users_df,
        activities=activities_df,
        label_col='injury_next_7d',
        include_users=True,
        include_activity_aggs=True
    )
    
    # Get the most recent day (today)
    df_model['date'] = pd.to_datetime(df_model['date'])
    latest_day = df_model.sort_values('date').iloc[-1]
    
    # Prepare features for prediction
    X = latest_day[feature_cols].to_frame().T
    
    # Fill missing features with 0
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    
    # Ensure correct order
    X = X[feature_cols].fillna(0.0)
    
    # Make prediction based on model type
    if model_type == "catboost":
        # CatBoost uses predict_proba for probabilities
        risk_score = float(model.predict_proba(X)[0, 1])  # Probability of positive class
    elif model_type == "xgboost":
        # XGBoost needs DMatrix
        dtest = xgb.DMatrix(X)
        pred = model.predict(dtest)
        # XGBoost might return probabilities or raw scores depending on objective
        # If it's probability, use as-is; if raw score, apply sigmoid
        risk_score = float(pred[0])
        if risk_score > 1.0:  # Likely raw score, apply sigmoid
            risk_score = 1.0 / (1.0 + np.exp(-risk_score))
    elif model_type == "lightgbm":
        # LightGBM predict returns probabilities by default
        risk_score = float(model.predict(X)[0])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Convert to percentage
    risk_percentage = risk_score * 100
    
    # Apply sprinting absolute risk (additive, like in data generation)
    # This matches the injury generation logic: sprint_km × 0.1381, capped at 40%
    # Sprinting on day t-1 (yesterday) affects risk on day t (today)
    # Note: Risk per km is the same for all fitness levels
    # Novice/recreational runners have fewer sprinting injuries because they do sprinting less frequently
    sprinting_risk_per_km = 0.1381  # 13.81% per km (same for all fitness levels)
    sprinting_risk_clip_max = 0.40  # Cap at 40%
    
    # Get sprinting from daily_df (need to check day t-1, which affects day t)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df_sorted = daily_df.sort_values('date')
    
    # Get sprinting on the day before the latest day (day t-1 affects day t)
    if len(daily_df_sorted) >= 2:
        # Day t-1 (yesterday) - sprinting here affects today's risk
        prev_day = daily_df_sorted.iloc[-2]
        sprint_km = float(prev_day['kms_sprinting']) if 'kms_sprinting' in prev_day and pd.notna(prev_day['kms_sprinting']) else 0.0
    elif len(daily_df_sorted) == 1:
        # Only one day - use that day's sprinting (might be same day)
        latest_daily = daily_df_sorted.iloc[-1]
        sprint_km = float(latest_daily['kms_sprinting']) if 'kms_sprinting' in latest_daily and pd.notna(latest_daily['kms_sprinting']) else 0.0
    else:
        sprint_km = 0.0
    
    # Calculate sprinting absolute risk
    sprinting_abs_risk = sprint_km * sprinting_risk_per_km
    sprinting_abs_risk = min(sprinting_abs_risk, sprinting_risk_clip_max)  # Cap at 40%
    sprinting_abs_risk_pct = sprinting_abs_risk * 100  # Convert to percentage
    
    # Add sprinting absolute risk to prediction
    risk_percentage += sprinting_abs_risk_pct
    
    # Apply illness and pains adjustments
    if user_data.get('illness', False):
        risk_percentage += 15  # Illness increases risk
    
    if user_data.get('pains', False):
        risk_percentage += 50  # Pains increase risk significantly
    
    # Cap at 100%
    risk_percentage = min(risk_percentage, 100.0)
    
    # Determine risk level based on thresholds
    if risk_percentage < 50:
        risk_level = 'green'
        message = "You're good to keep training. Maintain your normal routine."
    elif risk_percentage < 85:
        risk_level = 'orange'
        message = "Avoid intense training or particularly long runs. Consider easy sessions or rest."
    else:
        risk_level = 'red'
        message = "Your injury risk is high. Consider taking 1 or more rest days."
    
    return {
        'risk_level': risk_level,
        'risk_score': round(risk_percentage, 1),
        'message': message,
        'timestamp': datetime.now().isoformat()
    }


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/create_data')
def create_data():
    """Create sample data page."""
    return render_template('create_data.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for injury risk prediction using trained model."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate inputs
        illness = data.get('illness', False)
        pains = data.get('pains', False)
        
        if not isinstance(illness, bool):
            return jsonify({'error': 'illness must be a boolean'}), 400
        
        if not isinstance(pains, bool):
            return jsonify({'error': 'pains must be a boolean'}), 400
        
        # Make prediction using trained model
        result = predict_risk_with_model({
            'illness': illness,
            'pains': pains
        })
        
        return jsonify(result), 200
        
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_config', methods=['POST'])
def save_config():
    """Save user's training configuration and generate synthetic data."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['include_sprint', 'include_long', 'include_tempo']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create config directory if it doesn't exist
        config_dir = Path('./user_configs')
        config_dir.mkdir(exist_ok=True)
        
        # Save configuration to file
        config_file = config_dir / 'user_training_config.json'
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Generate synthetic data for the user (will overwrite existing if present)
        user_data_dir = Path('./user_data')
        try:
            # Remove existing data files if they exist (to ensure clean regeneration)
            if user_data_dir.exists():
                for file in ['daily.csv', 'users.csv', 'activities.csv']:
                    existing_file = user_data_dir / file
                    if existing_file.exists():
                        existing_file.unlink()
            
            generated_files = generate_user_data(data, user_data_dir)
            
            return jsonify({
                'status': 'success',
                'message': 'Configuration saved and data generated successfully',
                'config_file': str(config_file),
                'data_files': {k: str(v) for k, v in generated_files.items()}
            }), 200
        except Exception as gen_error:
            # Still save config even if generation fails
            return jsonify({
                'status': 'partial',
                'message': f'Configuration saved but data generation failed: {str(gen_error)}',
                'config_file': str(config_file),
                'error': str(gen_error)
            }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_config', methods=['GET'])
def get_config():
    """Get saved user configuration."""
    try:
        config_file = Path('./user_configs/user_training_config.json')
        
        if not config_file.exists():
            return jsonify({'error': 'No configuration found'}), 404
        
        with open(config_file) as f:
            config = json.load(f)
        
        return jsonify(config), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/check_data_exists', methods=['GET'])
def check_data_exists():
    """Check if user data files exist."""
    try:
        user_data_dir = Path('./user_data')
        daily_file = user_data_dir / 'daily.csv'
        
        exists = daily_file.exists()
        
        return jsonify({
            'exists': exists,
            'data_dir': str(user_data_dir)
        }), 200
        
    except Exception as e:
        return jsonify({'exists': False, 'error': str(e)}), 200


@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        load_model()
        return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


if __name__ == '__main__':
    # Try to load model on startup
    try:
        load_model()
    except Exception as e:
        print(f"⚠️  Warning: Could not load model on startup: {e}")
        print("   Model will be loaded on first prediction request")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
