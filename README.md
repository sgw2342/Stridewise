# StrideWise Synthetic Runner Data Generator - Production Ready

**Version**: 1.0  
**Date**: 2024  
**Status**: Production Ready ✅

---

## Overview

This is a production-ready system for generating realistic synthetic runner data for injury prediction model training. The system uses an algorithmic approach to generate physiological data and injury events based on real-world processes, avoiding post-hoc data manipulation.

### Key Features

- ✅ **Algorithmic Data Generation**: Realistic physiological data emerges from training patterns
- ✅ **Injury Prediction Models**: Main model (trained on synthetic data) and standalone model (trained on real CC0 data)
- ✅ **Validation Strategy**: Synthetic data validated against real-world CC0 dataset
- ✅ **Web Application**: Interactive interface for data generation and model evaluation
- ✅ **Production Performance**: Main model achieves ROC AUC ≥0.70

---

## System Architecture

### Core Goal
Predict injury risk in runners using rich smartwatch data, despite data privacy constraints limiting access to large real-world datasets.

### Validation Strategy
1. **Synthetic Data Generation**: Algorithmic process producing realistic physiological data
2. **Main Model**: Trained on synthetic data to predict injury risk from rich smartwatch data
3. **Standalone Model**: Trained on real CC0 data (aggregated format)
4. **Validation**: Convert synthetic data to CC0 format → test against standalone model
5. **Indirect Validation**: Reasonable performance validates synthetic data generation → validates main model for real data

### Current Performance

**Main Model** (trained on synthetic data):
- ROC AUC: **0.7136** (target ≥0.70 ✅)
- PR AUC: 0.2427
- Brier Score: 0.0973
- Dataset: 3,000 users, 180 days (395,693 rows)

**Standalone Model** (trained on real CC0 data):
- ROC AUC: **0.7121** (on real CC0 test set)
- PR AUC: 0.0450

**Standalone Model** (on synthetic CC0 data - 250 elite users):
- ROC AUC: **0.6167** (vs 0.7121 on real CC0)
- PR AUC: 0.0216
- Performance: 86.6% of real CC0 performance

**Synthetic Data Characteristics** (250 elite users):
- Injury rate: 1.50% (vs real CC0: ~1.4-1.6%)
- Performance on standalone model: 86.6% of real CC0

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For web app (optional)
pip install -r requirements_app.txt
```

### 2. Generate Synthetic Data

```bash
python stridewise_synth_generate.py \
  --n-users 1000 \
  --n-days 365 \
  --seed 42 \
  --out ./output
```

**Output Files**:
- `output/users.csv` - Runner profiles
- `output/daily.csv` - Daily metrics and signals
- `output/activities.csv` - Training sessions
- `output/metadata.json` - Dataset metadata

### 3. Train Main Model

```bash
python stridewise_train_main_model.py \
  --daily ./output/daily.csv \
  --users ./output/users.csv \
  --activities ./output/activities.csv \
  --out ./models/main_model
```

### 4. Validate Against Real Data (CC0 Format)

```bash
# Convert to CC0 format
python convert_synth_to_cc0.py \
  --daily ./output/daily.csv \
  --schema ./cc0_feature_schema.json \
  --out-dir ./output_cc0

# Evaluate against standalone model
python evaluate_synth_cc0.py \
  --synth-cc0 ./output_cc0/day_approach_maskedID_timeseries.csv \
  --model-dir ./models/standalone_cc0_real \
  --out ./validation_results
```

### 5. Statistical Validation & Visualization

```bash
# Comprehensive data validation visualizations (NEW)
python validation/comprehensive_data_validation_visualizations.py \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --synth-cc0 ./synth_cc0/day_approach_maskedID_timeseries.csv \
  --real-daily ./real_daily.csv \
  --synth-daily ./synth_daily.csv \
  --out ./validation_visualizations

# Create main model performance visualizations
python validation/create_main_model_visualizations.py \
  --model-dir ./models/main_model_large_dataset \
  --predictions ./eval_results/predictions.csv \
  --out ./main_model_visualizations

# Analyze injury distributions
python validation/analyze_injury_distribution_comparison.py

# Compare distributions
python validation/compare_all_distributions.py \
  --synth ./output/daily.csv \
  --real ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv
```

See `docs/STATISTICAL_VALIDATION_GUIDE.md`, `docs/VISUALIZATION_GUIDE.md`, and `docs/COMPREHENSIVE_VISUALIZATION_GUIDE.md` for details.

### 6. Run Web Application (Optional)

```bash
python app.py
```

Access at: `http://localhost:5001`

---

## Directory Structure

```
production_ready/
├── README.md                          # This file
├── requirements.txt                   # Core dependencies
├── requirements_app.txt               # Web app dependencies
├── cc0_feature_schema.json           # CC0 format schema
│
├── synthrun_gen/                     # Core data generation module
│   ├── __init__.py
│   ├── config.py                     # Configuration parameters
│   ├── users.py                      # User profile generation
│   ├── activities.py                 # Training session generation
│   ├── daily.py                      # Daily plan and metrics
│   ├── events.py                     # Injury/illness event generation
│   ├── pipeline.py                   # Main generation pipeline
│   └── main_model/
│       └── features.py               # Feature engineering for main model
│
├── scripts/                          # Utility scripts
│   ├── stridewise_synth_generate.py  # Data generation script
│   ├── stridewise_train_main_model.py # Main model training
│   ├── stridewise_train_standalone_cc0_paper_method.py  # Standalone model training
│   ├── evaluate_main_model.py        # Main model evaluation
│   ├── evaluate_synth_cc0.py         # CC0 validation
│   └── convert_synth_to_cc0.py       # CC0 format conversion
│
├── validation/                       # Validation & analysis scripts
│   ├── create_performance_visualizations.py  # Performance visualization
│   ├── analyze_injury_distribution_comparison.py  # Injury distribution analysis
│   ├── analyze_injury_distribution_by_profile.py  # Profile-based analysis
│   ├── analyze_cc0_training_patterns.py  # Training pattern analysis
│   ├── compare_all_distributions.py  # Distribution comparison
│   ├── compare_cc0_correlations.py  # Correlation analysis
│   ├── compare_cc0_distributions_comprehensive.py  # Comprehensive comparison
│   ├── compare_cc0_injury_causation.py  # Injury causation analysis
│   └── compare_feature_importance.py  # Feature importance comparison
│
├── models/                           # Trained models
│   ├── main_model_large_dataset/    # Latest main model (best performance)
│   │   ├── main_model_cat.cbm       # Trained CatBoost model
│   │   ├── main_model_schema.json   # Feature schema
│   │   └── main_model_metrics.json  # Performance metrics
│   └── standalone_cc0_real/         # Standalone model (trained on real CC0)
│       ├── bagged_models.joblib     # Trained bagged XGBoost models
│       └── model_meta.json          # Performance metrics
│
├── app.py                            # Web application
├── static/                           # Web app static files
├── templates/                        # Web app templates
│
└── docs/                             # Documentation
    ├── USER_GUIDE.md                 # User guide
    ├── API_REFERENCE.md              # API reference
    ├── VALIDATION_STRATEGY.md        # Validation approach
    ├── MODEL_PERFORMANCE.md          # Performance metrics
    ├── CONFIGURATION.md              # Configuration guide
    ├── STATISTICAL_VALIDATION_GUIDE.md  # Statistical validation guide
    └── VISUALIZATION_GUIDE.md        # Visualization guide
```

---

## Core Components

### 1. Synthetic Data Generation (`synthrun_gen/`)

**Purpose**: Generate realistic runner data algorithmically

**Key Modules**:
- `config.py`: Configuration parameters (injury hazards, risk factors, etc.)
- `users.py`: Generate user profiles (novice, recreational, advanced, elite)
- `activities.py`: Generate training sessions (easy, tempo, interval, long)
- `daily.py`: Build daily training plans and generate perceived metrics
- `events.py`: Generate injury/illness events based on risk factors
- `pipeline.py`: Orchestrate the generation process

**Key Features**:
- Profile-based training patterns
- Realistic physiological signals (HRV, RHR, sleep, stress)
- Injury events driven by sprinting, long-run spikes, training load
- Device wear compliance and missingness
- Algorithmic perceived metrics (exertion, training success, recovery)

### 2. Model Training

**Main Model** (`stridewise_train_main_model.py`):
- Trained on synthetic data
- Uses 335+ engineered features
- CatBoost algorithm
- Performance: ROC AUC 0.7136

**Standalone Model** (`stridewise_train_standalone_cc0_paper_method.py`):
- Trained on real CC0 data
- Approximates scientific paper's method
- Used for validation of synthetic data

### 3. Validation

**CC0 Conversion** (`convert_synth_to_cc0.py`):
- Converts rich synthetic data to aggregated CC0 format
- Matches format used in scientific paper

**Evaluation** (`evaluate_synth_cc0.py`):
- Tests synthetic CC0 data against standalone model
- Provides performance metrics for validation

---

## Configuration

### Key Parameters (`synthrun_gen/config.py`)

**Injury Hazards** (base injury probabilities):
- `injury_hazard_novice`: 0.00624
- `injury_hazard_recreational`: 0.00492
- `injury_hazard_advanced`: 0.00115
- `injury_hazard_elite`: 0.00138

**Spike Absolute Risk** (long-run spike injury risk):
- `spike_absolute_risk_small`: 0.04331 (+12% from original)
- `spike_absolute_risk_moderate`: 0.04950
- `spike_absolute_risk_large`: 0.06188

**Sprinting Absolute Risk**:
- `sprinting_absolute_risk_per_km`: 0.1381 (default)
- `sprinting_absolute_risk_per_km_advanced_elite`: 0.060

See `docs/CONFIGURATION.md` for full parameter documentation.

---

## Usage Examples

### Generate Small Test Dataset

```bash
python stridewise_synth_generate.py \
  --n-users 100 \
  --n-days 90 \
  --seed 42 \
  --out ./test_output
```

### Generate Large Training Dataset

```bash
python stridewise_synth_generate.py \
  --n-users 3000 \
  --n-days 365 \
  --seed 42 \
  --out ./training_data
```

### Generate Elite-Only Dataset

```bash
python stridewise_synth_generate.py \
  --n-users 250 \
  --n-days 365 \
  --elite-only \
  --seed 42 \
  --out ./elite_data
```

### Train Main Model

```bash
python stridewise_train_main_model.py \
  --daily ./training_data/daily.csv \
  --users ./training_data/users.csv \
  --activities ./training_data/activities.csv \
  --out ./models/main_model_v1
```

### Evaluate Model on New Data

```bash
python evaluate_main_model.py \
  --daily ./new_data/daily.csv \
  --users ./new_data/users.csv \
  --activities ./new_data/activities.csv \
  --model-dir ./models/main_model_recommendations_implemented \
  --out ./evaluation_results
```

---

## Performance Metrics

### Main Model Performance

**Training Data** (3,000 users, 180 days, 395,693 rows):
- ROC AUC (validation): 0.7136 ✅
- PR AUC (validation): 0.2427
- Brier Score (validation): 0.0973
- Overfitting gap: -0.0023 (validation slightly better - excellent generalization)
- Status: ✅ Target met (≥0.70)

### Standalone Model Validation

**Real CC0 Test Performance**:
- ROC AUC: 0.7121
- PR AUC: 0.0450

**Synthetic CC0 Performance** (250 elite users):
- ROC AUC: 0.6167 (86.6% of real)
- PR AUC: 0.0216
- Status: ✅ Reasonable validation performance

---

## Documentation

See `docs/` directory for detailed documentation:
- `USER_GUIDE.md`: Complete user guide
- `API_REFERENCE.md`: API documentation
- `VALIDATION_STRATEGY.md`: Validation approach details
- `MODEL_PERFORMANCE.md`: Performance analysis
- `CONFIGURATION.md`: Configuration parameters

---

## Requirements

### Python Version
- Python 3.8+

### Core Dependencies
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 0.24.0
- catboost >= 1.0.0
- xgboost >= 1.5.0

### Web App Dependencies (Optional)
- flask >= 2.0.0
- flask-cors >= 3.0.0

See `requirements.txt` and `requirements_app.txt` for complete lists.

---

## License

[Specify your license here]

---

## Support

For issues, questions, or contributions, please [specify contact method].

---

## Changelog

### Version 1.0 (Current)
- ✅ Main model achieves ROC AUC ≥0.70
- ✅ Algorithmic data generation (no post-hoc manipulation)
- ✅ Validation against real CC0 data
- ✅ Production-ready codebase
- ✅ Comprehensive documentation

---

**Last Updated**: 2024
