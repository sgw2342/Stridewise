# User Guide

Complete guide to using the StrideWise Synthetic Runner Data Generator.

---

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Data Generation](#data-generation)
4. [Model Training](#model-training)
5. [Validation](#validation)
6. [Web Application](#web-application)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Web app dependencies (optional)
pip install -r requirements_app.txt
```

### Step 2: Verify Installation

```bash
python -c "import numpy, pandas, catboost; print('✅ Dependencies installed')"
```

---

## Basic Usage

### Generate Your First Dataset

```bash
python stridewise_synth_generate.py \
  --n-users 100 \
  --n-days 90 \
  --seed 42 \
  --out ./my_first_dataset
```

This creates:
- `my_first_dataset/users.csv` - 100 synthetic runners
- `my_first_dataset/daily.csv` - Daily data for 90 days
- `my_first_dataset/activities.csv` - Training sessions
- `my_first_dataset/metadata.json` - Dataset information

### Explore the Data

```python
import pandas as pd

# Load data
users = pd.read_csv("my_first_dataset/users.csv")
daily = pd.read_csv("my_first_dataset/daily.csv")

# Basic statistics
print(f"Users: {len(users)}")
print(f"Daily records: {len(daily)}")
print(f"Injury rate: {daily['injury_onset'].sum() / len(daily) * 100:.2f}%")
```

---

## Data Generation

### Command-Line Options

```bash
python stridewise_synth_generate.py --help
```

**Key Options**:
- `--n-users`: Number of users to generate (default: 1000)
- `--n-days`: Number of days per user (default: 365)
- `--seed`: Random seed for reproducibility (default: 42)
- `--out`: Output directory (required)
- `--elite-only`: Generate only advanced/elite users
- `--start-date`: Start date (YYYY-MM-DD format)

### Profile Types

The generator creates four profile types:
- **Novice**: Low training volume, higher injury risk
- **Recreational**: Moderate training, balanced risk
- **Advanced**: High training volume, lower injury risk
- **Elite**: Very high training, optimized risk

### Example: Generate Elite Dataset

```bash
python stridewise_synth_generate.py \
  --n-users 250 \
  --n-days 365 \
  --elite-only \
  --seed 42 \
  --out ./elite_dataset
```

### Example: Generate Mixed Profile Dataset

```bash
python stridewise_synth_generate.py \
  --n-users 1000 \
  --n-days 365 \
  --seed 42 \
  --out ./mixed_dataset
```

---

## Model Training

### Train Main Model

The main model is trained on synthetic data and used for injury prediction.

```bash
python stridewise_train_main_model.py \
  --daily ./training_data/daily.csv \
  --users ./training_data/users.csv \
  --activities ./training_data/activities.csv \
  --out ./models/my_main_model
```

**Output**:
- `my_main_model/main_model_cat.cbm` - Trained model
- `my_main_model/main_model_schema.json` - Feature schema
- `my_main_model/main_model_metrics.json` - Performance metrics

### Train Standalone Model (for Validation)

The standalone model is trained on real CC0 data for validation purposes.

```bash
python stridewise_train_standalone_cc0_paper_method.py \
  --cc0-day ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --cc0-schema ./cc0_feature_schema.json \
  --out ./models/standalone_model
```

---

## Validation

### Step 1: Convert to CC0 Format

```bash
python convert_synth_to_cc0.py \
  --daily ./my_dataset/daily.csv \
  --schema ./cc0_feature_schema.json \
  --out-dir ./my_dataset_cc0
```

### Step 2: Evaluate Against Standalone Model

```bash
python evaluate_synth_cc0.py \
  --synth-cc0 ./my_dataset_cc0/day_approach_maskedID_timeseries.csv \
  --model-dir ./models/standalone_cc0_real \
  --out ./validation_results
```

**Expected Results**:
- ROC AUC: ~0.62-0.63 (good: >0.60, excellent: >0.65)
- PR AUC: ~0.02-0.03
- Performance ratio: ~85-90% of real CC0 performance

### Step 3: Evaluate Main Model

```bash
python evaluate_main_model.py \
  --daily ./my_dataset/daily.csv \
  --users ./my_dataset/users.csv \
  --activities ./my_dataset/activities.csv \
  --model-dir ./models/main_model_large_dataset \
  --out ./main_model_evaluation
```

---

## Web Application

### Start the Web App

```bash
python app.py
```

Access at: `http://localhost:5001`

### Features

- **Data Generation**: Generate synthetic datasets through web interface
- **Model Evaluation**: Evaluate models on new datasets
- **Visualization**: View performance metrics and distributions
- **Configuration**: Adjust generation parameters

### API Endpoints

See `docs/API_REFERENCE.md` for complete API documentation.

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'catboost'`
```bash
pip install catboost
```

**Issue**: `FileNotFoundError: cc0_feature_schema.json`
- Ensure you're running from the correct directory
- Check that `cc0_feature_schema.json` exists in the current directory

**Issue**: Low AUC scores
- Check injury rate (should be ~1.5-2.0%)
- Verify feature distributions match expected ranges
- Review configuration parameters

**Issue**: Memory errors with large datasets
- Reduce `--n-users` or `--n-days`
- Process in batches
- Increase system memory

### Getting Help

1. Check `docs/` directory for detailed documentation
2. Review error messages carefully
3. Verify input data format matches expected schema
4. Check configuration parameters are within valid ranges

---

## Best Practices

### Data Generation

1. **Use Reproducible Seeds**: Always specify `--seed` for reproducibility
2. **Start Small**: Test with small datasets (100 users, 90 days) first
3. **Validate Output**: Check metadata.json for expected statistics
4. **Profile Selection**: Use `--elite-only` for competitive runner scenarios

### Model Training

1. **Large Datasets**: Use 1000+ users for training main model
2. **Time Splits**: Use forward-time validation (default)
3. **Feature Engineering**: Ensure all features are properly engineered
4. **Regularization**: Monitor overfitting gap (<0.05 is good)

### Validation

1. **CC0 Conversion**: Always validate CC0 conversion output
2. **Standalone Model**: Use standalone model for realism validation
3. **Performance Targets**: Aim for >0.60 AUC on standalone model
4. **Distribution Checks**: Compare feature distributions to real data

---

## Advanced Usage

### Custom Configuration

Edit `synthrun_gen/config.py` to customize:
- Injury hazard rates
- Risk factor magnitudes
- Training pattern distributions
- Physiological signal parameters

### Feature Engineering

Customize features in `synthrun_gen/main_model/features.py`:
- Add new rolling window features
- Create interaction features
- Implement custom transformations

### Model Hyperparameters

Adjust hyperparameters in training scripts:
- `reg_lambda`: L2 regularization
- `min_child_weight`: Minimum samples per leaf
- `max_depth`: Tree depth
- `subsample`: Row subsampling

---

**For more details, see**:
- `docs/API_REFERENCE.md` - API documentation
- `docs/CONFIGURATION.md` - Configuration guide
- `docs/VALIDATION_STRATEGY.md` - Validation approach
