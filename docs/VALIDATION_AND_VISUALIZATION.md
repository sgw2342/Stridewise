# Validation & Visualization Tools

Complete guide to statistical validation and visualization tools.

---

## Overview

The production folder includes comprehensive tools for:
- **Statistical Validation**: Validate data quality and realism
- **Visualization**: Create performance and distribution visualizations
- **Comparison Analysis**: Compare synthetic vs real data

---

## Validation Scripts

### 1. Main Model Performance Visualizations ⭐ NEW

**Script**: `validation/create_main_model_visualizations.py`

Creates comprehensive visualizations demonstrating main model performance:
- ROC and PR curves
- Calibration curves
- Feature importance
- Prediction distributions
- Threshold analysis
- Confusion matrices
- Performance summary dashboard

```bash
python validation/create_main_model_visualizations.py \
  --model-dir ./models/main_model_large_dataset \
  --predictions ./eval_results/predictions.csv \
  --out ./main_model_visualizations
```

See `MAIN_MODEL_VISUALIZATION_GUIDE.md` for complete documentation.

---

### 2. Comprehensive Data Validation Visualizations

**Script**: `validation/comprehensive_data_validation_visualizations.py`

Creates comprehensive visualizations comparing synthetic vs real CC0 data across:
- Marginal distributions (histograms, quantiles)
- Joint structure (correlations, conditional distributions)
- Time-series properties (autocorrelation, persistence)
- Physiological couplings (lag relationships)

```bash
python validation/comprehensive_data_validation_visualizations.py \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --synth-cc0 ./synth_cc0/day_approach_maskedID_timeseries.csv \
  --real-daily ./real_daily.csv \
  --synth-daily ./synth_daily.csv \
  --out ./validation_visualizations
```

See `COMPREHENSIVE_VISUALIZATION_GUIDE.md` for complete documentation.

---

### 3. Performance Visualizations

**Script**: `validation/create_performance_visualizations.py`

Creates visualizations of model performance metrics.

```bash
python validation/create_main_model_visualizations.py \
  --model-dir ./models/main_model_large_dataset \
  --predictions ./eval_results/predictions.csv \
  --out ./main_model_visualizations
```

**Output**:
- ROC curves
- Precision-recall curves
- Feature importance plots
- Performance comparison charts

---

### 3. Injury Distribution Analysis

**Script**: `validation/analyze_injury_distribution_comparison.py`

Compares injury distributions between synthetic and real CC0 data.

```bash
python validation/analyze_injury_distribution_comparison.py
```

**Output**:
- Injury rate comparisons
- Injuries per athlete statistics
- Injury driver associations
- Distribution comparisons

---

### 4. Profile-Based Analysis

**Script**: `validation/analyze_injury_distribution_by_profile.py`

Analyzes injury patterns by athlete profile (novice, recreational, advanced, elite).

```bash
python validation/analyze_injury_distribution_by_profile.py \
  --synth-cc0 ./synth_data_cc0/day_approach_maskedID_timeseries.csv \
  --synth-users ./synth_data/users.csv \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --out ./profile_analysis.json
```

**Output**:
- Injury rates by profile
- Injuries per athlete by profile
- Injury drivers by profile
- Comparison with real data

---

### 5. Training Pattern Analysis

**Script**: `validation/analyze_cc0_training_patterns.py`

Analyzes training patterns (training blocks, easy weeks, athlete grouping).

```bash
python validation/analyze_cc0_training_patterns.py \
  --synth-cc0 ./synth_data_cc0/day_approach_maskedID_timeseries.csv \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv
```

**Output**:
- Training block patterns
- Easy week frequency
- Athlete grouping analysis
- Pattern comparisons

---

### 6. Distribution Comparisons

**Script**: `validation/compare_all_distributions.py`

Comprehensive comparison of feature distributions.

```bash
python validation/compare_all_distributions.py \
  --synth ./synth_data/daily.csv \
  --real ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --out ./distribution_comparison
```

**Output**:
- Feature distribution statistics
- Statistical tests (KS test, Mann-Whitney U)
- Distribution plots
- Mismatch reports

---

### 7. Correlation Analysis

**Script**: `validation/compare_cc0_correlations.py`

Compares correlations between features in synthetic vs real data.

```bash
python validation/compare_cc0_correlations.py \
  --synth-cc0 ./synth_data_cc0/day_approach_maskedID_timeseries.csv \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv
```

---

### 8. Injury Causation Analysis

**Script**: `validation/compare_cc0_injury_causation.py`

Analyzes how features are associated with injury in synthetic vs real data.

```bash
python validation/compare_cc0_injury_causation.py \
  --synth-cc0 ./synth_data_cc0/day_approach_maskedID_timeseries.csv \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv
```

**Output**:
- Feature associations with injury
- Percentage differences (injury vs non-injury days)
- Association strength comparisons

---

### 9. Feature Importance Comparison

**Script**: `validation/compare_feature_importance.py`

Compares feature importance between models.

```bash
python validation/compare_feature_importance.py \
  --model1 ./models/main_model_large_dataset \
  --model2 ./models/standalone_cc0_real
```

---

## Documentation

### Statistical Validation Guide

**File**: `docs/STATISTICAL_VALIDATION_GUIDE.md`

Complete guide to:
- Statistical validation methods
- Data quality checks
- Sanity checks
- Validation workflows

### Visualization Guides

**Files**: 
- `docs/VISUALIZATION_GUIDE.md` - Performance visualization guide
- `docs/COMPREHENSIVE_VISUALIZATION_GUIDE.md` - Comprehensive data validation visualization guide ⭐ NEW

Complete guides to:
- Creating visualizations
- Performance plots
- Distribution plots
- Comparison charts
- Marginal and joint distributions
- Time-series properties
- Physiological couplings

### Visualization Summary

**File**: `docs/VISUALIZATION_SUMMARY.md`

Summary of available visualizations and their use cases.

---

## Quick Reference

### Common Validation Workflow

```bash
# 1. Generate data
python stridewise_synth_generate.py --n-users 1000 --out ./data

# 2. Convert to CC0
python convert_synth_to_cc0.py --daily ./data/daily.csv --out-dir ./data_cc0

# 3. Compare distributions
python validation/compare_all_distributions.py \
  --synth ./data/daily.csv \
  --real ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv

# 4. Analyze injury patterns
python validation/analyze_injury_distribution_by_profile.py \
  --synth-cc0 ./data_cc0/day_approach_maskedID_timeseries.csv \
  --synth-users ./data/users.csv \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv

# 5. Create visualizations
python validation/create_main_model_visualizations.py \
  --model-dir ./models/main_model_large_dataset \
  --predictions ./eval_results/predictions.csv \
  --out ./main_model_visualizations
```

---

## Integration with Sanity Checks

The `synthrun_gen/sanity.py` module provides built-in sanity checks:
- Injury rate validation
- Feature distribution checks
- Temporal pattern validation
- Statistical consistency checks

These are automatically run during data generation.

---

## Best Practices

1. **Run validation after major changes**: Always validate after modifying configuration
2. **Compare against baselines**: Use real CC0 data as baseline for comparisons
3. **Document findings**: Keep records of validation results
4. **Monitor trends**: Track validation metrics over time
5. **Use multiple methods**: Combine statistical tests with visualizations

---

**For more details, see**:
- `docs/STATISTICAL_VALIDATION_GUIDE.md` - Complete validation guide
- `docs/VISUALIZATION_GUIDE.md` - Visualization guide
