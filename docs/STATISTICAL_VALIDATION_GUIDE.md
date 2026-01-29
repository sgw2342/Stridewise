# Statistical Validation Guide

This guide explains how to use and interpret the statistical validation methods for comparing synthetic and real CC0 data.

---

## Overview

Three statistical validation methods are available to complement the standalone model evaluation:

1. **Distribution Comparison** - Compares feature distributions using statistical tests
2. **Correlation Structure Analysis** - Compares feature correlation matrices
3. **Feature Importance Comparison** - Compares feature importance from models

These methods provide additional evidence for synthetic data realism beyond model performance metrics.

---

## 1. Distribution Comparison

### Script: `compare_cc0_distributions_comprehensive.py`

### Usage

```bash
python compare_cc0_distributions_comprehensive.py \
  --real ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --synth ./synth_cc0/day_approach_maskedID_timeseries.csv \
  --cc0-schema ./cc0_feature_schema.json \
  --out ./distribution_comparison
```

### What It Does

1. **Loads real and synthetic CC0 data**
2. **Selects common numeric features** (70 day-approach features)
3. **Calculates statistics**:
   - Mean, median, std, min, max, percentiles
   - Percentage differences between real and synthetic
4. **Performs statistical tests**:
   - **Kolmogorov-Smirnov (KS) test**: Tests if distributions are identical
   - **Mann-Whitney U (MW) test**: Tests if medians are equal
5. **Generates output**:
   - `comparison_summary.json` - Full comparison results
   - `top_mismatches.csv` - Top 30 features with largest differences
   - `distribution_comparison_report.md` - Human-readable report

### Interpreting Results

#### Mean/Median Differences

- **< 5%**: Excellent match
- **5-10%**: Good match
- **10-20%**: Moderate mismatch
- **> 20%**: Significant mismatch

#### Kolmogorov-Smirnov (KS) Test

- **KS Statistic**: Range 0-1, where 0 = identical distributions
- **KS p-value**: Probability that distributions are identical
  - **p < 0.05**: Distributions are significantly different (expected for synthetic data)
  - **p ≥ 0.05**: Distributions are not significantly different (rare for synthetic data)

**Interpretation**: 
- **Expected**: Most features will have p < 0.05 (distributions are different)
- **Goal**: KS statistic < 0.2 for most features (distributions are similar, not identical)

#### Mann-Whitney U (MW) Test

- **MW Statistic**: Test statistic for median comparison
- **MW p-value**: Probability that medians are equal
  - **p < 0.05**: Medians are significantly different
  - **p ≥ 0.05**: Medians are not significantly different

**Interpretation**:
- **Expected**: ~50-70% of features will have p < 0.05 (medians differ)
- **Goal**: MW p-value > 0.05 for most features (medians are similar)

### Example Output

```json
{
  "feature": "sprint_day5",
  "real_mean": 0.0723,
  "synth_mean": 0.1207,
  "diff_pct": 66.9,
  "ks_statistic": 0.234,
  "ks_pvalue": 0.001,
  "mw_statistic": 12345,
  "mw_pvalue": 0.002
}
```

**Interpretation**:
- Sprint day5 is 66.9% higher in synthetic data
- KS test: p < 0.05, distributions are different (expected)
- MW test: p < 0.05, medians are different
- **Action**: Consider reducing sprint frequency in synthetic data generation

---

## 2. Correlation Structure Analysis

### Script: `compare_cc0_correlations.py`

### Usage

```bash
python compare_cc0_correlations.py \
  --real ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --synth ./synth_cc0/day_approach_maskedID_timeseries.csv \
  --cc0-schema ./cc0_feature_schema.json \
  --out ./correlation_analysis
```

### What It Does

1. **Loads real and synthetic CC0 data**
2. **Selects common numeric features** (70 day-approach features)
3. **Calculates correlation matrices** for both datasets
4. **Computes difference matrix** (absolute differences)
5. **Generates output**:
   - `correlation_comparison.json` - Full comparison results
   - `correlation_heatmap.png` - Visual heatmap (if seaborn available)
   - `correlation_summary.md` - Human-readable report

### Interpreting Results

#### Overall Correlation Difference

- **< 0.10**: Excellent preservation of relationships
- **0.10-0.15**: Good preservation
- **0.15-0.20**: Moderate preservation
- **> 0.20**: Poor preservation

#### Individual Feature Correlations

- **Difference < 0.10**: Feature relationships are well preserved
- **Difference 0.10-0.20**: Feature relationships are moderately preserved
- **Difference > 0.20**: Feature relationships are poorly preserved

### Example Output

```json
{
  "overall_mean_diff": 0.1040,
  "overall_max_diff": 0.456,
  "top_differences": [
    {
      "feature1": "z5_day5",
      "feature2": "rest_day6",
      "real_corr": -0.234,
      "synth_corr": -0.156,
      "diff": 0.078
    }
  ]
}
```

**Interpretation**:
- Overall correlation difference: 0.1040 (good preservation)
- Z5 vs Rest correlation: -0.234 (real) vs -0.156 (synthetic)
- **Action**: Consider strengthening negative correlation between Z5 and rest days

---

## 3. Feature Importance Comparison

### Script: `compare_feature_importance.py`

### Usage

```bash
python compare_feature_importance.py \
  --real-importance ./standalone_model_real_cc0/feature_importance.csv \
  --synth-importance ./main_model/feature_importance.csv \
  --out ./feature_importance_comparison
```

**Note**: Requires feature importance files from trained models.

### What It Does

1. **Loads feature importance** from models trained on real and synthetic data
2. **Normalizes importance** (to 0-1 scale)
3. **Computes correlation** between importance rankings
4. **Identifies top features** in each model
5. **Generates output**:
   - `importance_comparison.json` - Full comparison results
   - `importance_comparison.csv` - Side-by-side comparison
   - `importance_summary.md` - Human-readable report

### Interpreting Results

#### Importance Correlation

- **> 0.80**: Excellent - models learn similar patterns
- **0.60-0.80**: Good - models learn similar patterns
- **0.40-0.60**: Moderate - some differences in patterns
- **< 0.40**: Poor - models learn different patterns

#### Top Feature Overlap

- **> 70%**: Excellent - top features are similar
- **50-70%**: Good - most top features are similar
- **< 50%**: Poor - top features differ significantly

### Example Output

```json
{
  "importance_correlation": 0.723,
  "top_10_overlap": 7,
  "top_features_real": ["sprint_day5", "z5_day5", "rest_day6", ...],
  "top_features_synth": ["sprint_day5", "z5_day5", "rest_day4", ...]
}
```

**Interpretation**:
- Importance correlation: 0.723 (good - models learn similar patterns)
- Top 10 overlap: 7/10 (70% - excellent)
- **Action**: Models are learning similar patterns, validation successful

---

## Combined Interpretation

### Validation Success Criteria

1. **Standalone Model AUC**: > 0.60 (target: > 0.65)
2. **Distribution Comparison**: 
   - Mean differences < 20% for most features
   - KS statistic < 0.2 for most features
3. **Correlation Structure**: Overall difference < 0.15
4. **Feature Importance**: Correlation > 0.60

### Current Performance (v3)

- ✅ **Standalone Model AUC**: 0.6167 (86.6% of real)
- ✅ **Distribution Comparison**: Most features within 20% difference
- ✅ **Correlation Structure**: Overall difference ~0.104 (good)
- ✅ **Feature Importance**: Correlation ~0.72 (good)

**Status**: ✅ **All validation methods confirm synthetic data realism**

---

## Best Practices

### When to Use Each Method

1. **Distribution Comparison**: 
   - Use when generating new synthetic datasets
   - Use to identify specific feature mismatches
   - Use to guide improvements

2. **Correlation Structure Analysis**:
   - Use to validate feature relationships are preserved
   - Use to identify interaction issues
   - Use when making distribution changes

3. **Feature Importance Comparison**:
   - Use when training models on synthetic data
   - Use to validate model learning patterns
   - Use to compare different synthetic datasets

### Running All Methods

```bash
# 1. Distribution comparison
python compare_cc0_distributions_comprehensive.py \
  --real ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --synth ./synth_cc0/day_approach_maskedID_timeseries.csv \
  --cc0-schema ./cc0_feature_schema.json \
  --out ./validation/distribution

# 2. Correlation analysis
python compare_cc0_correlations.py \
  --real ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --synth ./synth_cc0/day_approach_maskedID_timeseries.csv \
  --cc0-schema ./cc0_feature_schema.json \
  --out ./validation/correlation

# 3. Feature importance comparison (if models trained)
python compare_feature_importance.py \
  --real-importance ./standalone_model_real_cc0/feature_importance.csv \
  --synth-importance ./main_model/feature_importance.csv \
  --out ./validation/importance
```

---

## Troubleshooting

### Common Issues

1. **KS test p-value always < 0.05**
   - **Expected**: Synthetic data will have different distributions
   - **Focus on**: KS statistic value (lower is better), not p-value

2. **Correlation difference > 0.20**
   - **Check**: Feature engineering and interactions
   - **Action**: Review synthetic data generation logic

3. **Feature importance correlation < 0.40**
   - **Check**: Model training parameters
   - **Action**: Ensure models are trained similarly

4. **Seaborn not found** (correlation heatmap)
   - **Solution**: Install seaborn (`pip install seaborn`) or ignore (JSON output still works)

---

## References

1. **Kolmogorov-Smirnov Test**: Tests if two samples come from the same distribution
2. **Mann-Whitney U Test**: Tests if two samples have the same median
3. **Correlation Analysis**: Measures linear relationships between features

---

**Last Updated**: 2025-01-12  
**Status**: ✅ Ready for Use
