# Comprehensive Data Validation Visualization Guide

Complete guide to the comprehensive visualization suite for validating synthetic data against real CC0 data.

---

## Overview

The `comprehensive_data_validation_visualizations.py` script creates a complete set of visualizations to validate that synthetic CC0 data matches real CC0 data across multiple dimensions:

1. **Marginal Distributions** - Per-feature histograms and quantiles
2. **Joint Structure** - Correlations, conditional distributions
3. **Time-Series Properties** - Autocorrelation, seasonality, persistence
4. **Physiological Couplings** - Known relationships with lag analysis

---

## Usage

### Basic Usage (CC0 Data Only)

```bash
python validation/comprehensive_data_validation_visualizations.py \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --synth-cc0 ./synth_cc0/day_approach_maskedID_timeseries.csv \
  --out ./validation_visualizations
```

### Full Usage (With Daily Data for Time-Series Analysis)

```bash
python validation/comprehensive_data_validation_visualizations.py \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --synth-cc0 ./synth_cc0/day_approach_maskedID_timeseries.csv \
  --real-daily ./real_daily.csv \
  --synth-daily ./synth_daily.csv \
  --out ./validation_visualizations
```

---

## Generated Visualizations

### 1. Marginal Distributions (`marginal_distributions.png`)

**Purpose**: Compare per-feature distributions between real and synthetic data.

**What it shows**:
- Histograms overlaid for real (blue) and synthetic (purple) data
- Quantile lines (25th, 50th, 75th percentiles) shown as dashed lines
- Mean and standard deviation for each distribution
- Top 20 features by variance

**Interpretation**:
- Overlapping histograms indicate good distribution match
- Quantile lines should align closely
- Mean/standard deviation differences shown in title

**Example**:
```
Feature: total_km
Real: μ=15.2, σ=8.5
Synth: μ=14.8, σ=8.1
```

---

### 2. Quantile Comparison (`quantile_comparison.png`)

**Purpose**: Q-Q plots to compare quantile distributions.

**What it shows**:
- Scatter plot of real quantiles vs synthetic quantiles
- Diagonal line indicating perfect match
- Points on diagonal = good match
- Points above/below diagonal = distribution differences

**Interpretation**:
- Points should cluster around diagonal line
- Systematic deviations indicate distribution shape differences
- Outliers indicate tail differences

---

### 3. Correlation Matrices (`correlation_matrices.png`)

**Purpose**: Compare feature correlation structures.

**What it shows**:
- Three heatmaps side-by-side:
  1. Real data correlation matrix
  2. Synthetic data correlation matrix
  3. Difference matrix (Synthetic - Real)

**Interpretation**:
- Similar correlation patterns = preserved relationships
- Large differences (red/blue in difference matrix) = relationship mismatches
- Correlation difference statistics saved to `correlation_difference_stats.json`

**Key Metrics**:
- Mean absolute difference: Should be < 0.1
- Max absolute difference: Should be < 0.3
- Features with large differences listed in JSON

---

### 4. Conditional Distributions (`conditional_distributions.png`)

**Purpose**: Compare joint distributions and conditional relationships.

**What it shows**:
- Scatter plots of feature pairs
- Conditional means (binned averages) overlaid
- Correlation coefficients for each pair

**Feature Pairs Analyzed**:
- Load features vs Recovery features (e.g., `total_km` vs `resting_hr`)
- Training features vs Physiological features (e.g., `km_total_mean7` vs `lnrmssd`)

**Interpretation**:
- Similar scatter patterns = preserved joint structure
- Aligned conditional means = preserved conditional relationships
- Similar correlation coefficients = preserved linear relationships

---

### 5. Time-Series Autocorrelation (`time_series_autocorrelation.png`)

**Purpose**: Validate temporal persistence and memory in time-series.

**What it shows**:
- Autocorrelation functions (ACF) for key features
- Real vs Synthetic comparison
- Mean ACF across multiple users with confidence bands

**Features Analyzed**:
- `load_trimp` - Training load
- `resting_hr` - Resting heart rate
- `lnrmssd` - HRV (log-transformed)
- `sleep_duration_h` - Sleep duration
- `stress_score_0_100` - Stress score

**Interpretation**:
- Similar ACF patterns = preserved temporal structure
- High autocorrelation at lag 1 = strong day-to-day persistence
- Decay patterns should match (exponential decay typical)

**Expected Patterns**:
- Load: Moderate persistence (ACF ~0.3-0.5 at lag 1)
- RHR: High persistence (ACF ~0.7-0.9 at lag 1)
- HRV: Moderate persistence (ACF ~0.4-0.6 at lag 1)
- Sleep: Moderate persistence (ACF ~0.3-0.5 at lag 1)

---

### 6. Physiological Couplings (`physiological_couplings.png`)

**Purpose**: Validate known physiological relationships with lag analysis.

**What it shows**:
- Cross-correlation functions (CCF) for physiological couplings
- Real vs Synthetic comparison
- Expected lag marked with red dashed line

**Couplings Analyzed**:

1. **Load → RHR (next day)**
   - Expected: Positive correlation at lag 1
   - Physiological basis: High training load increases resting heart rate the next day

2. **Load → HRV (next day)**
   - Expected: Negative correlation at lag 1
   - Physiological basis: High training load decreases HRV the next day

3. **Acute Load → RHR (same day)**
   - Expected: Positive correlation at lag 0
   - Physiological basis: Acute load affects RHR immediately

4. **Sleep → HRV (same day)**
   - Expected: Positive correlation at lag 0
   - Physiological basis: Better sleep improves HRV

**Interpretation**:
- Peak correlation at expected lag = correct physiological coupling
- Similar peak heights = similar coupling strength
- Similar overall patterns = preserved physiological relationships

**Expected Patterns**:
- Load → RHR: Peak at lag 1, correlation ~0.2-0.4
- Load → HRV: Peak at lag 1, correlation ~-0.2 to -0.4
- Sleep → HRV: Peak at lag 0, correlation ~0.3-0.5

---

## Output Files

### Images

1. `marginal_distributions.png` - Feature histograms
2. `quantile_comparison.png` - Q-Q plots
3. `correlation_matrices.png` - Correlation heatmaps
4. `conditional_distributions.png` - Scatter plots with conditional means
5. `time_series_autocorrelation.png` - ACF plots (if daily data provided)
6. `physiological_couplings.png` - Cross-correlation plots (if daily data provided)

### Data Files

1. `correlation_difference_stats.json` - Correlation comparison statistics
   ```json
   {
     "mean_absolute_difference": 0.05,
     "max_absolute_difference": 0.23,
     "features_with_large_diff": [
       {
         "feature1": "total_km",
         "feature2": "resting_hr",
         "real_corr": 0.15,
         "synth_corr": 0.28,
         "difference": 0.13
       }
     ]
   }
   ```

---

## Requirements

### Python Packages

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

### Data Requirements

**Required**:
- Real CC0 data file
- Synthetic CC0 data file

**Optional** (for time-series analysis):
- Real daily data file
- Synthetic daily data file

**Note**: Daily data enables time-series and physiological coupling analysis. Without it, only marginal and joint structure analysis is performed.

---

## Interpreting Results

### Good Match Indicators

✅ **Marginal Distributions**:
- Overlapping histograms
- Quantile lines aligned
- Mean differences < 10%
- Standard deviation differences < 15%

✅ **Correlation Structure**:
- Mean absolute difference < 0.1
- Max absolute difference < 0.3
- Key relationships preserved (difference < 0.15)

✅ **Time-Series Properties**:
- Similar ACF patterns
- Similar decay rates
- Similar persistence levels

✅ **Physiological Couplings**:
- Peak correlations at expected lags
- Similar peak heights (±0.1)
- Similar overall patterns

### Warning Signs

⚠️ **Large Distribution Differences**:
- Mean differences > 20%
- Standard deviation differences > 30%
- Systematic quantile misalignment

⚠️ **Correlation Mismatches**:
- Mean absolute difference > 0.15
- Key relationships differ by > 0.2
- Systematic correlation differences

⚠️ **Time-Series Mismatches**:
- Very different ACF patterns
- Different persistence levels
- Missing expected autocorrelation

⚠️ **Physiological Coupling Issues**:
- Peak at wrong lag
- Peak height differs by > 0.2
- Missing expected relationships

---

## Integration with Other Validation Tools

This visualization suite complements:

1. **Statistical Validation** (`STATISTICAL_VALIDATION_GUIDE.md`)
   - Provides visual confirmation of statistical test results

2. **Distribution Comparison** (`compare_cc0_distributions_comprehensive.py`)
   - Visualizes results from comprehensive distribution comparison

3. **Correlation Analysis** (`compare_cc0_correlations.py`)
   - Provides detailed correlation heatmaps

4. **Performance Evaluation** (`evaluate_synth_cc0.py`)
   - Visual validation supports model performance metrics

---

## Best Practices

1. **Run after major changes**: Always validate after modifying data generation
2. **Compare systematically**: Use same real data baseline for all comparisons
3. **Document findings**: Keep records of visualization results
4. **Monitor trends**: Track visualization metrics over time
5. **Address issues**: Fix identified mismatches before proceeding

---

## Troubleshooting

### Issue: "No valid features for correlation analysis"

**Solution**: Check that CC0 files have numeric features. Ensure feature columns are not all excluded.

### Issue: "Missing columns" errors

**Solution**: Verify that both real and synthetic data have the same column structure. Check for column name mismatches.

### Issue: Time-series plots are empty

**Solution**: Ensure daily data files are provided and contain required columns (`user_id`, `date_index`, feature columns).

### Issue: Physiological coupling plots show no correlation

**Solution**: 
- Verify features exist in daily data
- Check that users have sufficient data (minimum 30 days)
- Ensure date columns are properly sorted

---

## Example Workflow

```bash
# 1. Generate synthetic data
python stridewise_synth_generate.py --n-users 1000 --out ./synth_data

# 2. Convert to CC0
python convert_synth_to_cc0.py --daily ./synth_data/daily.csv --out-dir ./synth_cc0

# 3. Run comprehensive visualizations
python validation/comprehensive_data_validation_visualizations.py \
  --real-cc0 ./cc0_competitive_runners/day_approach_maskedID_timeseries.csv \
  --synth-cc0 ./synth_cc0/day_approach_maskedID_timeseries.csv \
  --real-daily ./real_daily.csv \
  --synth-daily ./synth_data/daily.csv \
  --out ./validation_viz

# 4. Review visualizations
open ./validation_viz/*.png

# 5. Check statistics
cat ./validation_viz/correlation_difference_stats.json
```

---

**For more details, see**:
- `STATISTICAL_VALIDATION_GUIDE.md` - Statistical validation methods
- `VALIDATION_AND_VISUALIZATION.md` - Overview of all validation tools
