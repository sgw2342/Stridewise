# Main Model Performance Visualization Guide

Complete guide to visualizing main model performance.

---

## Overview

The `create_main_model_visualizations.py` script creates comprehensive visualizations to demonstrate main model performance across multiple dimensions.

---

## Usage

### Basic Usage

```bash
python validation/create_main_model_visualizations.py \
  --model-dir ./models/main_model_large_dataset \
  --out ./main_model_visualizations
```

### With Predictions

```bash
# First, generate predictions
python evaluate_main_model.py \
  --model-dir ./models/main_model_large_dataset \
  --daily ./large_dataset/daily.csv \
  --users ./large_dataset/users.csv \
  --out ./eval_results

# Then create visualizations
python validation/create_main_model_visualizations.py \
  --model-dir ./models/main_model_large_dataset \
  --predictions ./eval_results/predictions.csv \
  --out ./main_model_visualizations
```

---

## Generated Visualizations

### 1. Performance Summary (`performance_summary.png`)

**Purpose**: Comprehensive dashboard showing all key metrics.

**What it shows**:
- **Performance Metrics**: Train vs Validation (ROC AUC, PR AUC, Brier Score)
- **Overfitting Check**: Gap between train and validation performance
- **Dataset Statistics**: Total, train, and validation row counts
- **Label Prevalence**: Injury rate in train vs validation sets

**Interpretation**:
- Similar train/val metrics = good generalization
- Small overfitting gap (<0.05) = well-controlled
- Balanced prevalence = consistent data distribution

---

### 2. ROC Curve (`roc_curve.png`)

**Purpose**: Shows discrimination ability of the model.

**What it shows**:
- ROC curve with AUC value
- Random classifier baseline (diagonal line)
- Validation AUC from training metrics

**Interpretation**:
- Curve above diagonal = better than random
- Higher curve = better discrimination
- AUC > 0.70 = excellent performance
- AUC 0.60-0.70 = good performance

---

### 3. Precision-Recall Curve (`pr_curve.png`)

**Purpose**: Shows precision-recall trade-off (critical for imbalanced data).

**What it shows**:
- PR curve with PR-AUC value
- Baseline (prevalence line)
- Validation PR-AUC from training metrics

**Interpretation**:
- Higher curve = better precision-recall balance
- PR-AUC > 0.20 = good for imbalanced data
- More informative than ROC-AUC for rare events

---

### 4. Calibration Curve (`calibration_curve.png`)

**Purpose**: Shows how well-calibrated the predicted probabilities are.

**What it shows**:
- Predicted probability vs actual fraction of positives
- Perfect calibration line (diagonal)
- Expected Calibration Error (ECE)
- Bin counts for each calibration bin

**Interpretation**:
- Points on diagonal = well-calibrated
- ECE < 0.05 = excellent calibration
- ECE 0.05-0.10 = good calibration
- Points above diagonal = overconfident
- Points below diagonal = underconfident

---

### 5. Feature Importance (`feature_importance.png`)

**Purpose**: Shows which features the model considers most important.

**What it shows**:
- Top 20 features by importance (gain)
- Horizontal bar chart
- Importance values labeled on bars

**Interpretation**:
- Higher importance = more predictive
- Top features drive model decisions
- Useful for feature selection and interpretation

---

### 6. Prediction Distributions (`prediction_distributions.png`)

**Purpose**: Shows how predictions differ between positive and negative classes.

**What it shows**:
- **Left**: Histograms of predictions for each class
- **Right**: Box plots comparing distributions
- Mean values for each class

**Interpretation**:
- Good separation = positive class has higher predictions
- Overlapping distributions = model struggles to distinguish
- Clear separation = model is learning signal

---

### 7. Threshold Analysis (`threshold_analysis.png`)

**Purpose**: Shows how performance metrics change with different classification thresholds.

**What it shows**:
- **Top**: Precision, Recall, F1 Score vs threshold
- **Bottom**: Specificity vs threshold
- Optimal F1 threshold marked

**Interpretation**:
- Higher threshold = fewer positives, higher precision
- Lower threshold = more positives, higher recall
- Optimal threshold balances precision and recall
- Use to select threshold for production deployment

---

### 8. Confusion Matrices (`confusion_matrices.png`)

**Purpose**: Shows classification performance at different thresholds.

**What it shows**:
- Confusion matrices at multiple thresholds (0.1, optimal F1, 0.3, 0.5)
- True/False Positives and Negatives
- Heatmap visualization

**Interpretation**:
- True Positives: Correctly predicted injuries
- False Positives: False alarms
- False Negatives: Missed injuries
- True Negatives: Correctly predicted non-injuries
- Use to understand trade-offs at different thresholds

---

## Output Files

All visualizations are saved as high-resolution PNG files (150 DPI):

1. `performance_summary.png` - Comprehensive dashboard
2. `roc_curve.png` - ROC curve
3. `pr_curve.png` - Precision-recall curve
4. `calibration_curve.png` - Calibration plot
5. `feature_importance.png` - Top features
6. `prediction_distributions.png` - Prediction histograms
7. `threshold_analysis.png` - Threshold performance
8. `confusion_matrices.png` - Confusion matrices

---

## Requirements

### Python Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Input Files

**Required**:
- `main_model_metrics.json` - Model metrics (from training)

**Optional** (for full visualizations):
- `predictions.csv` - Model predictions (from evaluation)
- `main_model_feature_importance.csv` - Feature importance (from training)

---

## Interpreting Results

### Good Performance Indicators

✅ **ROC AUC > 0.70**: Excellent discrimination
✅ **PR AUC > 0.20**: Good precision-recall balance
✅ **ECE < 0.05**: Well-calibrated probabilities
✅ **Overfitting Gap < 0.05**: Good generalization
✅ **Clear Prediction Separation**: Model distinguishes classes well

### Warning Signs

⚠️ **ROC AUC < 0.60**: May need improvement
⚠️ **PR AUC < 0.10**: Low precision-recall performance
⚠️ **ECE > 0.10**: Poor calibration
⚠️ **Overfitting Gap > 0.10**: High overfitting risk
⚠️ **Overlapping Distributions**: Model struggles to distinguish classes

---

## Example Workflow

```bash
# 1. Train model (if not already done)
python stridewise_train_main_model.py \
  --daily ./large_dataset/daily.csv \
  --users ./large_dataset/users.csv \
  --out ./models/main_model_large_dataset

# 2. Evaluate model (to get predictions)
python evaluate_main_model.py \
  --model-dir ./models/main_model_large_dataset \
  --daily ./large_dataset/daily.csv \
  --users ./large_dataset/users.csv \
  --out ./eval_results

# 3. Create visualizations
python validation/create_main_model_visualizations.py \
  --model-dir ./models/main_model_large_dataset \
  --predictions ./eval_results/predictions.csv \
  --out ./main_model_visualizations

# 4. Review visualizations
open ./main_model_visualizations/*.png
```

---

## Best Practices

1. **Always include predictions**: Full visualizations require predictions
2. **Compare iterations**: Re-run to compare different model versions
3. **Document findings**: Keep records of visualization results
4. **Use for presentations**: High-resolution images suitable for reports
5. **Monitor overfitting**: Check overfitting gap regularly

---

**For more details, see**:
- `VISUALIZATION_GUIDE.md` - General visualization guide
- `MODEL_PERFORMANCE.md` - Performance metrics documentation
