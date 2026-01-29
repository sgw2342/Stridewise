# Model Performance

Complete performance analysis for all models.

## Main Model Performance

**Training Data** (3,000 users, 180 days, 395,693 rows):
- ROC AUC (validation): 0.7136 ✅
- PR AUC (validation): 0.2427
- Brier Score (validation): 0.0973
- Overfitting gap: -0.0023 (validation slightly better - excellent generalization)
- Status: Target met (≥0.70)

## Standalone Model Performance

**Real CC0 Test**:
- ROC AUC: 0.7121
- PR AUC: 0.0450

**Synthetic CC0** (250 elite users):
- ROC AUC: 0.6167 (86.6% of real)
- PR AUC: 0.0216
- Status: Reasonable validation performance

## Feature Importance

Top features in main model:
1. Training load features (7d, 28d)
2. ACWR features
3. Sprinting features
4. Spike features
5. Recovery features

See model files for complete feature importance rankings.
