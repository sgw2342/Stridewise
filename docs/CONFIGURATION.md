# Configuration Guide

Complete guide to configuration parameters.

## Injury Hazards

Base injury probabilities by profile:
- `injury_hazard_novice`: 0.00624
- `injury_hazard_recreational`: 0.00492
- `injury_hazard_advanced`: 0.00115
- `injury_hazard_elite`: 0.00138

## Risk Factors

### Spike Absolute Risk (Long-Run Spikes)
- `spike_absolute_risk_small`: 0.04331
- `spike_absolute_risk_moderate`: 0.04950
- `spike_absolute_risk_large`: 0.06188

### Sprinting Absolute Risk
- `sprinting_absolute_risk_per_km`: 0.1381 (default)
- `sprinting_absolute_risk_per_km_advanced_elite`: 0.060

## Training Patterns

See `synthrun_gen/config.py` for complete parameter list.

## Modifying Configuration

Edit `synthrun_gen/config.py` directly, then regenerate datasets.
