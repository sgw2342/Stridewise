# API Reference

Complete API documentation for all scripts and modules.

## Data Generation

### `stridewise_synth_generate.py`

Generate synthetic runner data.

**Arguments**:
- `--n-users`: Number of users (default: 1000)
- `--n-days`: Days per user (default: 365)
- `--seed`: Random seed (default: 42)
- `--out`: Output directory (required)
- `--elite-only`: Elite users only (flag)

**Example**:
```bash
python stridewise_synth_generate.py --n-users 500 --n-days 365 --out ./data
```

## Model Training

### `stridewise_train_main_model.py`

Train main injury prediction model.

**Arguments**:
- `--daily`: Daily data CSV (required)
- `--users`: Users CSV (required)
- `--activities`: Activities CSV (required)
- `--out`: Output directory (required)

## Evaluation

### `evaluate_main_model.py`

Evaluate main model on new data.

**Arguments**:
- `--daily`: Daily data CSV (required)
- `--users`: Users CSV (required)
- `--model-dir`: Model directory (required)
- `--out`: Output directory (required)

See individual script `--help` for complete argument lists.
