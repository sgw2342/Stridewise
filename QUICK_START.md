# Quick Start Guide

Get up and running in 5 minutes.

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## Generate Your First Dataset

```bash
python stridewise_synth_generate.py \
  --n-users 100 \
  --n-days 90 \
  --seed 42 \
  --out ./my_data
```

**Output**: `my_data/users.csv`, `my_data/daily.csv`, `my_data/activities.csv`

---

## Train a Model

```bash
python stridewise_train_main_model.py \
  --daily ./my_data/daily.csv \
  --users ./my_data/users.csv \
  --activities ./my_data/activities.csv \
  --out ./my_model
```

---

## Evaluate a Model

```bash
python evaluate_main_model.py \
  --daily ./my_data/daily.csv \
  --users ./my_data/users.csv \
  --activities ./my_data/activities.csv \
  --model-dir ./models/main_model_large_dataset \
  --out ./results
```

---

## Run Web App

```bash
python app.py
```

Open: `http://localhost:5001`

---

## Next Steps

- Read `README.md` for complete overview
- See `docs/USER_GUIDE.md` for detailed usage
- Check `docs/CONFIGURATION.md` to customize parameters

---

**That's it! You're ready to go.** 🚀
