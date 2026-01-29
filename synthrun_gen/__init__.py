"""StrideWise SynthRun synthetic Garmin-like dataset generator.

Produces:
- users: per-athlete profile table
- daily: per-day signals (load, sleep, HRV/RHR, stress, resp/temp, wear compliance + missingness, labels)
- activities: per-activity sessions (easy/tempo/interval/long)

Designed for ML benchmarking and injury risk prediction.
"""

__all__ = ["generate_dataset"]
from .pipeline import generate_dataset
