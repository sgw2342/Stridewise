from __future__ import annotations
import numpy as np
import pandas as pd

def _pct_nan(s: pd.Series) -> float:
    return float(s.isna().mean())

def run_sanity_checks(users: pd.DataFrame, daily: pd.DataFrame, activities: pd.DataFrame) -> dict:
    """Lightweight sanity checks designed to catch obvious realism regressions.

    Returns a JSON-serializable dict with metrics + pass/fail flags.
    """
    report = {"summary": {}, "checks": []}

    report["summary"]["users"] = int(users.shape[0])
    report["summary"]["daily_rows"] = int(daily.shape[0])
    report["summary"]["activities"] = int(activities.shape[0])

    # --- Practical audits (non-blocking, but very informative) ---
    # These are *descriptive* stats; we keep pass/fail checks lightweight.
    try:
        from .audit import per_user_forward_time_rate_baseline, label_time_trend

        # Per-user injury rate distribution (are labels mostly "user propensity"?)
        if {"user_id", "injury_next_7d"}.issubset(daily.columns):
            user_rates = (
                daily.groupby("user_id")["injury_next_7d"].mean().astype(float).to_numpy()
            )
            if len(user_rates):
                report["summary"]["injury_rate_by_user"] = {
                    "n_users": int(len(user_rates)),
                    "mean": float(np.mean(user_rates)),
                    "std": float(np.std(user_rates)),
                    "p05": float(np.percentile(user_rates, 5)),
                    "p50": float(np.percentile(user_rates, 50)),
                    "p95": float(np.percentile(user_rates, 95)),
                    "max": float(np.max(user_rates)),
                }

        # Forward-time baseline that predicts each user's train label-rate onto future days
        if {"user_id", "date", "injury_next_7d"}.issubset(daily.columns):
            report["summary"]["user_rate_baseline_forward_time"] = per_user_forward_time_rate_baseline(
                daily, label_col="injury_next_7d", val_frac=0.2
            )

        # Global time trend of injury_next_7d (are labels strongly drifting over the season?)
        if {"date", "injury_next_7d"}.issubset(daily.columns):
            tt = label_time_trend(daily, label_col="injury_next_7d", freq="W")
            if len(tt) >= 6:
                # simple slope proxy: compare first vs last 4 buckets
                first = float(tt["rate"].head(4).mean())
                last = float(tt["rate"].tail(4).mean())
                report["summary"]["injury_time_trend_weekly"] = {
                    "n_buckets": int(len(tt)),
                    "rate_first_4w": first,
                    "rate_last_4w": last,
                    "ratio_last_over_first": float(last / max(first, 1e-9)),
                }
    except Exception:
        # Never fail sanity checks due to audit issues
        pass

    # ranges
    def add_check(name, ok, details):
        report["checks"].append({"name": name, "ok": bool(ok), "details": details})

    if "rhr_bpm" in daily.columns:
        rhr = daily["rhr_bpm"].dropna()
        add_check("RHR range", (rhr.between(35, 110).mean() > 0.995), {
            "min": float(rhr.min()) if len(rhr) else None,
            "max": float(rhr.max()) if len(rhr) else None,
            "pct_in_range": float(rhr.between(35,110).mean()) if len(rhr) else None,
        })

    if "hrv_ms" in daily.columns:
        hrv = daily["hrv_ms"].dropna()
        add_check("HRV range", (hrv.between(10, 220).mean() > 0.995), {
            "min": float(hrv.min()) if len(hrv) else None,
            "max": float(hrv.max()) if len(hrv) else None,
            "pct_in_range": float(hrv.between(10,220).mean()) if len(hrv) else None,
        })

    add_check("ACWR reasonable", (daily["acwr"].between(0.0, 5.0).mean() > 0.99), {
        "p99": float(np.nanpercentile(daily["acwr"], 99)),
        "mean": float(np.nanmean(daily["acwr"])),
    })

    # injury prevalence sanity (not too high)
    inj_rate = float(daily["injury_next_7d"].mean())
    add_check("Injury prevalence", (0.01 <= inj_rate <= 0.20), {"injury_next_7d_rate": inj_rate})

    # missingness checks
    for col, ind in [("rhr_bpm","missing_rhr"), ("hrv_ms","missing_hrv"), ("sleep_hours","missing_sleep"),
                     ("stress_score","missing_stress"), ("resp_rate_rpm","missing_resp"), ("skin_temp_c","missing_temp")]:
        if col in daily.columns and ind in daily.columns:
            pct_nan = _pct_nan(daily[col])
            pct_ind = float(daily[ind].mean())
            # nan should be close to indicator
            add_check(f"Missingness aligned: {col}", abs(pct_nan - pct_ind) < 0.03, {
                "pct_nan": pct_nan, "pct_indicator": pct_ind
            })

    # activity duration sanity
    if len(activities) > 0 and "duration_min" in activities.columns:
        dur = activities["duration_min"]
        add_check("Activity duration positive", (dur.gt(0).mean() > 0.999), {
            "min": float(dur.min()), "p99": float(np.percentile(dur, 99))
        })

    # Merge consistency: all activity user_ids exist in users
    if len(activities) > 0:
        bad = ~activities["user_id"].isin(users["user_id"])
        add_check("Activity user_id in users", (bad.mean() == 0.0), {"n_bad": int(bad.sum())})

    # device worn rate reasonable
    worn = float(daily["device_worn"].mean())
    add_check("Device worn rate", (0.55 <= worn <= 0.98), {"device_worn_rate": worn})

    report["summary"]["ok"] = bool(all(c["ok"] for c in report["checks"]))
    return report
