from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    safe_auc_roc, safe_auc_pr, safe_brier,
    precision_at_top_pct,
    calibration_bins, expected_calibration_error,
    alert_rates, per_athlete_summary, summarize_numeric,
)

def _auto_pick_prob_col(df: pd.DataFrame) -> str:
    for c in ["p", "p_final", "p_sparse", "prob", "proba", "prediction"]:
        if c in df.columns:
            return c
    # fallback: first float-like column excluding id/label
    candidates = [c for c in df.columns if c not in ["row_idx", "y", "label", "athlete_id", "maskedID", "anchor_date"]]
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("Could not infer probability column. Provide --pred-p-col.")

def compute_cc0_report(
    df_truth: pd.DataFrame,
    df_pred: pd.DataFrame,
    row_idx_col: str = "row_idx",
    y_col: str = "y",
    p_col: Optional[str] = None,
    athlete_col: str = "athlete_id",
    thresholds: Optional[Dict[str, Any]] = None,
    top_pcts: Optional[List[float]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    if top_pcts is None:
        top_pcts = [1, 2, 5, 10, 15]

    # Merge truth and predictions
    # Drop 'y' from predictions if it exists (we use truth 'y')
    df_pred_clean = df_pred.drop(columns=[y_col]) if y_col in df_pred.columns else df_pred
    df = df_truth[[row_idx_col, athlete_col, y_col]].merge(df_pred_clean, on=row_idx_col, how="inner")
    
    if len(df) == 0:
        raise ValueError("No rows matched between truth and predictions on row_idx.")

    if p_col is None:
        p_col = _auto_pick_prob_col(df)

    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).to_numpy()
    p = pd.to_numeric(df[p_col], errors="coerce").astype(float).to_numpy()
    p = np.clip(p, 0.0, 1.0)

    prevalence = float(np.mean(y))

    auc_roc = safe_auc_roc(y, p)
    auc_pr = safe_auc_pr(y, p)
    brier = safe_brier(y, p)

    # Precision@k
    p_at = [precision_at_top_pct(y, p, pct) for pct in top_pcts]

    # Calibration bins + ECE
    edges, counts, mean_p, frac_pos = calibration_bins(y, p, n_bins=n_bins)
    ece = expected_calibration_error(counts, mean_p, frac_pos)

    # Alert rates
    alert = None
    if thresholds and ("amber_threshold" in thresholds) and ("red_threshold" in thresholds):
        alert = {
            "thresholds": {
                "amber_threshold": float(thresholds["amber_threshold"]),
                "red_threshold": float(thresholds["red_threshold"]),
                "amber_rate_target": thresholds.get("amber_rate"),
                "red_rate_target": thresholds.get("red_rate"),
            },
            **alert_rates(p, float(thresholds["amber_threshold"]), float(thresholds["red_threshold"]))
        }

    # Per-athlete summaries
    per_a = per_athlete_summary(df[athlete_col].astype(str).to_numpy(), y, p)
    per_a_df = pd.DataFrame(per_a)
    auc_dist = summarize_numeric(per_a_df["auc_roc"].to_numpy(dtype=float))
    brier_dist = summarize_numeric(per_a_df["brier"].to_numpy(dtype=float))
    prev_dist = summarize_numeric(per_a_df["prevalence"].to_numpy(dtype=float))

    report = {
        "n_rows_matched": int(len(df)),
        "prevalence": prevalence,
        "metrics": {
            "roc_auc": auc_roc,
            "pr_auc": auc_pr,
            "brier": brier,
            "ece": ece,
            "prob_col": p_col,
        },
        "precision_at_top_pct": p_at,
        "calibration": {
            "n_bins": int(n_bins),
            "bin_edges": edges.tolist(),
            "bin_counts": counts.tolist(),
            "bin_mean_p": [None if not np.isfinite(x) else float(x) for x in mean_p],
            "bin_frac_pos": [None if not np.isfinite(x) else float(x) for x in frac_pos],
        },
        "per_athlete_distributions": {
            "auc_roc": auc_dist,
            "brier": brier_dist,
            "prevalence": prev_dist,
        },
    }
    if alert is not None:
        report["alerts"] = alert

    return report

def write_cc0_report_artifacts(
    out_dir: str | Path,
    df_truth: pd.DataFrame,
    df_pred: pd.DataFrame,
    report: Dict[str, Any],
    row_idx_col: str = "row_idx",
    y_col: str = "y",
    athlete_col: str = "athlete_id",
    p_col: Optional[str] = None,
    make_plots: bool = True,
    make_pdf: bool = False,
    title: str = "StrideWise CC0 Benchmark Report",
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist JSON + key-value CSV
    (out_dir / "metrics_summary.json").write_text(pd.Series(report).to_json(indent=2), encoding="utf-8")
    # nicer JSON dump
    import json
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # flatten key metrics to CSV
    rows = []
    rows.append(("n_rows_matched", report.get("n_rows_matched")))
    rows.append(("prevalence", report.get("prevalence")))
    m = report.get("metrics", {})
    for k in ["roc_auc", "pr_auc", "brier", "ece", "prob_col"]:
        rows.append((k, m.get(k)))
    if "alerts" in report:
        a = report["alerts"]
        rows.append(("amber_threshold", a["thresholds"]["amber_threshold"]))
        rows.append(("red_threshold", a["thresholds"]["red_threshold"]))
        rows.append(("amber_rate_actual", a.get("amber_rate_actual")))
        rows.append(("red_rate_actual", a.get("red_rate_actual")))
        rows.append(("amber_rate_target", a["thresholds"].get("amber_rate_target")))
        rows.append(("red_rate_target", a["thresholds"].get("red_rate_target")))
    pd.DataFrame(rows, columns=["metric","value"]).to_csv(out_dir / "metrics_table.csv", index=False)

    # calibration bins CSV
    cal = report["calibration"]
    edges = np.asarray(cal["bin_edges"], dtype=float)
    df_bins = pd.DataFrame({
        "bin_low": edges[:-1],
        "bin_high": edges[1:],
        "count": cal["bin_counts"],
        "mean_p": cal["bin_mean_p"],
        "frac_pos": cal["bin_frac_pos"],
    })
    df_bins.to_csv(out_dir / "calibration_bins.csv", index=False)

    # per-athlete CSV
    # rebuild matched frame to compute per athlete table consistent with chosen p_col
    # Drop 'y' from predictions if it exists (we use truth 'y')
    df_pred_clean = df_pred.drop(columns=[y_col]) if y_col in df_pred.columns else df_pred
    df = df_truth[[row_idx_col, athlete_col, y_col]].merge(df_pred_clean, on=row_idx_col, how="inner")
    if p_col is None:
        # infer using same logic as report
        p_col = report["metrics"]["prob_col"]
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce").astype(float).clip(0,1)
    per_a = per_athlete_summary(df[athlete_col].astype(str).to_numpy(),
                               pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).to_numpy(),
                               df[p_col].to_numpy(dtype=float))
    pd.DataFrame(per_a).to_csv(out_dir / "per_athlete_summary.csv", index=False)

    artifacts: Dict[str, str] = {
        "metrics_summary_json": str(out_dir / "metrics_summary.json"),
        "metrics_table_csv": str(out_dir / "metrics_table.csv"),
        "calibration_bins_csv": str(out_dir / "calibration_bins.csv"),
        "per_athlete_summary_csv": str(out_dir / "per_athlete_summary.csv"),
    }

    if make_plots:
        # lightweight plots with matplotlib (no seaborn)
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve

        y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).to_numpy()
        p = df[p_col].to_numpy(dtype=float)

        # ROC
        try:
            fpr, tpr, _ = roc_curve(y, p)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC curve")
            roc_path = out_dir / "roc_curve.png"
            plt.savefig(roc_path, dpi=160, bbox_inches="tight")
            plt.close()
            artifacts["roc_curve_png"] = str(roc_path)
        except Exception:
            pass

        # PR
        try:
            prec, rec, _ = precision_recall_curve(y, p)
            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall curve")
            pr_path = out_dir / "pr_curve.png"
            plt.savefig(pr_path, dpi=160, bbox_inches="tight")
            plt.close()
            artifacts["pr_curve_png"] = str(pr_path)
        except Exception:
            pass

        # Calibration
        plt.figure()
        plt.plot([0,1],[0,1])
        plt.plot(df_bins["mean_p"], df_bins["frac_pos"])
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction positive")
        plt.title("Calibration curve")
        cal_path = out_dir / "calibration_curve.png"
        plt.savefig(cal_path, dpi=160, bbox_inches="tight")
        plt.close()
        artifacts["calibration_curve_png"] = str(cal_path)

        # Histogram
        plt.figure()
        plt.hist(p, bins=30)
        plt.xlabel("Predicted probability")
        plt.ylabel("Count")
        plt.title("Prediction distribution")
        hist_path = out_dir / "p_hist.png"
        plt.savefig(hist_path, dpi=160, bbox_inches="tight")
        plt.close()
        artifacts["p_hist_png"] = str(hist_path)

    if make_pdf:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import ImageReader

            pdf_path = out_dir / "cc0_benchmark_report.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            width, height = A4

            # Title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(2*cm, height - 2*cm, title)

            # Key metrics
            c.setFont("Helvetica", 11)
            y0 = height - 3.2*cm
            lines = [
                f"Rows matched: {report.get('n_rows_matched')}",
                f"Prevalence: {report.get('prevalence'):.4f}" if report.get("prevalence") is not None else "Prevalence: n/a",
                f"ROC AUC: {m.get('roc_auc')}" ,
                f"PR AUC: {m.get('pr_auc')}",
                f"Brier: {m.get('brier')}",
                f"ECE: {m.get('ece')}",
                f"Prob column: {m.get('prob_col')}",
            ]
            for ln in lines:
                c.drawString(2*cm, y0, ln)
                y0 -= 0.55*cm

            # Add plots (if available)
            plot_files = [artifacts.get(k) for k in ["roc_curve_png","pr_curve_png","calibration_curve_png","p_hist_png"] if k in artifacts]
            x = 2*cm
            y_img = y0 - 0.5*cm
            img_w = 8.5*cm
            img_h = 6.0*cm

            # 2x2 grid
            positions = [
                (2*cm, y_img - img_h),
                (2*cm + img_w + 0.7*cm, y_img - img_h),
                (2*cm, y_img - 2*img_h - 0.7*cm),
                (2*cm + img_w + 0.7*cm, y_img - 2*img_h - 0.7*cm),
            ]
            for i, pf in enumerate(plot_files[:4]):
                try:
                    img = ImageReader(pf)
                    px, py = positions[i]
                    c.drawImage(img, px, py, width=img_w, height=img_h, preserveAspectRatio=True, anchor="sw")
                except Exception:
                    continue

            c.showPage()
            c.save()
            artifacts["report_pdf"] = str(pdf_path)
        except Exception:
            # reportlab optional
            pass

    return artifacts
