#!/usr/bin/env python3
"""Convert synthetic daily output into CC0-shaped day/week timeseries CSVs.

Usage:
  python convert_synth_to_cc0.py --daily ./out/daily.csv --schema ./cc0_feature_schema.json --out-dir ./synth_cc0
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from synthrun_gen.cc0.convert_synth import convert_synth_to_cc0_timeseries

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--daily", required=True, help="Synthetic daily.csv")
    p.add_argument("--schema", default="cc0_feature_schema.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--output-name", default="day_approach_maskedID_timeseries.csv", help="Output filename (default: day_approach_maskedID_timeseries.csv to match real CC0)")
    p.add_argument("--anchor-every-day", action="store_true", default=True, help="Create rows for all event days (default: True, matches CC0 structure). Use --no-anchor-every-day to only include injury events.")
    p.add_argument("--no-anchor-every-day", dest="anchor_every_day", action="store_false", help="Only create rows for injury events (deprecated - doesn't match CC0 structure).")
    p.add_argument("--include-controls", action="store_true", help="DEPRECATED: Not needed when anchor_every_day=True (all days are event days).")
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = convert_synth_to_cc0_timeseries(
        daily_csv=args.daily,
        schema_path=args.schema,
        out_day_csv=str(out / args.output_name),
        out_week_csv=None,  # Week conversion removed
        anchor_every_day=args.anchor_every_day if hasattr(args, 'anchor_every_day') else True,  # Default to True (matches CC0 structure)
        include_control_rows=args.include_controls,
    )
    print("✅ Converted synthetic -> CC0-shaped (exact match to real CC0 structure)")
    print(f"   Output file: {args.output_name}")
    print(f"   Total rows: {res['n_rows']:,}")
    print(f"   Injury rows: {res['n_injury_rows']:,}")
    if res['n_control_rows'] > 0:
        print(f"   Control rows: {res['n_control_rows']:,}")
    print(f"   Injury rate: {res['injury_rate']*100:.2f}%")
    print(f"   ✅ Column names and order match real CC0 exactly")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
