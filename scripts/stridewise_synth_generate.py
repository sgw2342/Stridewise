#!/usr/bin/env python3
"""CLI: StrideWise synthetic Garmin-like dataset generator.

Examples:
  python stridewise_synth_generate.py --out ./out --n-users 500 --n-days 365 --seed 42 --format csv
  python stridewise_synth_generate.py --out ./out --config ./config.json
"""
from __future__ import annotations
import argparse, json
import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in cast')  # pandas CSV cast warning
from synthrun_gen.config import GeneratorConfig
from synthrun_gen.pipeline import generate_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--config", default=None, help="Optional config JSON (overrides defaults)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-users", type=int, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--n-days", type=int, default=None)
    p.add_argument(
        "--injury-warmup-days",
        type=int,
        default=None,
        help="Prevent injury onsets for the first N days per athlete (stabilises ACWR-like proxies).",
    )
    p.add_argument("--format", type=str, default=None, choices=["csv","parquet","both"])
    p.add_argument("--no-checks", action="store_true", help="Skip sanity checks")
    p.add_argument(
        "--elite-only",
        action="store_true",
        help="Generate only advanced/elite runners (competitive runners, 60+ km/week)"
    )
    return p.parse_args()

def main():
    args = parse_args()
    cfg = GeneratorConfig()
    if args.config:
        cfg = GeneratorConfig.from_json(args.config)

    # apply CLI overrides
    for key, val in {
        "seed": args.seed,
        "n_users": args.n_users,
        "start_date": args.start_date,
        "n_days": args.n_days,
        "injury_warmup_days": args.injury_warmup_days,
        "out_format": args.format,
    }.items():
        if val is not None:
            setattr(cfg, key, val)

    res = generate_dataset(cfg, out_dir=args.out, run_checks=not args.no_checks, elite_only=args.elite_only)
    print("✅ Done.")
    print(f"metadata.json: {res['metadata_path']}")
    for k, v in res["outputs"].items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
