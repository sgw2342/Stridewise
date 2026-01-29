from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

@dataclass
class CC0FeatureSchema:
    version: str
    id_col_candidates: List[str]
    label_col_candidates: List[str]
    row_idx_col: str
    day_window_suffixes: List[str]
    week_window_suffixes: List[str]
    day_features: List[str]
    week_features: List[str]
    subjective_rest_value: float
    ratio_clip_min: float
    ratio_clip_max: float
    ratio_log1p: bool

    @staticmethod
    def from_json(path: str | Path) -> "CC0FeatureSchema":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return CC0FeatureSchema(
            version=d.get("version","1.0"),
            id_col_candidates=d.get("id_col_candidates", ["maskedID"]),
            label_col_candidates=d.get("label_col_candidates", ["y"]),
            row_idx_col=d.get("row_idx_col","row_idx"),
            day_window_suffixes=d["day_window"]["suffixes"],
            week_window_suffixes=d["week_window"]["suffixes"],
            day_features=d.get("day_features", []),
            week_features=d.get("week_features", []),
            subjective_rest_value=float(d.get("subjective_rest_value", -0.01)),
            ratio_clip_min=float(d.get("ratio_transform", {}).get("clip_min", 0.0)),
            ratio_clip_max=float(d.get("ratio_transform", {}).get("clip_max", 4.0)),
            ratio_log1p=bool(d.get("ratio_transform", {}).get("log1p", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # rename to match json structure loosely
        return d
