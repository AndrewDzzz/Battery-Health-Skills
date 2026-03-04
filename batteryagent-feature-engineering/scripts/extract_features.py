#!/usr/bin/env python3
"""Extract 10 mechanism-based battery features per cycle."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _to_numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_ratio(num: float | int, den: float | int) -> float:
    return float(num) / float(den) if den not in (0, 0.0) else float("nan")


def _gradient_abs_mean(values: pd.Series, times: pd.Series) -> float:
    if len(values) < 2:
        return float("nan")
    t = times.to_numpy(dtype=float)
    v = values.to_numpy(dtype=float)
    dt = np.diff(t)
    dv = np.diff(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.abs(np.divide(dv, dt, out=np.zeros_like(dv, dtype=float), where=dt != 0))
    return float(np.nanmean(grad))


def _cc_ratio(volt: pd.Series, times: pd.Series, slope_threshold: float) -> float:
    if len(volt) < 3:
        return float("nan")
    grad = _gradient_abs_mean(volt, times)
    if np.isnan(grad):
        return float("nan")
    t = times.to_numpy(dtype=float)
    v = volt.to_numpy(dtype=float)
    dt = np.diff(t)
    dv = np.abs(np.diff(v))
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.divide(dv, dt, out=np.zeros_like(dv, dtype=float), where=dt != 0)
    return float(np.mean(slope <= slope_threshold))


@dataclass
class FeatureSpec:
    timestamp_col: str
    cycle_col: str | None
    current_col: str | None
    voltage_col: str | None
    pack_voltage_col: str | None
    cell_voltage_col: str | None
    temp_col: str | None
    soc_col: str | None
    cc_slope_threshold: float


def compute_cycle_features(cycle_df: pd.DataFrame, cycle_id: int, spec: FeatureSpec) -> dict[str, float]:
    cdf = cycle_df.sort_values(spec.timestamp_col).copy()
    n = len(cdf)
    initial_n = max(1, int(math.ceil(n * 0.2)))

    time_s = _to_numeric(cdf, spec.timestamp_col)
    current_s = _to_numeric(cdf, spec.current_col)
    voltage_s = _to_numeric(cdf, spec.voltage_col)
    pack_s = _to_numeric(cdf, spec.pack_voltage_col)
    cell_s = _to_numeric(cdf, spec.cell_voltage_col)
    temp_s = _to_numeric(cdf, spec.temp_col)
    soc_s = _to_numeric(cdf, spec.soc_col)

    pack_cell_ratio = float(np.nan)
    if len(pack_s) >= 1 and len(cell_s) >= 1:
        pack_cell_ratio = _safe_ratio(float(pack_s.mean(skipna=True)), float(cell_s.mean(skipna=True)))

    correlation = float(pack_s.corr(cell_s)) if len(pack_s) > 2 and len(cell_s) > 2 else float("nan")
    cc_phase_ratio = _cc_ratio(voltage_s, time_s, spec.cc_slope_threshold) if len(voltage_s) > 2 else float("nan")
    volt_grad = _gradient_abs_mean(voltage_s, time_s) if len(voltage_s) > 1 else float("nan")
    initial_min_voltage = float(voltage_s.iloc[:initial_n].min(skipna=True)) if len(voltage_s) else float("nan")

    max_temp_diff = float(temp_s.max(skipna=True) - temp_s.min(skipna=True)) if len(temp_s) else float("nan")
    terminal_temp = float(temp_s.iloc[-1]) if len(temp_s) else float("nan")
    max_temp_rate = _gradient_abs_mean(temp_s, time_s) if len(temp_s) > 1 else float("nan")

    return {
        "cycle_id": cycle_id,
        "cycle_number": float(cycle_id),
        "cc_phase_ratio": cc_phase_ratio,
        "soc_max": float(soc_s.max(skipna=True)) if len(soc_s) else float("nan"),
        "pack_cell_voltage_ratio": pack_cell_ratio,
        "voltage_correlation": correlation,
        "initial_min_voltage": initial_min_voltage,
        "voltage_gradient": volt_grad,
        "max_temp_diff": max_temp_diff,
        "max_temp_rate": max_temp_rate,
        "terminal_temp": terminal_temp,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BatteryAgent per-cycle feature extractor.")
    parser.add_argument("--input", required=True, help="Input telemetry CSV")
    parser.add_argument("--output", required=True, help="Output feature CSV")
    parser.add_argument("--timestamp-col", default="timestamp", help="Time column name")
    parser.add_argument("--cycle-col", default="", help="Cycle identifier column. Empty when using --samples-per-cycle.")
    parser.add_argument("--samples-per-cycle", type=int, default=0, help="Fallback cycle grouping when no cycle-col exists.")
    parser.add_argument("--current-col", default="", help="Current column")
    parser.add_argument("--voltage-col", default="", help="Primary cell or pack voltage column")
    parser.add_argument("--pack-voltage-col", default="", help="Pack voltage column")
    parser.add_argument("--cell-voltage-col", default="", help="Cell voltage column")
    parser.add_argument("--temp-col", default="", help="Temperature column")
    parser.add_argument("--soc-col", default="", help="SOC column")
    parser.add_argument("--cc-slope-threshold", type=float, default=0.02, help="V/s threshold for constant-current-like regime")
    parser.add_argument("--stats-output", default="", help="Optional path to write per-feature NaN rate and row counts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.input)
    if args.timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {args.timestamp_col}")

    if args.cycle_col:
        if args.cycle_col not in df.columns:
            raise ValueError(f"Missing cycle column: {args.cycle_col}")
        groups = df.groupby(args.cycle_col, sort=True)
        cycle_items = [(int(cid), grp.copy()) for cid, grp in groups]
    else:
        if args.samples_per_cycle <= 0:
            raise ValueError("Use --cycle-col or provide --samples-per-cycle.")
        df = df.copy()
        df["_auto_cycle"] = (np.arange(len(df)) // args.samples_per_cycle).astype(int)
        groups = df.groupby("_auto_cycle", sort=True)
        cycle_items = [(int(cid), grp.copy()) for cid, grp in groups]

    spec = FeatureSpec(
        timestamp_col=args.timestamp_col,
        cycle_col=args.cycle_col or "_auto_cycle",
        current_col=args.current_col or None,
        voltage_col=args.voltage_col or None,
        pack_voltage_col=args.pack_voltage_col or None,
        cell_voltage_col=args.cell_voltage_col or None,
        temp_col=args.temp_col or None,
        soc_col=args.soc_col or None,
        cc_slope_threshold=args.cc_slope_threshold,
    )

    feature_rows = [compute_cycle_features(cycle_df, cid, spec) for cid, cycle_df in cycle_items]
    feature_df = pd.DataFrame(feature_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)

    result = {
        "input_rows": len(df),
        "cycles": len(feature_df),
        "features_per_cycle": max(1, len(feature_df.columns) - 1),
        "output": str(output_path),
    }

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats = {
            "rows": len(feature_df),
            "columns": list(feature_df.columns),
            "nan_rates": feature_df.isna().mean().to_dict(),
            "cycles": len(feature_df),
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        result["stats"] = str(stats_path)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

