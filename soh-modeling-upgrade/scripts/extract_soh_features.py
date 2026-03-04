#!/usr/bin/env python3
"""Extract mechanism-aware SOH-oriented cycle features from battery telemetry."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SOH-oriented cycle features.")
    parser.add_argument("--input", required=True, help="Input telemetry CSV")
    parser.add_argument("--output", required=True, help="Output feature CSV")
    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp/time column")
    parser.add_argument("--cycle-col", default="", help="Cycle column, empty if samples-per-cycle is used")
    parser.add_argument("--samples-per-cycle", type=int, default=0, help="Fallback cycle sizing")
    parser.add_argument("--current-col", default="", help="Current column")
    parser.add_argument("--voltage-col", default="", help="Voltage column")
    parser.add_argument("--temp-col", default="", help="Temperature column")
    parser.add_argument("--soc-col", default="", help="SOC column")
    parser.add_argument("--soh-proxy-col", default="", help="Column containing direct SOH/capacity proxy label")
    parser.add_argument("--cc-slope-threshold", type=float, default=0.015, help="V/s threshold for CC-like segments")
    parser.add_argument("--cv-slope-threshold", type=float, default=0.003, help="V/s threshold for CV-like segments")
    parser.add_argument("--cv-quantile", type=float, default=0.90, help="High-voltage quantile defining CV region")
    parser.add_argument("--stats-output", default="", help="Optional JSON with NaN rates and cycle summaries")
    return parser.parse_args()


def _to_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _finite(series: pd.Series) -> pd.Series:
    return series[np.isfinite(series)]


def _integral(values: np.ndarray, times: np.ndarray) -> float:
    if len(values) < 2 or len(times) < 2:
        return float("nan")
    return float(np.trapz(values, times))


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and den not in (0, 0.0) else float("nan")


def _gradient_abs_mean(values: np.ndarray, times: np.ndarray) -> float:
    if len(values) < 3 or len(times) < 3:
        return float("nan")
    dt = np.diff(times)
    dv = np.abs(np.diff(values))
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.divide(dv, dt, out=np.zeros_like(dv, dtype=float), where=np.abs(dt) > 0)
    return float(np.nanmean(np.abs(grad)))


def _frac_below_slope(values: np.ndarray, times: np.ndarray, threshold: float) -> float:
    if len(values) < 3 or len(times) < 3:
        return float("nan")
    dt = np.diff(times)
    dv = np.diff(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.abs(np.divide(dv, dt, out=np.zeros_like(dv, dtype=float), where=np.abs(dt) > 0))
    return float(np.mean(slope <= threshold)) if len(slope) else float("nan")


def _cv_region_ratio(values: np.ndarray, times: np.ndarray, threshold: float, cv_quantile: float) -> float:
    if len(values) < 3 or len(times) < 3:
        return float("nan")
    q = np.nanquantile(values, cv_quantile)
    mask = values[:-1] >= q
    if not np.any(mask):
        return float("nan")
    dt = np.diff(times)
    dv = np.abs(np.diff(values))
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.divide(dv, dt, out=np.zeros_like(dv, dtype=float), where=np.abs(dt) > 0)
    return float(np.mean(np.abs(slope[mask]) <= threshold))


def _rate_features(values: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    if len(values) < 3 or len(times) < 3:
        return float("nan"), float("nan")
    rate = np.diff(values) / np.diff(times)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(rate)
        if not np.any(valid):
            return float("nan"), float("nan")
        rate = rate[valid]
    return float(np.max(np.abs(rate))), float(np.nanmean(np.abs(rate)))


def _dqdv_features(voltage: np.ndarray, current: np.ndarray, times: np.ndarray) -> tuple[float, float, float, float]:
    if len(voltage) < 4 or len(current) < 4 or len(times) < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")

    v = _finite(pd.Series(voltage))
    i = _finite(pd.Series(current))
    t = _finite(pd.Series(times))
    if len(v) < 4 or len(i) < 4 or len(t) < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")

    merged = pd.DataFrame({"v": voltage, "i": current, "t": times}).dropna()
    if len(merged) < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")

    merged = merged.sort_values("t")
    v = merged["v"].to_numpy(dtype=float)
    i = merged["i"].to_numpy(dtype=float)
    t = merged["t"].to_numpy(dtype=float)

    if np.all(~np.isfinite(v)) or np.all(~np.isfinite(i)):
        return float("nan"), float("nan"), float("nan"), float("nan")

    dt = np.diff(t)
    if np.all(np.abs(dt) < 1e-9):
        return float("nan"), float("nan"), float("nan"), float("nan")

    dq = -0.5 * (i[:-1] + i[1:]) * dt / 3600.0
    q = np.r_[0.0, np.cumsum(dq)]
    dv = np.diff(v)
    q_mid = 0.5 * (q[:-1] + q[1:])
    v_mid = 0.5 * (v[:-1] + v[1:])

    if len(dv) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    valid_dv = np.abs(dv) > 1e-6
    if not np.any(valid_dv):
        return float("nan"), float("nan"), float("nan"), float("nan")

    dqdv = dq[1:] / dv
    dqdv = dqdv[1:]
    valid = np.isfinite(dqdv)
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan"), float("nan")

    dqdv_abs = np.abs(dqdv[valid])
    q_mid_valid = q_mid[1:][valid]
    v_mid_valid = v_mid[valid]
    peak_idx = int(np.argmax(dqdv_abs))
    peak_val = float(dqdv_abs[peak_idx])
    q_at_peak = float(q_mid_valid[peak_idx])
    mean_abs = float(np.mean(dqdv_abs))
    spread = float(v_mid_valid.max() - v_mid_valid.min()) if len(v_mid_valid) else float("nan")
    return peak_val, q_at_peak, mean_abs, spread


def _resistance_proxy(voltage: np.ndarray, current: np.ndarray) -> float:
    if len(voltage) < 3 or len(current) < 3:
        return float("nan")
    dv = np.diff(voltage)
    di = np.diff(current)
    valid = np.isfinite(dv) & np.isfinite(di) & (np.abs(di) > 1e-3)
    if not np.any(valid):
        return float("nan")
    ratios = np.abs(dv[valid] / di[valid])
    return float(median(ratios.tolist()))


@dataclass
class FeatureSpec:
    timestamp_col: str
    current_col: str | None
    voltage_col: str | None
    temp_col: str | None
    soc_col: str | None
    soh_proxy_col: str | None
    cc_slope_threshold: float
    cv_slope_threshold: float
    cv_quantile: float


def compute_cycle_features(cycle_id: int, cycle_df: pd.DataFrame, spec: FeatureSpec) -> dict[str, Any]:
    df = cycle_df.sort_values(spec.timestamp_col).copy()
    n = len(df)
    if n == 0:
        return {"cycle_id": cycle_id}

    t_raw = _to_numeric(df.get(spec.timestamp_col, pd.Series()))
    t = t_raw.to_numpy(dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return {"cycle_id": cycle_id, "cycle_points": n}

    duration = float(t[-1] - t[0]) if len(t) else float("nan")
    time_span = df[spec.timestamp_col].nunique()

    current = _to_numeric(df.get(spec.current_col))
    voltage = _to_numeric(df.get(spec.voltage_col))
    temp = _to_numeric(df.get(spec.temp_col))
    soc = _to_numeric(df.get(spec.soc_col))

    i = current.to_numpy(dtype=float)
    v = voltage.to_numpy(dtype=float)
    ti = t_raw.to_numpy(dtype=float)

    v_clean = _finite(voltage).to_numpy(dtype=float)
    t_for_v = _finite(pd.Series(ti)).reindex(voltage.index).to_numpy()

    v_slope_ratio = _frac_below_slope(v_clean, t_for_v[: len(v_clean)], spec.cc_slope_threshold) if len(v_clean) >= 2 else float("nan")
    v_cv_ratio = _cv_region_ratio(v_clean, t_for_v[: len(v_clean)], spec.cv_slope_threshold, spec.cv_quantile) if len(v_clean) >= 2 else float("nan")

    dq_dv_peak, q_at_peak, dq_dv_mean, dq_dv_spread = _dqdv_features(v, i, ti)
    dqdv_ratio = _safe_ratio(dq_dv_peak, dq_dv_mean)

    v_range = float(np.nanmax(v) - np.nanmin(v)) if len(v_clean) else float("nan")
    v_mean = float(np.nanmean(v)) if len(v_clean) else float("nan")
    v_std = float(np.nanstd(v)) if len(v_clean) else float("nan")

    temp_range = float(np.nanmax(temp) - np.nanmin(temp)) if len(temp) else float("nan")
    temp_max_rate, temp_mean_rate = _rate_features(temp.to_numpy(dtype=float), ti)

    soc_min = float(np.nanmin(soc)) if len(soc) else float("nan")
    soc_max = float(np.nanmax(soc)) if len(soc) else float("nan")
    soc_span = soc_max - soc_min if all(np.isfinite([soc_min, soc_max])) else float("nan")

    capacity_ah = float(-_integral(i, ti) / 3600.0) if np.isfinite(i).sum() > 1 and np.isfinite(ti).sum() > 1 else float("nan")
    energy_wh = float(_integral(v * i, ti) / 3600.0) if np.isfinite(v).sum() > 1 and np.isfinite(i).sum() > 1 and np.isfinite(ti).sum() > 1 else float("nan")

    resistance_proxy = _resistance_proxy(v, i)
    imp = float(np.nanmax(temp)) if len(temp) else float("nan")

    return {
        "cycle_id": cycle_id,
        "cycle_points": int(n),
        "time_span_sec": float(duration),
        "time_unique_points": int(time_span),
        "capacity_ah": capacity_ah,
        "energy_wh": energy_wh,
        "voltage_mean": v_mean,
        "voltage_std": v_std,
        "voltage_range": v_range,
        "cc_ratio": v_slope_ratio,
        "cv_ratio": v_cv_ratio,
        "temp_range": temp_range,
        "temp_max_abs_rate": temp_max_rate,
        "temp_mean_abs_rate": temp_mean_rate,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "soc_span": soc_span,
        "dqdv_peak_abs": dq_dv_peak,
        "dqdv_peak_q": q_at_peak,
        "dqdv_mean_abs": dq_dv_mean,
        "dqdv_spread_v": dq_dv_spread,
        "dqdv_peak_to_mean_ratio": dqdv_ratio,
        "resistance_proxy": resistance_proxy,
        "terminal_temp": float(temp.iloc[-1]) if len(temp) else float("nan"),
        "terminal_voltage": float(v[-1]) if len(v) else float("nan"),
        "terminal_soc": float(soc.iloc[-1]) if len(soc) else float("nan"),
        "terminal_ohmic_proxy": imp,
    }


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    if args.timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {args.timestamp_col}")

    if args.cycle_col:
        if args.cycle_col not in df.columns:
            raise ValueError(f"Missing cycle column: {args.cycle_col}")
        cycle_iter = [(int(cycle_id), g) for cycle_id, g in df.groupby(args.cycle_col, sort=True)]
    else:
        if args.samples_per_cycle <= 0:
            raise ValueError("Use --cycle-col or --samples-per-cycle.")
        df = df.copy()
        df["_auto_cycle"] = (np.arange(len(df)) // args.samples_per_cycle).astype(int)
        cycle_iter = [(int(cycle_id), g) for cycle_id, g in df.groupby("_auto_cycle", sort=True)]

    spec = FeatureSpec(
        timestamp_col=args.timestamp_col,
        current_col=args.current_col or None,
        voltage_col=args.voltage_col or None,
        temp_col=args.temp_col or None,
        soc_col=args.soc_col or None,
        soh_proxy_col=args.soh_proxy_col or None,
        cc_slope_threshold=args.cc_slope_threshold,
        cv_slope_threshold=args.cv_slope_threshold,
        cv_quantile=args.cv_quantile,
    )

    rows = [compute_cycle_features(cid, cycle_df, spec) for cid, cycle_df in cycle_iter]
    feature_df = pd.DataFrame(rows)

    if args.soh_proxy_col and args.soh_proxy_col in df.columns:
        proxy_vals = pd.to_numeric(df[args.soh_proxy_col], errors="coerce")
        proxy_by_cycle: dict[int, float] = {}
        if args.cycle_col:
            for cid, g in df.groupby(args.cycle_col, sort=True):
                proxy_by_cycle[int(cid)] = float(_finite(proxy_vals.loc[g.index]).iloc[-1])
        else:
            for cid, g in df.groupby("_auto_cycle", sort=True):
                proxy_by_cycle[int(cid)] = float(_finite(proxy_vals.loc[g.index]).iloc[-1])
        feature_df["soh_proxy_from_input"] = feature_df["cycle_id"].map(proxy_by_cycle).astype(float)
        base = feature_df["soh_proxy_from_input"].dropna()
        if not base.empty and np.isfinite(base.iloc[0]) and base.iloc[0] != 0:
            feature_df["soh_proxy"] = feature_df["soh_proxy_from_input"] / float(base.iloc[0])

    if "soh_proxy" in feature_df.columns and len(feature_df) >= 5:
        feature_df["soh_trend_5"] = feature_df["soh_proxy"].rolling(5).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 5 else np.nan,
            raw=False,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output, index=False)

    if args.stats_output:
        summary: dict[str, Any] = {
            "rows": int(len(feature_df)),
            "features": len(feature_df.columns) - 1,
            "nan_rates": feature_df.isna().mean().to_dict(),
            "columns": list(feature_df.columns),
            "cycles": int(len(feature_df)),
        }
        Path(args.stats_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote features to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
