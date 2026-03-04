#!/usr/bin/env python3
"""Integrity checks for battery telemetry streams."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run telemetry integrity checks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--time-col", required=True)
    parser.add_argument("--id-col", default="")
    parser.add_argument("--voltage-col", default="")
    parser.add_argument("--current-col", default="")
    parser.add_argument("--temp-col", default="")
    parser.add_argument("--output-dir", default="./telemetry-integrity")
    parser.add_argument("--voltage-jump-threshold", type=float, default=3.0)
    parser.add_argument("--current-jump-threshold", type=float, default=25.0)
    parser.add_argument("--temp-jump-threshold", type=float, default=8.0)
    parser.add_argument("--flatline-window", type=int, default=20)
    parser.add_argument("--gap-multiplier", type=float, default=3.0)
    return parser.parse_args()


def _hash_row_signature(row: pd.Series) -> str:
    payload = [str(v) for v in row.to_numpy() if pd.notna(v)]
    return "|".join(payload)


def run_checks(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    time_col = args.time_col
    id_col = args.id_col or "__single_device__"
    if id_col == "__single_device__":
        df[id_col] = "all"

    required = [time_col]
    for col in [args.voltage_col, args.current_col, args.temp_col]:
        if col:
            required.append(col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    flags = []
    group_cols = [id_col]
    for gid, g in df.groupby(group_cols, sort=False):
        g = g.reset_index()
        # Duplicate rows by id+time
        dup_idx = set(g.duplicated(subset=[id_col, time_col], keep=False)[g.duplicated(subset=[id_col, time_col], keep=False)].index.tolist())
        # Timestamp reorder
        dt = g[time_col].diff().dt.total_seconds()
        ts_bad = set(g.index[(dt < 0).fillna(False)].tolist())
        # gaps
        median_dt = float(dt.abs().replace(0, np.nan).median(skipna=True) or 0.0)
        gap_mask = dt > args.gap_multiplier * median_dt if median_dt > 0 else pd.Series([False] * len(g))
        gap_bad = set(g.index[gap_mask.fillna(False)].tolist())

        # Jumps
        vol_bad = set()
        cur_bad = set()
        temp_bad = set()
        if args.voltage_col:
            delta = g[args.voltage_col].diff().abs()
            vol_bad = set(g.index[(delta > args.voltage_jump_threshold).fillna(False)].tolist())
        if args.current_col:
            delta = g[args.current_col].diff().abs()
            cur_bad = set(g.index[(delta > args.current_jump_threshold).fillna(False)].tolist())
        if args.temp_col:
            delta = g[args.temp_col].diff().abs()
            temp_bad = set(g.index[(delta > args.temp_jump_threshold).fillna(False)].tolist())

        # Flatline detection
        flat_bad = set()
        if args.voltage_col and args.current_col:
            v = g[args.voltage_col]
            c = g[args.current_col]
            vol_flat = (v == v.shift()).astype(int)
            cur_flat = (c == c.shift()).astype(int)
            run = 0
            for i in range(1, len(g)):
                if vol_flat.iloc[i] and cur_flat.iloc[i]:
                    run += 1
                    if run >= args.flatline_window:
                        flat_bad.add(i)
                else:
                    run = 0

        # Replay pattern via repeated signature
        sig = g.drop(columns=["index"]).apply(_hash_row_signature, axis=1)
        sig_counts = Counter(sig)
        replay_idx = set(sig[sig.map(sig_counts) > 2].index.tolist())

        for idx in g.index.tolist():
            reasons = []
            if idx in dup_idx:
                reasons.append("duplicate_row")
            if idx in ts_bad:
                reasons.append("timestamp_reorder")
            if idx in gap_bad:
                reasons.append("gap")
            if idx in vol_bad:
                reasons.append("voltage_jump")
            if idx in cur_bad:
                reasons.append("current_jump")
            if idx in temp_bad:
                reasons.append("temp_jump")
            if idx in flat_bad:
                reasons.append("flatline")
            if idx in replay_idx:
                reasons.append("replay_pattern")
            if reasons:
                row = g.loc[idx].to_dict()
                row["flagged"] = True
                row["reasons"] = ",".join(reasons)
                row["asset_id"] = gid
                flags.append(row)

    flag_df = pd.DataFrame(flags)
    severity = {
        "duplicate_row": int(flag_df["reasons"].str.contains("duplicate_row", na=False).sum()) if not flag_df.empty else 0,
        "timestamp_reorder": int(flag_df["reasons"].str.contains("timestamp_reorder", na=False).sum()) if not flag_df.empty else 0,
        "gap": int(flag_df["reasons"].str.contains("gap", na=False).sum()) if not flag_df.empty else 0,
        "voltage_jump": int(flag_df["reasons"].str.contains("voltage_jump", na=False).sum()) if not flag_df.empty else 0,
        "current_jump": int(flag_df["reasons"].str.contains("current_jump", na=False).sum()) if not flag_df.empty else 0,
        "temp_jump": int(flag_df["reasons"].str.contains("temp_jump", na=False).sum()) if not flag_df.empty else 0,
        "flatline": int(flag_df["reasons"].str.contains("flatline", na=False).sum()) if not flag_df.empty else 0,
        "replay_pattern": int(flag_df["reasons"].str.contains("replay_pattern", na=False).sum()) if not flag_df.empty else 0,
    }
    report = {
        "rows_total": int(len(df)),
        "flagged_rows": int(len(flag_df)),
        "severity": severity,
        "parameters": {
            "voltage_jump_threshold": args.voltage_jump_threshold,
            "current_jump_threshold": args.current_jump_threshold,
            "temp_jump_threshold": args.temp_jump_threshold,
            "flatline_window": args.flatline_window,
            "gap_multiplier": args.gap_multiplier,
        },
    }
    return flag_df, report


def write_outputs(flag_df: pd.DataFrame, report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flag_path = out_dir / "integrity_flags.csv"
    if flag_df.empty:
        pd.DataFrame(columns=["flagged"]).to_csv(flag_path, index=False)
    else:
        flag_df.to_csv(flag_path, index=False)

    (out_dir / "telemetry_integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    flagged = report["flagged_rows"]
    summary = [
        "# Telemetry Integrity Summary",
        "",
        f"- Total rows: {report['rows_total']}",
        f"- Flagged rows: {flagged}",
        "",
        "## Severity counts",
    ]
    for k, v in report["severity"].items():
        summary.append(f"- {k}: {v}")
    summary.append("")
    summary.append("Action: investigate flagged rows first; if `replay_pattern` exceeds zero, treat as possible tampering.")
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    flags, report = run_checks(df, args)
    write_outputs(flags, report, Path(args.output_dir))
    print(f"Wrote integrity output to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

