#!/usr/bin/env python3
"""Run a realistic battery-field workflow demo end-to-end."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run battery field workflow demo.")
    parser.add_argument("--telemetry-csv", default="", help="Input telemetry CSV")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic telemetry")
    parser.add_argument("--out-dir", default="./field-demo-run", help="Run output directory")
    parser.add_argument("--n-cells", type=int, default=8, help="Synthetic cells")
    parser.add_argument("--cycles-per-cell", type=int, default=20, help="Cycles per synthetic cell")
    parser.add_argument("--points-per-cycle", type=int, default=120, help="Points per cycle")
    parser.add_argument("--time-step", type=float, default=12.0, help="Synthetic time step in seconds")
    parser.add_argument("--seed", type=int, default=2026, help="Synthetic seed")
    parser.add_argument("--skip-integrity", action="store_true", help="Skip telemetry integrity step")
    parser.add_argument("--skip-security", action="store_true", help="Skip security audit step")
    parser.add_argument("--skip-soh", action="store_true", help="Skip SOH modeling step")
    parser.add_argument("--soh-label-col", default="soh_proxy", help="SOH label column in feature set")
    parser.add_argument("--security-assets", default="", help="Comma-separated security assets")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(cwd)},
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    root = _repo_root()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    telemetry_dir = out / "telemetry"
    soh_dir = out / "soh"
    security_dir = out / "security"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    soh_dir.mkdir(parents=True, exist_ok=True)
    security_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = Path(args.telemetry_csv).resolve() if args.telemetry_csv else telemetry_dir / "demo_telemetry.csv"
    if args.generate_data or not args.telemetry_csv:
        gen_script = root / "soh-field-demo" / "scripts" / "generate_ev_telemetry_demo.py"
        gen_cmd = [
            sys.executable,
            str(gen_script),
            "--output",
            str(telemetry_path),
            "--n-cells",
            str(args.n_cells),
            "--cycles-per-cell",
            str(args.cycles_per_cell),
            "--points-per-cycle",
            str(args.points_per_cycle),
            "--time-step",
            str(args.time_step),
            "--seed",
            str(args.seed),
        ]
        code, out_txt, err_txt = _run(gen_cmd, root)
        if code != 0:
            raise RuntimeError(f"generator failed: {err_txt}")

    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry input missing: {telemetry_path}")

    summary: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "telemetry_csv": str(telemetry_path),
        "steps": {},
    }

    if not args.skip_integrity:
        integrity_script = root / "battery-telemetry-integrity" / "scripts" / "check_telemetry_integrity.py"
        integrity_out = telemetry_dir / "telemetry_integrity"
        integrity_cmd = [
            sys.executable,
            str(integrity_script),
            "--input",
            str(telemetry_path),
            "--time-col",
            "timestamp",
            "--id-col",
            "cell_id",
            "--voltage-col",
            "voltage",
            "--current-col",
            "current",
            "--temp-col",
            "temperature",
            "--output-dir",
            str(integrity_out),
        ]
        code, out_txt, err_txt = _run(integrity_cmd, root)
        summary["steps"]["telemetry_integrity"] = {
            "returncode": code,
            "stdout": out_txt[-2000:],
            "stderr": err_txt[-2000:],
        }
        if code != 0:
            raise RuntimeError(f"integrity check failed: {err_txt}")

    feature_script = root / "soh-modeling-upgrade" / "scripts" / "extract_soh_features.py"
    feature_csv = soh_dir / "soh_features.csv"
    feature_cmd = [
        sys.executable,
        str(feature_script),
        "--input",
        str(telemetry_path),
        "--output",
        str(feature_csv),
        "--timestamp-col",
        "timestamp",
        "--cycle-col",
        "cycle_id",
        "--current-col",
        "current",
        "--voltage-col",
        "voltage",
        "--temp-col",
        "temperature",
        "--soc-col",
        "soc",
        "--soh-proxy-col",
        "capacity",
    ]
    code, out_txt, err_txt = _run(feature_cmd, root)
    summary["steps"]["soh_feature_extraction"] = {
        "returncode": code,
        "stdout": out_txt[-2000:],
        "stderr": err_txt[-2000:],
    }
    if code != 0:
        raise RuntimeError(f"SOH feature extraction failed: {err_txt}")

    if args.skip_soh:
        summary["steps"]["soh_train"] = {"returncode": 0, "status": "skipped", "reason": "skip-soh flag set"}
    else:
        # check for an available SOH target first so users can run feature extraction only
        import pandas as pd  # local import to avoid mandatory dependency at import time

        feature_preview = pd.read_csv(feature_csv)
        if args.soh_label_col not in feature_preview.columns:
            summary["steps"]["soh_train"] = {
                "returncode": 0,
                "status": "skipped",
                "reason": f"missing label column {args.soh_label_col}",
            }
        else:
            train_script = root / "soh-modeling-upgrade" / "scripts" / "train_soh_with_uncertainty.py"
            train_cmd = [
                sys.executable,
                str(train_script),
                "--input",
                str(feature_csv),
                "--soh-label-col",
                args.soh_label_col,
                "--output-dir",
                str(soh_dir),
            ]
            code, out_txt, err_txt = _run(train_cmd, root)
            summary["steps"]["soh_train"] = {
                "returncode": code,
                "stdout": out_txt[-2000:],
                "stderr": err_txt[-2000:],
            }
            if code != 0:
                raise RuntimeError(f"SOH model training failed: {err_txt}")

    if not args.skip_security:
        security_script = root / "battery-security-audit" / "scripts" / "generate_security_audit.py"
        security_cmd = [
            sys.executable,
            str(security_script),
            "--assets",
            args.security_assets or "CellTelemetry,ChargingGateway,InverterCloud,FeatureStore,ModelRegistry,InferenceAPI",
            "--output-dir",
            str(security_dir),
        ]
        code, out_txt, err_txt = _run(security_cmd, root)
        summary["steps"]["security_audit"] = {
            "returncode": code,
            "stdout": out_txt[-2000:],
            "stderr": err_txt[-2000:],
        }
        if code != 0:
            raise RuntimeError(f"security audit failed: {err_txt}")

    soh_metrics = _load_json(soh_dir / "soh_metrics.json")
    drift_summary = _load_json(soh_dir / "soh_drift_summary.json")
    integrity_report = _load_json(telemetry_dir / "telemetry_integrity" / "telemetry_integrity_report.json")
    risk_summary = _load_json(security_dir / "risk_summary.json")

    summary["artifacts"] = {
        "telemetry_integrity_report": str(telemetry_dir / "telemetry_integrity" / "telemetry_integrity_report.json") if not args.skip_integrity else None,
        "soh_features": str(feature_csv),
        "soh_metrics": str(soh_dir / "soh_metrics.json"),
        "soh_uncertainty": str(soh_dir / "soh_uncertainty.csv"),
        "security_risk_summary": str(security_dir / "risk_summary.json") if not args.skip_security else None,
    }
    summary["soh_metrics"] = soh_metrics
    summary["soh_drift_summary"] = drift_summary
    summary["telemetry_integrity"] = integrity_report
    summary["security_risk_summary"] = risk_summary
    summary["status"] = "pass" if all(s["returncode"] == 0 for s in summary["steps"].values()) else "fail"
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()

    out_summary = out / "demo_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote demo summary to {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
