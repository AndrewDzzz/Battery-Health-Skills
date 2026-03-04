#!/usr/bin/env python3
"""Generate a ranked battery security risk register from assets and threat profiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


DEFAULT_THREATS = [
    {"threat": "Sensor data spoofing", "likelihood": 4, "impact": 5, "owner_hint": "Controls + Data"},
    {"threat": "Replay injection", "likelihood": 3, "impact": 4, "owner_hint": "Telemetry"},
    {"threat": "Timestamp manipulation", "likelihood": 3, "impact": 4, "owner_hint": "Telemetry"},
    {"threat": "OTA downgrade or fake image", "likelihood": 2, "impact": 5, "owner_hint": "Firmware"},
    {"threat": "Secrets leakage", "likelihood": 2, "impact": 5, "owner_hint": "Platform Security"},
    {"threat": "Training data poisoning", "likelihood": 3, "impact": 5, "owner_hint": "MLOps"},
    {"threat": "Model theft", "likelihood": 2, "impact": 4, "owner_hint": "AI Platform"},
    {"threat": "Prompt/input injection", "likelihood": 3, "impact": 3, "owner_hint": "Model Service"},
    {"threat": "Inference DoS", "likelihood": 4, "impact": 3, "owner_hint": "Platform Security"},
    {"threat": "Insider over-privilege", "likelihood": 2, "impact": 4, "owner_hint": "Governance"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ranked security audit artifacts.")
    parser.add_argument("--assets", default="", help="Comma-separated asset names")
    parser.add_argument(
        "--assets-csv",
        default="",
        help="Optional CSV with columns: asset,owner,criticality",
    )
    parser.add_argument(
        "--threat-profile",
        default="",
        help="Optional YAML/JSON with list of threat dicts {threat, likelihood, impact, owner_hint}",
    )
    parser.add_argument("--output-dir", default="./out", help="Output directory")
    return parser.parse_args()


def load_assets(args: argparse.Namespace) -> list[dict]:
    assets = []
    if args.assets_csv:
        with open(args.assets_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assets.append(
                    {
                        "asset": row.get("asset", "").strip(),
                        "owner": row.get("owner", "unassigned").strip() or "unassigned",
                        "criticality": int(row.get("criticality", "3")),
                    }
                )
    else:
        for asset in [a.strip() for a in args.assets.split(",") if a.strip()]:
            assets.append({"asset": asset, "owner": "unassigned", "criticality": 3})
    return assets


def load_threats(path: str) -> list[dict]:
    if not path:
        return DEFAULT_THREATS
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if isinstance(payload, dict):
        payload = payload.get("threats", payload)
    if not isinstance(payload, list):
        raise ValueError("Threat profile must be list-like")
    return payload


def build_rows(assets: list[dict], threats: list[dict]) -> list[dict]:
    rows = []
    for asset in assets:
        for threat in threats:
            likelihood = int(threat.get("likelihood", 3))
            impact = int(threat.get("impact", 3))
            criticality = int(asset.get("criticality", 3))
            score = likelihood * impact * max(1, criticality - 1)
            rows.append(
                {
                    "asset": asset["asset"],
                    "owner": asset["owner"],
                    "threat": str(threat["threat"]),
                    "likelihood": likelihood,
                    "impact": impact,
                    "criticality": criticality,
                    "risk_score": score,
                    "recommended_owner": threat.get("owner_hint", asset["owner"]),
                    "status": "open",
                }
            )
    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    return rows


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out = out_dir / "risk_register.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["asset", "owner", "threat", "likelihood", "impact", "criticality", "risk_score", "recommended_owner", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_out = out_dir / "risk_register.md"
    md_out.write_text(
        "\n".join(
            [
                "# Battery Security Risk Register",
                "",
                "| Asset | Threat | Likelihood | Impact | Criticality | Risk Score | Owner | Recommended Owner | Status |",
                "|---|---|---:|---:|---:|---:|---|---|---|",
            ]
            + [
                f"| {r['asset']} | {r['threat']} | {r['likelihood']} | {r['impact']} | {r['criticality']} | {r['risk_score']} | {r['owner']} | {r['recommended_owner']} | {r['status']} |"
                for r in rows
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    high = [r for r in rows if r["risk_score"] >= 40]
    controls = []
    for r in rows[:20]:
        controls.append(
            {
                "asset": r["asset"],
                "threat": r["threat"],
                "owner": r["recommended_owner"],
                "action": "Define control + validation proof + due date",
                "due_by": "TBD",
            }
        )

    control_out = out_dir / "controls_plan.md"
    control_out.write_text(
        "\n".join(
            [
                "# Battery Security Controls Plan",
                "",
                "| Asset | Threat | Owner | Action | Due By |",
                "|---|---|---|---|---|",
            ]
            + [
                f"| {c['asset']} | {c['threat']} | {c['owner']} | {c['action']} | {c['due_by']} |"
                for c in controls
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "total_rows": len(rows),
        "high_risk_count": len(high),
        "top_risk_scores": sorted({r["risk_score"] for r in rows}, reverse=True)[:10],
        "top_ten": rows[:10],
    }
    (out_dir / "risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    assets = load_assets(args)
    if not assets:
        raise ValueError("No assets found. Provide --assets or --assets-csv.")
    threats = load_threats(args.threat_profile)
    rows = build_rows(assets, threats)
    write_outputs(rows, Path(args.output_dir))
    print(f"Wrote {len(rows)} risk rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

